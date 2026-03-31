"""
main.py — Entry point for the binding-site prediction pipeline.

Pipeline:
  1. Parse PDB → Protein object
  2. Build NeighborSearch (KD-tree)
  3. Generate SAS points (FreeSASA + local sampling)
  4. Extract features for every SAS point
  5. Label points (requires ligand in PDB; skipped in predict-only mode)
  6. Train / load a RandomForest model
  7. Cluster high-probability SAS points → pockets
  8. Write outputs:
       • CSV listing residues per pocket   (output/<name>_residues.csv)
       • Visualization PDB for PyMOL/Chimera (output/<name>_pockets.pdb)
"""

import os
import csv
import argparse
import numpy as np
import joblib


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, f1_score


# ── Project modules ────────────────────────────────────────────────────────────
from data.pdb_parser import Protein
from geometry.neighbors import NeighborSearch
from geometry.sas import SASPointGenerator
from geometry.features import FeatureExtractor
from model.labels import LabelGenerator
from output.pocket_writer import PocketWriter
from visualization.visualize import visualize_clusters

# ══════════════════════════════════════════════════════════════════════════════
# 1. PROTEIN PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_protein(pdb_path, pssm_path=None):
    """
    Parse a PDB file and compute per-point features + labels.

    Returns
    -------
    X          : np.ndarray (N, F)  — feature matrix
    y          : np.ndarray (N,)    — binary labels (1 = binding, 0 = not)
                 All zeros when no ligand is present in the structure.
    sas_points : np.ndarray (N, 3) — 3-D coordinates of every SAS point
    protein    : Protein            — parsed protein object (needed by writers)
    ns         : NeighborSearch     — KD-tree built on protein atoms
    """
    print(f"\n[1] Parsing: {pdb_path}")
    protein = Protein(pdb_path)
    print(f"    {len(protein.atoms)} protein atoms | "
          f"{len(protein.ligand_atoms)} ligand atoms | "
          f"{len(protein.residues)} residues")
    
    # -------------------------
    # Build KDTree
    # -------------------------
    print("[2] Building neighbour search …")
    atom_coords = protein.get_atom_coordinates()
    ns = NeighborSearch(atom_coords)

    # -------------------------
    # Generate SAS points
    # -------------------------
    print("[3] Generating SAS points …")
    sas_gen    = SASPointGenerator(protein, ns)
    sas_points = sas_gen.generate_SAS()
    print(f"    {len(sas_points)} SAS points generated")

    # -------------------------
    # Features extraction
    # -------------------------
    print("[4] Extracting features …")
    extractor = FeatureExtractor(protein, ns)
    if pssm_path and os.path.exists(pssm_path):
        extractor.load_pssm(pssm_path)
    X = extractor.extract_all(sas_points)

    # -------------------------
    # Label generation
    # -------------------------
    print("[5] Labelling points …")
    labeller = LabelGenerator(protein, ns)
    y = labeller.label_all(sas_points)
    n_binding = int(y.sum())
    print(f"    Binding: {n_binding} / {len(y)}  "
          f"({'no ligand found' if n_binding == 0 else 'OK'})")

    return X, y, sas_points, protein, ns


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLUSTERING  →  POCKET OBJECTS
# ══════════════════════════════════════════════════════════════════════════════

def cluster_points(sas_points, probabilities, threshold=0.3,
                   eps=4.0, min_samples=5):
    """
    Cluster SAS points with probability >= threshold using DBSCAN.

    Returns
    -------
    pockets : list of dicts, each with keys:
        'center'     np.ndarray (3,)
        'size'       int
        'score'      float   (mean probability of cluster members)
        'points'     np.ndarray (K, 3)  — SAS points in this cluster
    """
    mask = probabilities >= threshold
    candidate_points = sas_points[mask]
    candidate_probs  = probabilities[mask]

    if len(candidate_points) == 0:
        return []

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(candidate_points)

    pockets = []
    for label in set(labels):
        if label == -1:          # noise
            continue
        idx    = labels == label
        pts    = candidate_points[idx]
        probs  = candidate_probs[idx]
        pockets.append({
            "center": pts.mean(axis=0),
            "size":   int(idx.sum()),
            "score":  float(probs.mean()),
            "points": pts,
        })

    # Sort by score descending
    pockets.sort(key=lambda p: p["score"], reverse=True)
    return pockets


# ══════════════════════════════════════════════════════════════════════════════
# 3. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_evaluation_log(output_csv):
    """Create/clear the evaluation CSV file for the current run."""
    if not output_csv:
        return
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(output_csv):
        os.remove(output_csv)


def save_evaluation(metrics, output_csv):
    """Append one evaluation summary row to a CSV file."""
    if not output_csv or metrics is None:
        return

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.exists(output_csv)
    fieldnames = [
        "label", "threshold", "n_samples", "n_binding_points",
        "accuracy", "precision", "recall", "f1_score", "auc",
        "tp", "fp", "tn", "fn",
        "min_prob", "mean_prob", "max_prob",
    ]

    with open(output_csv, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def evaluate_scores(probs, y_test, threshold=0.1, label="TEST SET", output_csv=None):
    """Evaluate probability scores against ground-truth labels and print metrics."""
    if probs is None or y_test is None or len(y_test) == 0:
        print(f"\n===== {label} EVALUATION =====")
        print("No evaluation data available.")
        return None, None

    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else None
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

    metrics = {
        "label": label,
        "threshold": float(threshold),
        "n_samples": int(len(y_test)),
        "n_binding_points": int(np.sum(y_test)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "auc": "" if auc is None else float(auc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "min_prob": float(np.min(probs)),
        "mean_prob": float(np.mean(probs)),
        "max_prob": float(np.max(probs)),
    }

    print(f"\n===== {label} EVALUATION =====")
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    if auc is not None:
        print(f"AUC:       {auc:.4f}")
    else:
        print("AUC:       n/a (only one class present in ground truth)")

    print(f"\nTP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print("\nScore stats:")
    print(f"Min prob:  {np.min(probs):.3f}")
    print(f"Mean prob: {np.mean(probs):.3f}")
    print(f"Max prob:  {np.max(probs):.3f}")

    save_evaluation(metrics, output_csv)
    return preds, metrics


def evaluate_model(model, X_test, y_test, threshold=0.1, label="TEST SET", output_csv=None):
    """Predict probabilities with a model and print evaluation metrics."""
    probs = model.predict_proba(X_test)[:, 1]
    preds, metrics = evaluate_scores(
        probs,
        y_test,
        threshold=threshold,
        label=label,
        output_csv=output_csv,
    )
    return probs, preds, metrics


# ══════════════════════════════════════════════════════════════════════════════
# 4. OUTPUTS  (residue list + visualization PDB)
# ═════════════════════════════════════════════════════════════════════════════

def write_outputs(pockets, protein, ns, protein_name, output_dir="output", results_dir="results"):
    if not pockets:
        print("\n[!] No pockets to write.")
        return

    csv_dir = os.path.join(output_dir, "csv")
    pdb_dir = os.path.join(output_dir, "pockets")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(pdb_dir, exist_ok=True)

    writer = PocketWriter(protein, ns)

    pocket_centers        = [p["center"] for p in pockets]
    sas_pts_per_pocket    = [p["points"] for p in pockets]
    pocket_scores         = [p["score"]  for p in pockets]  # ← add this

    # ── 6a. CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(csv_dir, f"{protein_name}_residues.csv")
    print(f"\n[6a] Writing residue list → {csv_path}")
    writer.write_residues_csv(pocket_centers, csv_path)

    # ── 6b. Combined PDB for PyMOL/ChimeraX ──────────────────────────────
    pdb_path = os.path.join(pdb_dir, f"{protein_name}_pockets.pdb")
    print(f"[6b] Writing visualization PDB → {pdb_path}")
    writer.write_visualization_pdb(pocket_centers, sas_pts_per_pocket, pdb_path)

    # ── 6c. Chimera format (one PDB per cluster + log) ────────────────────
    print(f"[6c] Writing Chimera format → {results_dir}/")
    writer.write_chimera_format(
        pockets               = pocket_centers,
        sas_points_per_pocket = sas_pts_per_pocket,
        scores                = pocket_scores,
        struct_id             = protein_name,
        results_dir           = results_dir
    )

    # ── 6d. Auto-visualize with ChimeraX ─────────────────────────────────
    print(f"[6d] Running ChimeraX visualization …")
    visualize_clusters(
        pdb_id      = protein_name,
        top_n       = 3,
        rotate      = False,
        results_dir = results_dir
    )

# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING MODE
# ══════════════════════════════════════════════════════════════════════════════

def train(
    pdb_files,
    pssm_dir=None,
    model_out="my_model.pkl",
    eval_threshold=0.1,
    eval_log_path=None,
):
    """
    Train a RandomForest on a list of PDB files that contain ligands.
    Saves the trained model to *model_out*.
    """
    all_X, all_y = [], []

    for pdb in pdb_files:
        name      = os.path.splitext(os.path.basename(pdb))[0]
        pssm_path = os.path.join(pssm_dir, f"{name}.pssm") if pssm_dir else None
        X, y, _, _, _ = process_protein(pdb, pssm_path)
        if X is not None:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        print("No data collected — aborting training.")
        return

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    print(f"\nTotal samples: {len(y)}  |  binding: {int(y.sum())}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    print("\n── Evaluation on held-out test split ──")
    _, preds, _ = evaluate_model(
        clf,
        X_te,
        y_te,
        threshold=eval_threshold,
        label="HELD-OUT TEST SET",
        output_csv=eval_log_path,
    )

    print("\nClassification report:")
    print(classification_report(y_te, preds, zero_division=0))

    if eval_log_path:
        print(f"Training evaluation saved → {eval_log_path}")

    joblib.dump(clf, model_out)
    print(f"Model saved → {model_out}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PREDICTION MODE
# ══════════════════════════════════════════════════════════════════════════════

def predict(
    pdb_path,
    model_path,
    pssm_dir=None,
    threshold=0.3,
    eval_threshold=0.1,
    output_dir="output",
    results_dir="results",
    eval_log_path=None,
):
    """
    Run the full prediction pipeline on a single PDB file and write outputs.
    Returns the predicted probabilities and labels for optional global evaluation.
    """
    if not os.path.exists(model_path):
        print(f"Error: model not found at {model_path}")
        return None, None

    model = joblib.load(model_path)
    print(f"Model loaded: {model_path}")

    name      = os.path.splitext(os.path.basename(pdb_path))[0]
    pssm_path = os.path.join(pssm_dir, f"{name}.pssm") if pssm_dir else None

    X, y, sas_points, protein, ns = process_protein(pdb_path, pssm_path)
    if X is None:
        return None, None

    print("[5] Predicting binding probabilities …")
    probs   = model.predict_proba(X)[:, 1]
    pockets = cluster_points(sas_points, probs, threshold=threshold)

    print(f"\n    Pockets found: {len(pockets)}")
    for i, p in enumerate(pockets[:5], 1):
        print(f"    Pocket {i}: score={p['score']:.3f}  size={p['size']}")

    evaluate_scores(
        probs,
        y,
        threshold=eval_threshold,
        label=f"{name} POINT-LEVEL",
        output_csv=eval_log_path,
    )
    write_outputs(pockets, protein, ns, name, output_dir, results_dir)
    return probs, y


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="Binding-pocket predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ── train ─────────────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Train a new model")
    tr.add_argument("pdb_dir",  help="Directory of training PDB files")
    tr.add_argument("--pssm_dir", default=None, help="Directory of .pssm files")
    tr.add_argument("--model_out", default="my_model.pkl",
                    help="Output path for the saved model")
    tr.add_argument("--eval_threshold", type=float, default=0.1,
                    help="Probability threshold used to compute evaluation scores")

    # ── predict ───────────────────────────────────────────────────────────────
    pr = sub.add_parser("predict", help="Predict pockets for a PDB file")
    pr.add_argument("pdb_path",  help="Path to a .pdb file or directory of .pdb files")
    pr.add_argument("--model",   default="my_model.pkl", help="Trained model (.pkl)")
    pr.add_argument("--pssm_dir", default=None, help="Directory of .pssm files")
    pr.add_argument("--threshold", type=float, default=0.3,
                    help="Minimum predicted probability to consider a point")
    pr.add_argument("--eval_threshold", type=float, default=0.1,
                    help="Probability threshold used to compute evaluation scores")
    pr.add_argument("--output_dir", default="output",
                    help="Directory for output files")
    pr.add_argument("--results_dir", default="results",
                help="Directory for chimera format outputs and screenshots")

    return p

# =========================================================
# MAIN
# =========================================================


def main():
    args = build_parser().parse_args()

    if args.mode == "train":
        pdb_files = [
            os.path.join(args.pdb_dir, f)
            for f in os.listdir(args.pdb_dir)
            if f.endswith(".pdb")
        ]
        if not pdb_files:
            print(f"No .pdb files found in {args.pdb_dir}")
            return

        eval_log_path = os.path.join("results", "logs", "train_evaluation.csv")
        prepare_evaluation_log(eval_log_path)

        train(
            pdb_files,
            pssm_dir=args.pssm_dir,
            model_out=args.model_out,
            eval_threshold=args.eval_threshold,
            eval_log_path=eval_log_path,
        )

    elif args.mode == "predict":
        # Accept a single file or a whole directory
        targets = []
        if os.path.isdir(args.pdb_path):
            targets = [
                os.path.join(args.pdb_path, f)
                for f in os.listdir(args.pdb_path)
                if f.endswith(".pdb")
            ]
        elif os.path.isfile(args.pdb_path):
            targets = [args.pdb_path]
        else:
            print(f"Path not found: {args.pdb_path}")
            return

        eval_log_path = os.path.join(args.results_dir, "logs", "prediction_evaluation.csv")
        prepare_evaluation_log(eval_log_path)

        all_probs = []
        all_y = []

        for pdb in sorted(targets):
            probs, y_true = predict(
                pdb_path       = pdb,
                model_path     = args.model,
                pssm_dir       = args.pssm_dir,
                threshold      = args.threshold,
                eval_threshold = args.eval_threshold,
                output_dir     = args.output_dir,
                results_dir    = args.results_dir,
                eval_log_path  = eval_log_path,
            )
            if probs is not None and y_true is not None:
                all_probs.append(probs)
                all_y.append(y_true)

        if all_probs and all_y:
            print("\n=== GLOBAL TEST-SET EVALUATION ===")
            evaluate_scores(
                np.concatenate(all_probs),
                np.concatenate(all_y),
                threshold=args.eval_threshold,
                label="GLOBAL TEST SET",
                output_csv=eval_log_path,
            )
            print(f"Evaluation summary saved → {eval_log_path}")

        print("\nDone.")


if __name__ == "__main__":
    main()
