from data.pdb_parser import Protein
from geometry.neighbors import NeighborSearch
from geometry.sas import SASPointGenerator
from geometry.features import FeatureExtractor
from prepro import pdb_to_fasta
from evolution import mock_pssm_generator
from model.labels import LabelGenerator

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.utils import resample
from sklearn.cluster import DBSCAN

import numpy as np
import os
import argparse
import joblib

# Directories to store intermediate files
FASTA_DIR = "data/fastas/"
PSSM_DIR = "data/pssms/"


# =========================================================
# DATA PROCESSING
# =========================================================

def process_protein(pdb_file):
    """
    Process one protein: - Load structure  - Generate SAS points  - Extract features  - Generate labels
    Return features (X) and labels (y) for one protein
    """

    # Extract unique identifier from filename (without extension)
    struct_id = os.path.splitext(os.path.basename(pdb_file))[0]

    print(f"\nProcessing {struct_id}")

    # -------------------------
    # Load protein
    # -------------------------
    protein = Protein(pdb_file)
    protein.load()

    # Skip proteins without ligands
    if len(protein.ligand_atoms) == 0:
        print("  Skipping (no ligand)")
        return None, None, None

    # -------------------------
    # Build KDTree
    # -------------------------
    coords = protein.get_atom_coordinates()     # Nx3 array of atom coordinates
    neighbor_search = NeighborSearch(coords)

    # -------------------------
    # Generate SAS points
    # -------------------------
    sas_generator = SASPointGenerator(protein, neighbor_search)
    sas_points = sas_generator.generate_SAS()

    # Skip if no surface points were generated
    if len(sas_points) == 0:
        print("  Skipping (no SAS points)")
        return None, None, None

    # -------------------------
    # Generate FASTA and PSSM
    # -------------------------
    fasta_file = os.path.join(FASTA_DIR, f"{struct_id}.fasta")
    pssm_file  = os.path.join(PSSM_DIR,  f"{struct_id}.pssm")

    # Ensure directories exist
    os.makedirs(FASTA_DIR, exist_ok=True)
    os.makedirs(PSSM_DIR, exist_ok=True)

    # Generate FASTA only if not already present
    if not os.path.exists(fasta_file):
        pdb_to_fasta(pdb_file, FASTA_DIR)

    # Generate mock PSSM (random conservation scores)
    if not os.path.exists(pssm_file):
        mock_pssm_generator(fasta_file, PSSM_DIR)

    # -------------------------
    # Features extraction
    # -------------------------
    extractor = FeatureExtractor(protein, neighbor_search, radius=10.0)
    extractor.load_pssm(pssm_file)  # Load conservation scores

    # Compute feature vector for each SAS point
    X = extractor.extract_all(sas_points)

    # -------------------------
    # Label generation
    # -------------------------
    label_gen = LabelGenerator(protein, neighbor_search)
    y = label_gen.label_all(sas_points)     # 1 = binding, 0 = non-binding

    # Skip if no positive samples (cannot train)
    if np.sum(y) == 0:
        print("  Skipping (no positive labels)")
        return None, None, None

    return X, y, sas_points


def load_dataset(directory):
    """Process all PDBs in a directory and build dataset"""

    # Collect all PDB files
    pdb_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".pdb")
    ]

    all_X, all_y = [], []

    for pdb_file in sorted(pdb_files):
        print(f"Processing {os.path.basename(pdb_file)}")

        # Process each protein independently -> get its features (X) and labels (y)
        X, y, _ = process_protein(pdb_file)

        if X is None:
            print("  Skipped")
            continue

        print(f"  Points: {len(y)} | Positives: {np.sum(y)}")

        # Accumulate data
        all_X.append(X)
        all_y.append(y)

    # If no valid proteins
    if len(all_X) == 0:
        return None, None

    # Concatenate all proteins into one dataset
    return np.vstack(all_X), np.concatenate(all_y)


# =========================================================
# BALANCING
# =========================================================

def balance_dataset(X, y):
    """Balance dataset by undersampling negatives"""

    # Separate positive and negative samples
    X_pos = X[y == 1]
    X_neg = X[y == 0]

    print(f"\nBefore balancing: pos={len(X_pos)}, neg={len(X_neg)}")

    # Randomly downsample negatives to match positives
    X_neg_down = resample(
        X_neg,
        replace=False,          # no repetition
        n_samples=len(X_pos),   # match number of positives
        random_state=42         # reproducibility
    )

    # Combine balanced dataset
    X_bal = np.vstack([X_pos, X_neg_down])
    y_bal = np.hstack([
        np.ones(len(X_pos)),
        np.zeros(len(X_neg_down))
    ])

    print(f"After balancing: {len(X_bal)} samples")

    return X_bal, y_bal


# =========================================================
# CLUSTERING (POCKETS)
# =========================================================

def cluster_points(sas_points, probs, threshold=0.1, eps=3.0, min_samples=5):
    """
    Cluster high-scoring SAS points into binding pockets
    """

    # Select only points with high probability -> ranking
    mask = probs > threshold
    selected_points = sas_points[mask]
    selected_scores = probs[mask]

    if len(selected_points) == 0:
        return []

    # Apply DBSCAN clustering in 3D space
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(selected_points)

    pockets = []

    # Iterate over clusters
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue

        pts = selected_points[labels == cluster_id]
        scores = selected_scores[labels == cluster_id]

        # Compute cluster center (mean coordinate)
        center = np.mean(pts, axis=0)
        # Pocket score (sum of squared scores)
        pocket_score = np.sum(scores ** 2)

        pockets.append({
            "center": center,
            "size": len(pts),
            "score": pocket_score
        })

    # Sort pockets by score
    pockets = sorted(pockets, key=lambda x: x["score"], reverse=True)

    return pockets


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate model and print metrics"""

    # Predict probabilities with the model 
    # The output is a ranking
    probs = model.predict_proba(X_test)[:, 1]

    # Convert to binary predictions using threshold
    threshold = 0.1
    preds = (probs > threshold).astype(int)

    # Compute metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    # Confusion matrix: TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    print("\n===== POINT-LEVEL EVALUATION =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    print(f"\nTP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")

    # Information about score distribution
    print("\nScore stats:")
    print(f"Max prob: {np.max(probs):.3f}")
    print(f"Mean prob: {np.mean(probs):.3f}")

    return probs


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate binding site predictor")
    parser.add_argument("train_dir", help="Training PDB directory")
    parser.add_argument("--test_dir", help="Test PDB directory (optional)")

    args = parser.parse_args()

    # -------------------------
    # TRAIN
    # -------------------------
    print("\n=== TRAINING ===")

    # Load training dataset
    X_train, y_train = load_dataset(args.train_dir)
    
    if X_train is None:
        print("No training data!")
        return

    print(f"\nTraining samples: {X_train.shape[0]}")

    # Balance dataset
    X_train, y_train = balance_dataset(X_train, y_train)

    # TRAIN Random Forest model
    model = RandomForestClassifier(
        n_estimators=200,   # number of trees
        n_jobs=-1           # use all CPU cores
    )

    model.fit(X_train, y_train)
    print("Model trained")

    # Save trained model
    joblib.dump(model, "rf_model.pkl")
    print("Model saved!")

    # -------------------------
    # TEST
    # -------------------------
    if args.test_dir:
        print("\n=== TESTING ===")

        pdb_files = [
            os.path.join(args.test_dir, f)
            for f in os.listdir(args.test_dir)
            if f.endswith(".pdb")
        ]

        all_probs = []
        all_y = []

        for pdb_file in pdb_files:
            print(f"\nTesting {os.path.basename(pdb_file)}")

            # Process test protein
            X_test, y_test, sas_points = process_protein(pdb_file)

            if X_test is None:
                continue
            
            # PREDICT scores
            probs = model.predict_proba(X_test)[:, 1]

            # ---- clustering ----
            pockets = cluster_points(sas_points, probs)

            print(f"Detected {len(pockets)} pockets")

            for i, p in enumerate(pockets[:3]):
                print(f"Pocket {i+1}: size={p['size']} score={p['score']:.2f}")

            # ---- per-protein evaluation ----
            evaluate_model(model, X_test, y_test)

            # ---- accumulate ----
            all_probs.append(probs)
            all_y.append(y_test)


        # =========================
        # GLOBAL EVALUATION
        # =========================
        print("\n\n=== GLOBAL EVALUATION ===")

        all_probs = np.concatenate(all_probs)
        all_y = np.concatenate(all_y)

        threshold = 0.1
        preds = (all_probs > threshold).astype(int)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

        acc = accuracy_score(all_y, preds)
        prec = precision_score(all_y, preds)
        rec = recall_score(all_y, preds)
        f1 = f1_score(all_y, preds)
        auc = roc_auc_score(all_y, all_probs)

        tn, fp, fn, tp = confusion_matrix(all_y, preds).ravel()

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC:       {auc:.4f}")

        print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")


if __name__ == "__main__":
    main()