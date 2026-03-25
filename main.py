from data.pdb_parser import Protein
from geometry.neighbors import NeighborSearch #imports the KDTree-based neighbor search
from geometry.sas import SASPointGenerator #imports the SAS point generator
from geometry.features import FeatureExtractor
from prepro import pdb_to_fasta
from evolution import mock_pssm_generator       
import numpy as np
import os
import argparse

def save_sas_points(filename, points):
    with open(filename, "w") as f:
        for i, p in enumerate(points):
            f.write(
                f"HETATM{i:5d}  C   SAS A   1    "
                f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00           C\n"
            )

def run_pipeline(pdb_file):
    """Run the full feature extraction pipeline for a single PDB file."""

    # Derive all paths from the input file — no hardcoded IDs
    struct_id  = os.path.basename(pdb_file).split(".")[0]   # e.g. "1GUA"
    fasta_dir  = "data/fastas/"
    pssm_dir   = "data/pssms/"
    fasta_file = os.path.join(fasta_dir, f"{struct_id}.fasta")
    pssm_file  = os.path.join(pssm_dir,  f"{struct_id}.pssm")
    sas_file   = f"sas_points_{struct_id}.pdb"
    feat_file  = f"data/{struct_id}_features.npy"

    os.makedirs(fasta_dir, exist_ok=True)
    os.makedirs(pssm_dir,  exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Processing: {struct_id}")
    print(f"{'='*50}")

    
    # ---------------------------------------
    # Load protein
    # ---------------------------------------
    print("\n[1/5] Loading protein...")
    protein = Protein(pdb_file)
    protein.load()
    print(f"  Atoms:        {len(protein.atoms)}")
    print(f"  Residues:     {len(protein.residues)}")
    print(f"  Ligand atoms: {len(protein.ligand_atoms)}")


    # ---------------------------------------
    # Build neighbor search (KDTree)
    # ---------------------------------------
    print("\nBuilding KDTree...")
    coords = protein.get_atom_coordinates()
    neighbor_search = NeighborSearch(coords)

    # Test neighbor query
    test_point = coords[0]
    neighbors = neighbor_search.query(test_point, radius=5.0)
    print(f"Neighbors within 5Å of first atom: {len(neighbors)}")

    # ---------------------------------------
    # Generate SAS points
    # ---------------------------------------
    print("\nGenerating SAS points...")
    sas_generator = SASPointGenerator(protein, neighbor_search)
    sas_points = sas_generator.generate_SAS()
    print(f"Number of SAS points: {len(sas_points)}")

    # ---------------------------------------
    # Basic sanity checks
    # ---------------------------------------
    if len(sas_points) == 0:
        print("WARNING: No SAS points generated!")
    else:
        print("SAS generation OK")

    print("\nPipeline working correctly!")

    save_sas_points("sas_points.pdb", sas_points)
    

    # ---------------------------------------
    # Generate FASTA + mock PSSM
    # ---------------------------------------
    print("\n[4/5] Generating FASTA and PSSM...")
    pdb_to_fasta(pdb_file, fasta_dir)           # → data/fastas/<struct_id>.fasta
    mock_pssm_generator(fasta_file, pssm_dir)   # → data/pssms/<struct_id>.pssm
    print(f"  FASTA: {fasta_file}")
    print(f"  PSSM:  {pssm_file}")

    # ---------------------------------------
    # Feature extraction
    # ---------------------------------------
    print("\nExtracting features...")
    extractor = FeatureExtractor(protein, neighbor_search, radius=10.0)
    extractor.load_pssm(pssm_file)
    feature_matrix = extractor.extract_all(sas_points)

    print(f"  Feature matrix shape: {feature_matrix.shape}")
    # → (N_sas_points, 39) — 7 geo + 27 phys + 4 curve + 1 evol

    np.save(feat_file, feature_matrix)
    print(f"  Saved to {feat_file}")

    return feature_matrix

def main():
    # ── Option A: pass a specific PDB file as argument ─────────────────────
    # Usage: python main.py data/1GUA.pdb
    # ── Option B: process all PDB files in data/ ───────────────────────────
    # Usage: python main.py

    parser = argparse.ArgumentParser(description="Pocket binding prediction pipeline")
    parser.add_argument(
        "pdb_file", nargs="?", default=None,
        help="Path to a single PDB file. If omitted, all PDBs in data/ are processed."
    )
    args = parser.parse_args()

    if args.pdb_file:
        argpath = args.pdb_file

        if os.path.isdir(argpath):
            # Directory mode – process every .pdb file in provided folder
            pdb_files = [
                os.path.join(argpath, f)
                for f in os.listdir(argpath)
                if f.endswith(".pdb")
            ]
            if len(pdb_files) == 0:
                print(f"No PDB files found in directory '{argpath}'")
                return

            print(f"Found {len(pdb_files)} PDB file(s) in '{argpath}': {[os.path.basename(f) for f in pdb_files]}")

            for pdb_file in sorted(pdb_files):
                run_pipeline(pdb_file)

            print("\nAll proteins processed.")

        elif os.path.isfile(argpath):
            # Single file mode
            run_pipeline(argpath)

        else:
            print(f"Error: path '{argpath}' does not exist.")
            print("Provide a valid PDB file path or a directory containing .pdb files, or run with no arguments to process data/.")
            return

    else:
        # Batch mode — process every .pdb file in data/
        pdb_files = [
            os.path.join("data", f)
            for f in os.listdir("data")
            if f.endswith(".pdb")
        ]
        if len(pdb_files) == 0:
            print("No PDB files found in data/")
            return

        print(f"Found {len(pdb_files)} PDB file(s): {[os.path.basename(f) for f in pdb_files]}")

        for pdb_file in sorted(pdb_files):
            run_pipeline(pdb_file)

        print("\nAll proteins processed.")


if __name__ == "__main__":
    main()

# pip install:
# biopython
# freesasa
# scipy
# numpy