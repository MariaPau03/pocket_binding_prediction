from data.pdb_parser import Protein
from geometry.neighbors import NeighborSearch #imports the KDTree-based neighbor search
from geometry.sas import SASPointGenerator #imports the SAS point generator
from geometry.features import FeatureExtractor
import numpy as np

def save_sas_points(filename, points):
    with open(filename, "w") as f:
        for i, p in enumerate(points):
            f.write(
                f"HETATM{i:5d}  C   SAS A   1    "
                f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00           C\n"
            )

def main():
    pdb_file = "data/1GUA.pdb"  # <-- cambia esto por tu PDB

    print("Loading protein...")
    protein = Protein(pdb_file)
    protein.load()

    print(f"Number of atoms: {len(protein.atoms)}")
    print(f"Number of residues: {len(protein.residues)}")

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
    # Feature extraction
    # ---------------------------------------
    print("\nExtracting features...")
    extractor = FeatureExtractor(protein, neighbor_search, radius=10.0)
    feature_matrix = extractor.extract_all(sas_points)

    print(f"Feature matrix shape: {feature_matrix.shape}")
    # → (N_sas_points, 34)  — 7 geometry + 27 physicochemical

    # Save for later use in the Random Forest step
    np.save("features.npy", feature_matrix)
    print("Features saved to features.npy")

    


if __name__ == "__main__":
    main()

# pip install:
# biopython
# freesasa
# scipy
# numpy