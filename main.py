from data.pdb_parser import Protein
from geometry.neighbors import NeighborSearch
from geometry.sas import SASPointGenerator

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

if __name__ == "__main__":
    main()

# pip install:
# biopython
# freesasa
# scipy
# numpy