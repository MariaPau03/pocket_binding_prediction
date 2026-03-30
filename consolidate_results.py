# consolidate_results.py
#
# Reads all cluster PDB files from results/pdbs/ and writes a single
# consolidated CSV with one row per residue, including:
#   protein, pocket_id, chain, residue_name, residue_id, x, y, z
#
# Usage:
#   python consolidate_results.py
#   python consolidate_results.py --pdbs_dir results/pdbs/ --output summary.csv

import os
import re
import argparse
import csv


def parse_cluster_filename(filename):
    """
    Extract protein name and pocket ID from a cluster PDB filename.

    Examples:
        121pcluster_1.pdb  →  protein='121p',  pocket_id=1
        1GUAcluster_2.pdb  →  protein='1GUA',  pocket_id=2
    """
    match = re.match(r"^(.+)cluster_(\d+)\.pdb$", filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def parse_pdb_residues(pdb_path):
    """
    Read ATOM records from a cluster PDB file.
    Returns a list of unique residues as dicts:
        { chain, residue_name, residue_id, x, y, z }
    Coordinates are the average position of all atoms in that residue.
    """
    residue_atoms = {}   # key: (chain, residue_id) → list of (x, y, z)
    residue_info  = {}   # key: (chain, residue_id) → (residue_name, chain)

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            try:
                residue_name = line[17:20].strip()
                chain        = line[21].strip()
                residue_id   = int(line[22:26].strip())
                x            = float(line[30:38].strip())
                y            = float(line[38:46].strip())
                z            = float(line[46:54].strip())
            except (ValueError, IndexError):
                continue

            key = (chain, residue_id)
            if key not in residue_atoms:
                residue_atoms[key] = []
                residue_info[key]  = residue_name

            residue_atoms[key].append((x, y, z))

    # Average coordinates across all atoms in each residue
    residues = []
    for (chain, residue_id), coords in residue_atoms.items():
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        residues.append({
            "chain":        chain,
            "residue_name": residue_info[(chain, residue_id)],
            "residue_id":   residue_id,
            "x":            round(sum(xs) / len(xs), 3),
            "y":            round(sum(ys) / len(ys), 3),
            "z":            round(sum(zs) / len(zs), 3),
        })

    # Sort by chain then residue number
    residues.sort(key=lambda r: (r["chain"], r["residue_id"]))
    return residues


def consolidate(pdbs_dir, output_path):
    """
    Walk through all cluster PDB files in pdbs_dir and write a
    single consolidated CSV.
    """
    # Collect all cluster PDB files
    cluster_files = sorted([
        f for f in os.listdir(pdbs_dir)
        if f.endswith(".pdb") and "cluster_" in f
    ])

    if len(cluster_files) == 0:
        print(f"No cluster PDB files found in {pdbs_dir}")
        return

    print(f"Found {len(cluster_files)} cluster file(s) in {pdbs_dir}")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "protein", "pocket_id", "chain",
            "residue_name", "residue_id", "x", "y", "z"
        ])
        writer.writeheader()

        total_rows = 0

        for filename in cluster_files:
            protein_name, pocket_id = parse_cluster_filename(filename)

            if protein_name is None:
                print(f"  [skip] Could not parse filename: {filename}")
                continue

            pdb_path = os.path.join(pdbs_dir, filename)
            residues = parse_pdb_residues(pdb_path)

            if len(residues) == 0:
                print(f"  [warn] No residues found in {filename}")
                continue

            for res in residues:
                writer.writerow({
                    "protein":      protein_name,
                    "pocket_id":    pocket_id,
                    "chain":        res["chain"],
                    "residue_name": res["residue_name"],
                    "residue_id":   res["residue_id"],
                    "x":            res["x"],
                    "y":            res["y"],
                    "z":            res["z"],
                })
                total_rows += 1

            print(f"  {filename}  →  {protein_name} pocket {pocket_id}: {len(residues)} residues")

    print(f"\nDone. {total_rows} rows written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate all cluster PDB files into a single CSV summary."
    )
    parser.add_argument(
        "--pdbs_dir", default="results/pdbs/",
        help="Directory containing cluster PDB files (default: results/pdbs/)"
    )
    parser.add_argument(
        "--output", default="results/all_binding_residues.csv",
        help="Output CSV file path (default: results/all_binding_residues.csv)"
    )
    args = parser.parse_args()

    consolidate(args.pdbs_dir, args.output)


if __name__ == "__main__":
    main()
