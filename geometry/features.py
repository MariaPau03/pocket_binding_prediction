# geometry/features.py
import numpy as np

# ── Hydrophobicity scale (Kyte-Doolittle) ─────────────────────────────────────
# Higher = more hydrophobic. Used to score the local chemical environment.
# Used for identify both surface-exposed regions as well as transmembrane regions. 
# Source: https://resources.qiagenbioinformatics.com/manuals/clcgenomicsworkbench/650/Hydrophobicity_scales.html
HYDROPHOBICITY = {
    "ALA":  1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS":  2.5,
    "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE":  4.5,
    "LEU":  3.8, "LYS": -3.9, "MET":  1.9, "PHE":  2.8, "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL":  4.2,
}

# ── Formal charge at pH 7 ─────────────────────────────────────────────────────
CHARGE = {
    "ARG": +1, "LYS": +1, "HIS": +0.1,   # positive
    "ASP": -1, "GLU": -1,                  # negative
}

# ── H-bond donors (atoms that donate H) ──────────────────────────────────────
HBOND_DONORS = {"N", "NH1", "NH2", "NZ", "NE", "ND1", "ND2", "NE2",
                 "OG", "OG1", "OH", "NE1"}

# ── H-bond acceptors (atoms that accept H) ───────────────────────────────────
HBOND_ACCEPTORS = {"O", "OD1", "OD2", "OE1", "OE2", "OH",
                    "OG", "OG1", "ND1", "NE2", "SD"}

# ── Canonical amino acids for one-hot encoding (20 aa) ───────────────────────
AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]

#creates a lookup dictionary mapping each aa to its index (0-19) for one-hot encoding
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


class FeatureExtractor:
    def __init__(self, protein, neighbor_search, radius=10.0):
        """
        Stores:
        - protein         : Protein object from pdb_parser.py
        - neighbor_search : NeighborSearch object from neighbors.py
        - radius          : local environment radius in Å (default 10 Å)
        """
        self.protein = protein
        self.neighbor_search = neighbor_search
        self.radius = radius
        # Precompute protein centroid once — used for depth feature
        all_coords = np.array([a.coord for a in protein.atoms])
        self.centroid = all_coords.mean(axis=0)

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHOD
    # ─────────────────────────────────────────────────────────────────────────

    def extract(self, sas_point): 
        """
        Given a single SAS point (3D numpy array), return a 1D feature vector.

        Output shape: (N_features,) — concatenation of geometry + physicochemical
        """
        neighbor_idx = self.neighbor_search.query(sas_point, self.radius)
        neighbors = [self.protein.atoms[i] for i in neighbor_idx]

        geo  = self._geometry_features(sas_point, neighbors)
        phys = self._physicochemical_features(neighbors)

        return np.concatenate([geo, phys])

    def extract_all(self, sas_points):
        """
        Convenience method: extract features for all SAS points at once.
        Returns a 2D matrix of shape (N_points, N_features).
        """
        return np.array([self.extract(pt) for pt in sas_points])

    # ─────────────────────────────────────────────────────────────────────────
    # GROUP 1 — GEOMETRY  (7 features)
    # ─────────────────────────────────────────────────────────────────────────

    def _geometry_features(self, point, neighbors):
        """
        Captures the shape and spatial density of the local environment.

        Features:
          [0]   neighbor_count     — how many atoms are within radius
          [1]   mean_distance      — average distance to neighbors
          [2]   std_distance       — spread of distances (pocket = low spread)
          [3]   min_distance       — distance to the closest atom
          [4]   density            — atoms per unit volume of the sphere
          [5]   depth              — distance from SAS point to protein centroid
          [6]   depth_norm         — depth normalized by protein radius
        """
        if len(neighbors) == 0:
            # Return zeros if no neighbors found (edge case)
            return np.zeros(7)

        # Distances from the SAS point to each neighboring atom
        coords = np.array([a.coord for a in neighbors])
        dists = np.linalg.norm(coords - point, axis=1)

        # [0] Total neighbor count
        neighbor_count = float(len(neighbors))

        # [1] Mean distance to neighbors
        mean_dist = float(np.mean(dists))

        # [2] Std of distances — low std = tight pocket, high std = flat surface
        std_dist = float(np.std(dists))

        # [3] Minimum distance — how close is the nearest atom
        min_dist = float(np.min(dists))

        # [4] Local density — atoms per Å³ inside the query sphere
        sphere_volume = (4 / 3) * np.pi * (self.radius ** 3)
        density = neighbor_count / sphere_volume

        # [5] Depth — distance of the SAS point from the protein's centroid
        #     Points deep inside cavities are close to the centroid
        depth = float(np.linalg.norm(point - self.centroid))

        # [6] Normalized depth — divide by max possible depth (protein radius)
        all_coords = np.array([a.coord for a in self.protein.atoms])
        protein_radius = float(np.max(
            np.linalg.norm(all_coords - self.centroid, axis=1)
        ))
        depth_norm = depth / protein_radius if protein_radius > 0 else 0.0

        return np.array([
            neighbor_count, mean_dist, std_dist,
            min_dist, density, depth, depth_norm
        ])

    # ─────────────────────────────────────────────────────────────────────────
    # GROUP 2 — PHYSICOCHEMICAL  (26 features)
    # ─────────────────────────────────────────────────────────────────────────

    def _physicochemical_features(self, neighbors):
        """
        Captures the chemical character of the residues surrounding the SAS point.

        Features:
          [0]     mean_hydrophobicity  — average Kyte-Doolittle score
          [1]     std_hydrophobicity   — spread (mixed = potential interface)
          [2]     total_charge         — net formal charge of local residues
          [3]     n_positive           — count of positively charged residues
          [4]     n_negative           — count of negatively charged residues
          [5]     n_hbond_donors       — count of H-bond donor atoms
          [6]     n_hbond_acceptors    — count of H-bond acceptor atoms
          [7-26]  one_hot_aa           — fraction of each of the 20 amino acids
                                         in the local neighborhood
        """
        if len(neighbors) == 0:
            return np.zeros(27)

        # ── Hydrophobicity ────────────────────────────────────────────────────
        hydro_scores = [
            HYDROPHOBICITY.get(a.residue_name, 0.0) for a in neighbors
        ]
        mean_hydro = float(np.mean(hydro_scores))
        std_hydro  = float(np.std(hydro_scores))

        # ── Charge ────────────────────────────────────────────────────────────
        # Use unique residues to avoid counting a residue multiple times
        # (a residue has many atoms but one charge)
        seen_residues = set()
        charge_values = []
        n_positive = 0
        n_negative = 0

        for atom in neighbors:
            key = (atom.chain_id, atom.residue_id)
            if key not in seen_residues:
                seen_residues.add(key)
                c = CHARGE.get(atom.residue_name, 0)
                charge_values.append(c)
                if c > 0:
                    n_positive += 1
                elif c < 0:
                    n_negative += 1

        total_charge = float(sum(charge_values))
        n_positive   = float(n_positive)
        n_negative   = float(n_negative)

        # ── H-bond donors and acceptors ───────────────────────────────────────
        # Count at the atom level — atom name (e.g. "NH1") determines role
        n_donors    = float(sum(1 for a in neighbors if a.name in HBOND_DONORS))
        n_acceptors = float(sum(1 for a in neighbors if a.name in HBOND_ACCEPTORS))

        # ── Amino acid composition (one-hot fraction) ─────────────────────────
        # Instead of a hard one-hot, we use the *fraction* of each aa type
        # among neighboring residues — more informative for variable-size windows
        aa_counts = np.zeros(20)
        seen_for_aa = set()

        for atom in neighbors:
            key = (atom.chain_id, atom.residue_id)
            if key not in seen_for_aa and atom.residue_name in AA_INDEX:
                seen_for_aa.add(key)
                aa_counts[AA_INDEX[atom.residue_name]] += 1

        total_residues = aa_counts.sum()
        aa_fractions = aa_counts / total_residues if total_residues > 0 else aa_counts

        return np.concatenate([
            [mean_hydro, std_hydro, total_charge,
             n_positive, n_negative, n_donors, n_acceptors],
            aa_fractions   # 20 values
        ])  # total: 7 + 20 = 27 features