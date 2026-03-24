# Pocket Binding Prediction

A machine learning pipeline for predicting protein-ligand binding sites from PDB structures, inspired by the P2Rank approach. The pipeline extracts geometric, physicochemical and evolutionary features from protein surfaces and prepares them for training a Random Forest classifier.

---

## Project structure

```
pocket_binding_prediction/
├── data/
│   ├── *.pdb                   # input protein structures
│   ├── fastas/                 # generated FASTA sequences
│   ├── pssms/                  # generated conservation scores
│   ├── pdb_parser.py           # PDB loading and data model
│   └── prepro.py         # PDB → FASTA conversion
├── geometry/
│   ├── neighbors.py            # KDTree-based neighbor search
│   ├── sas.py                  # SAS point generation
│   └── features.py             # feature extraction (39 features)
├── evolution.py                # mock PSSM generator
└── main.py                     # pipeline entry point
```

---

## Installation

```bash
pip install biopython freesasa scipy numpy
```

---

## Usage

**Convert PDB --> Fasta:**
```bash
python prepro.py
```

Make sure that there's a PDB file inside the data folder

**Obtain the PSSM from Fasta file:**
```bash
python evolution.py
```

Will produce a CSV file of the PSSM

**Process a single PDB file:**
```bash
python main.py data/1GUA.pdb
```

**Process all PDB files in `data/` at once:**
```bash
python main.py
```

---

## Pipeline

The pipeline runs 5 sequential steps for each PDB file:

```
[1] Load protein         pdb_parser.py     → Protein object (atoms, residues, ligands)
[2] Build KDTree         neighbors.py      → Efficient spatial search index
[3] Generate SAS points  sas.py            → Surface point cloud
[4] Generate PSSM        evolution.py      → Conservation scores per residue
[5] Extract features     features.py       → Feature matrix (N_points × 39)
```

### Output files per protein (e.g. `1GUA`)

| File | Description |
|---|---|
| `data/fastas/1GUA.fasta` | Protein sequence in FASTA format |
| `data/pssms/1GUA.pssm` | Per-residue conservation scores |
| `data/1GUA_features.npy` | Feature matrix — input to the classifier |
| `sas_points_1GUA.pdb` | SAS points — visualizable in PyMOL or ChimeraX |

---

## Modules

### `data/pdb_parser.py`

Parses a `.pdb` file into three Python objects:

- `Atom` — stores 3D coordinates, element, name and residue info
- `Residue` — groups atoms and computes the geometric center
- `Protein` — top-level container that separates protein atoms from ligand atoms (water molecules are excluded)

```python
protein = Protein("data/1GUA.pdb")
protein.load()

protein.atoms         # list of Atom objects
protein.residues      # list of Residue objects
protein.ligand_atoms  # list of Atom objects (ligands only)
```

### `data/pdb_to_fasta.py`

Converts a PDB file to FASTA format using `Bio.PPBuilder`, which physically traces connected amino acids in the structure. Handles multi-chain proteins.

```python
pdb_to_fasta("data/1GUA.pdb", "data/fastas/")
# → data/fastas/1GUA.fasta
```

### `geometry/neighbors.py`

A thin wrapper around `scipy.spatial.KDTree`. Built once from all atom coordinates and used throughout the pipeline for efficient radius searches.

```python
neighbor_search = NeighborSearch(coords)
indices = neighbor_search.query(point, radius=10.0)
```

### `geometry/sas.py`

Generates the Solvent Accessible Surface (SAS) point cloud:

1. Uses `freesasa` (Shrake-Rupley algorithm) to identify exposed atoms (SASA > threshold)
2. Places random points around each surface atom at a fixed distance
3. Filters out any point that falls inside the protein (closer than 1.2 Å to any atom)

```python
generator  = SASPointGenerator(protein, neighbor_search)
sas_points = generator.generate_SAS()   # numpy array (N, 3)
```

### `geometry/features.py`

Extracts a 39-dimensional feature vector for each SAS point. Features are computed from the local atomic environment within a 10 Å radius.

| Group | Features | Count |
|---|---|---|
| Geometry | Neighbor count, mean/std/min distance, local density, depth, normalized depth | 7 |
| Physicochemical | Hydrophobicity (mean/std), charge (total/pos/neg), H-bond donors/acceptors, amino acid composition (fractions) | 27 |
| Advanced geometry | Surface curvature (PCA), pocket depth (ray casting), hydrophobic patch score, charge dipole magnitude | 4 |
| Evolutionary | PSSM conservation score of nearest residue | 1 |
| **Total** | | **39** |

```python
extractor = FeatureExtractor(protein, neighbor_search, radius=10.0)
extractor.load_pssm("data/pssms/1GUA.pssm")

feature_vector = extractor.extract(sas_point)       # shape: (39,)
feature_matrix = extractor.extract_all(sas_points)  # shape: (N, 39)
```

**Feature descriptions:**

- `neighbor_count` — number of protein atoms within 10 Å; pockets have higher counts than flat surfaces
- `mean_distance / std_distance` — low std indicates a tight enclosed pocket
- `density` — atoms per Å³; deep pockets are denser
- `depth / depth_norm` — distance to protein centroid normalized by protein radius
- `mean_hydrophobicity` — Kyte-Doolittle scale average over neighboring residues
- `total_charge / n_positive / n_negative` — net charge and charge counts of unique neighboring residues at pH 7
- `n_hbond_donors / n_hbond_acceptors` — atom-level counts of hydrogen bond participants
- `aa_fractions` — fraction of each of the 20 canonical amino acids among neighboring residues
- `curvature` — PCA eigenvalue ratio; high value = concave/curved (pocket-like)
- `pocket_depth` — fraction of outward rays blocked by protein atoms (0 = exposed, 1 = fully enclosed)
- `hydrophobic_patch_score` — inverse mean pairwise distance of hydrophobic atoms; high = clustered patch
- `charge_dipole` — magnitude of the spatial charge separation vector
- `conservation_score` — PSSM score of the nearest residue (0 = variable, 1 = conserved)

### `evolution.py`

Generates a mock PSSM CSV with random conservation scores. This is a placeholder for real PSI-BLAST output, designed so that the rest of the pipeline can be tested without configuring a BLAST server.

The output format is:
```
Residue_Index,Residue,Conservation_Score
1,M,0.8543
2,K,0.2341
...
```

To replace with real PSSM data, run PSI-BLAST and parse the output into the same CSV format — no other changes are needed.

---

## Feature lookup tables

All lookup tables are defined at module level in `features.py`:

- `HYDROPHOBICITY` — Kyte-Doolittle scale for all 20 amino acids
- `CHARGE` — formal charge at pH 7 (ARG/LYS +1, HIS +0.1, ASP/GLU -1)
- `HBOND_DONORS` — atom names that donate hydrogen bonds
- `HBOND_ACCEPTORS` — atom names that accept hydrogen bonds
- `AMINO_ACIDS / AA_INDEX` — ordered list and index map for one-hot encoding
- `HYDROPHOBIC_RESIDUES` — set of 9 hydrophobic residue names used for patch scoring

---

## Next steps

- [ ] **Labeling** — assign `1` (binding) / `0` (non-binding) to each SAS point using `protein.ligand_atoms`
- [ ] **Training** — train a `RandomForestClassifier` on the labeled feature matrix
- [ ] **Clustering** — group high-score SAS points into predicted pockets (DBSCAN)
- [ ] **Evaluation** — compute DCC, DVO, and AUC on benchmark datasets (COACH420, HOLO4K)
- [ ] **Real PSSM** — replace mock scores with PSI-BLAST output against UniRef90
- [ ] **ESM embeddings** — add ESM-2 residue embeddings as additional evolutionary features

---

## References

- Xia Y., Pan X., Shen H-B. (2024). *A comprehensive survey on protein-ligand binding site prediction*. Current Opinion in Structural Biology, 86, 102793.
- Krivák R., Hoksza D. (2018). *P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure*. J Cheminformatics, 10, 1–12.
- Jumper J. et al. (2021). *Highly accurate protein structure prediction with AlphaFold*. Nature, 596, 583–589.