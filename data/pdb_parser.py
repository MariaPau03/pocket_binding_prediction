from Bio.PDB import PDBParser
import numpy as np


class Atom:
    def __init__(self, coord, element, name, residue_id, residue_name, chain_id):
        self.coord = np.array(coord)
        self.element = element
        self.name = name
        self.residue_id = residue_id
        self.residue_name = residue_name
        self.chain_id = chain_id


class Residue:
    def __init__(self, residue_id, residue_name, chain_id):
        self.residue_id = residue_id
        self.residue_name = residue_name
        self.chain_id = chain_id
        self.atoms = []

    def add_atom(self, atom):
        # Add an Atom object to the residue's list of atoms
        self.atoms.append(atom)

    def get_center(self):
        # Calculate the geometric center of the residue by averaging the coordinates of its atoms
        coords = np.array([atom.coord for atom in self.atoms])
        return coords.mean(axis=0)


class Protein:
    def __init__(self, pdb_file, auto_load=True):
        self.pdb_file = pdb_file
        self.atoms = []
        self.ligand_atoms = []
        self.residues = []
        self._loaded = False

        if auto_load:
            self.load()

    def load(self):
        # Load the PDB file and parse its contents to populate the atoms, ligand_atoms, and residues lists
        # Reset containers to keep load() idempotent.
        self.atoms = []
        self.ligand_atoms = []
        self.residues = []

        # Use Biopython's PDBParser to read the structure from the PDB file
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", self.pdb_file)

        residue_dict = {}

        # Iterate through all models, chains, and residues in the structure
        for model in structure:
            for chain in model:
                for residue in chain:

                    hetflag = residue.id[0]
                    res_id = residue.id[1]
                    res_name = residue.resname
                    chain_id = chain.id

                    # PROTEIN
                    # Only consider standard amino acid residues (hetflag == " ") and ignore water molecules (hetflag == "W")
                    if hetflag == " ":
                        key = (chain_id, res_id)

                        if key not in residue_dict:
                            residue_dict[key] = Residue(res_id, res_name, chain_id)

                        for atom in residue:
                            atom_obj = Atom(
                                coord=atom.coord,
                                element=atom.element,
                                name=atom.name,
                                residue_id=res_id,
                                residue_name=res_name,
                                chain_id=chain_id
                            )

                            self.atoms.append(atom_obj)
                            residue_dict[key].add_atom(atom_obj)

                    # LIGAND
                    elif hetflag != "W":
                        for atom in residue:
                            atom_obj = Atom(
                                coord=atom.coord,
                                element=atom.element,
                                name=atom.name,
                                residue_id=res_id,
                                residue_name=res_name,
                                chain_id=chain_id
                            )

                            self.ligand_atoms.append(atom_obj)

        self.residues = list(residue_dict.values())
        self._loaded = True

    def get_atom_coordinates(self):
        # Return a numpy array of the coordinates of all atoms in the protein
        if not self._loaded:
            self.load()
        return np.array([atom.coord for atom in self.atoms])

    def get_residue_centers(self):
        # Return a numpy array of the geometric centers of all residues in the protein
        if not self._loaded:
            self.load()
        return np.array([res.get_center() for res in self.residues])
