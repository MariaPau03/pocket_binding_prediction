import numpy as np
import freesasa

class SASPointGenerator:
    def __init__(self, protein, neighbor_search):
        self.protein = protein
        self.neighbor_search = neighbor_search

    def generate_SAS(self, sasa_threshold=1.0, n_points=20, distance=1.5):
        """
        Generates solvent-accessible surface (SAS) points using FreeSASA and local sampling.
        1 - Identify surface atoms based on SASA threshold.
        2 - Generate a cloud of points around each surface atom.
        """
        # 1 - Identify surface atoms robustly
        surface_atoms = self._get_surface_atoms(sasa_threshold)

        sas_points = []
        for atom in surface_atoms:
            # 2 - Generate points around each surface atom
            points = self._generate_points_around_atom(atom, n_points, distance)
            sas_points.extend(points)

        return np.array(sas_points)
    
    def _get_surface_atoms(self, sasa_threshold):
        """
        Identify surface atoms using FreeSASA by passing coordinates and radii directly, avoiding index mismatches.
        """
        # Prepare the coordinate and radius lists for FreeSASA
        coords = []
        radii = []
        
        for atom in self.protein.atoms:
            # coords need to be a flat list [x1, y1, z1, x2, y2, z2...]
            coords.extend(atom.coord)
            # Asign Van der Waals radii based on element type (simplified)
            # (1.7Å is a safe average for C, N, O)
            radii.append(1.7)

        # calcCoord is used instead of calc(structure) to directly compute SASA from coordinates and radii, avoiding any internal parsing issues.
        # returns a result object that contains the computed SASA for each atom
        result = freesasa.calcCoord(coords, radii)
        
        surface_atoms = []
        # Check the SASA for each atom and select those above the threshold
        for i in range(len(self.protein.atoms)):
            # sasa is the solvent-accessible surface area for the i-th atom, obtained from the result object
            sasa = result.atomArea(i)
            if sasa > sasa_threshold:
                surface_atoms.append(self.protein.atoms[i])

        return surface_atoms

    def _generate_points_around_atom(self, atom, n_points, distance):
        """
        Generate points on a sphere around an atom, ensuring they are accessible (not inside the protein).
        """
        points = []
        for _ in range(n_points):
            # Random direction in 3D space
            direction = np.random.normal(size=3)
            norm = np.linalg.norm(direction)
            if norm == 0: continue
            direction /= norm

            # Place the point at a fixed distance from the atom in the random direction
            point = atom.coord + direction * distance

            # Check if the point is accessible (not inside the protein)
            if self._is_accessible(point):
                points.append(point)
        return points
    
    def _is_accessible(self, point, min_distance=1.2):
        """
        Check if a point is accessible (not inside the protein).
        """
        # Query the neighbor search for atoms within min_distance of the point
        neighbors = self.neighbor_search.query(point, radius=min_distance)
        return len(neighbors) == 0