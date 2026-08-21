"""Nearest Neighbor heuristic for TSP with a fixed start (index 0) and fixed end (index n-1).

Entry points:
  - NearestNeighborSolver.solve()     — run on a prebuilt NumPy matrix
  - NearestNeighborSolver.from_file() — load a CSV distance matrix and run

Standard and OSRM calculators call solve() via DistanceCalculator.compare().
"""

import os
import time

import numpy as np
import pandas as pd


class NearestNeighborSolver:
    """Nearest Neighbor TSP solver with a fixed start (index 0) and
    fixed end (index n-1). The algorithm is the same regardless of
    how the distance matrix is obtained."""

    def solve(
        self,
        matrix: np.ndarray,
        point_names: list[str],
        verbose: bool = True,
    ) -> tuple[list[str], float]:
        """Run the NN heuristic on a prebuilt NumPy distance matrix.

        Args:
            matrix:      NxN distance matrix (km).
            point_names: Labels for each point (length N).
            verbose:     Print progress and result.

        Returns:
            (path_names, total_distance) or (None, None) on failure.
        """
        start_time = time.time()
        n = matrix.shape[0]

        if n < 2:
            print("Error: Matrix must contain at least 2 points.")
            return None, None

        start_node_idx = 0
        end_node_idx = n - 1

        if verbose:
            print(
                f"Starting TSP-NN from '{point_names[start_node_idx]}' "
                f"to '{point_names[end_node_idx]}'."
            )
            print(f"Visiting {n - 2} intermediate points...")

        # Trivial case
        if n == 2:
            path_names = [point_names[0], point_names[1]]
            total_distance = matrix[0, 1]
            if verbose:
                print("Matrix has only 2 points. Path is direct start -> end.")
            return path_names, total_distance

        path_indices = [start_node_idx]
        total_distance = 0.0
        current_node_idx = start_node_idx

        unvisited = set(range(n))
        unvisited.remove(start_node_idx)
        unvisited.remove(end_node_idx)

        # --- Nearest Neighbour Iteration ---
        while unvisited:
            nearest_idx = -1
            min_dist = float("inf")

            for neighbor_idx in unvisited:
                d = matrix[current_node_idx, neighbor_idx]
                if d < min_dist:
                    min_dist = d
                    nearest_idx = neighbor_idx

            if nearest_idx == -1:
                print("Error: Could not find a nearest neighbour. Check the distance matrix.")
                print(f"Current node: {point_names[current_node_idx]}")
                return None, None

            total_distance += min_dist
            current_node_idx = nearest_idx
            path_indices.append(current_node_idx)
            unvisited.remove(current_node_idx)

        # Final leg to fixed end
        final_leg = matrix[current_node_idx, end_node_idx]
        total_distance += final_leg
        path_indices.append(end_node_idx)

        if verbose:
            print(
                f"Final leg to '{point_names[end_node_idx]}': {final_leg:.3f} km."
            )

        path_names = [point_names[i] for i in path_indices]

        if verbose:
            elapsed = time.time() - start_time
            print(f"Total distance: {total_distance:.3f} km  ({elapsed:.4f}s)")

        return path_names, total_distance

    def from_file(
        self,
        file_path: str,
        verbose: bool = True,
    ) -> tuple[list[str], float]:
        """Load a precomputed distance matrix CSV and run the solver.

        The CSV must have point names as both the index (first column)
        and column headers. This is the format produced by
        src/compute_distance_matrices.py.

        Args:
            file_path: Path to the distance matrix CSV.
            verbose:   Print loading and solver progress.

        Returns:
            (path_names, total_distance) or (None, None) on failure.
        """
        if verbose:
            print(f"Loading matrix from '{file_path}'...")

        if not os.path.exists(file_path):
            print(f"Error: File not found at '{file_path}'")
            return None, None

        try:
            df = pd.read_csv(file_path, index_col=0)

            if df.shape[0] != df.shape[1]:
                print(
                    f"Error: Matrix is not square ({df.shape[0]}x{df.shape[1]})."
                )
                return None, None

            matrix = df.to_numpy(dtype=float)
            point_names = df.index.tolist()

            if verbose:
                print(f"Matrix loaded: {matrix.shape[0]}x{matrix.shape[1]} points.")

        except pd.errors.EmptyDataError:
            print(f"Error: '{file_path}' is empty.")
            return None, None
        except ValueError as e:
            print(f"Error converting matrix data to numbers: {e}")
            return None, None
        except Exception as e:
            print(f"Unexpected error loading '{file_path}': {e}")
            return None, None

        return self.solve(matrix, point_names, verbose=verbose)

