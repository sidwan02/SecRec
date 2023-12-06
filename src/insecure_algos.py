# Insecure algorithms used for benchmarking performance tests

import numpy as np
from scipy.sparse import linalg as slinalg
from typing import List, tuple

class InsecureClip:
    pass

class InsecureSciPySVD:
    def __init__(self):
        pass

    def compute_SVD(self, A: np.ndarray[float], r: int = 6) -> tuple[np.ndarray[float], np.ndarray[float]]:
        U, _, vT = slinalg.svds(A, k=r)
        return np.array(U), np.array(vT)

class InsecureSVD:
    def __init__(self, debug : bool = True):
        self.debug = debug

    # Helper function to generate a random unit vector (or uniform vector)
    def random_unit_vector(self, n : int, uniform : bool = False) -> np.ndarray[float]:
        if uniform:
            x : np.array = np.ones(0, 2, n)
            # Ensure we have at least one nonzero value
            x[np.random.randint(0, n)] = 1
        else:
            x : np.array = np.random.randint(0, 2, n)
        norm : float = np.linalg.norm(x)
        return x / norm

    # Helper function to extract the largest eigenvector (and throw away the eigenvalue)
    # Note that inputted matrix is not necessarily square
    def svd_1d(self, A : np.ndarray[float], epsilon : float = 1e-1, max_iter = 100) -> np.ndarray[float]:
        rows : int = A.shape[0]
        cols : int = A.shape[1]
        dim : int = min(rows, cols)

        # Initialize starting values for iteration
        eigenvector : np.ndarray[float] = self.random_unit_vector(dim)

        # We need to do this multiplication to guarentee we have a square matrix
        if rows > cols:
            B : np.ndarray[float] = A.T @ A
        elif rows < cols:
            B : np.ndarray[float] = A @ A.T
        else:
            B : np.ndarray[float] = A

         # Cap iterations at max_iter (can use this to empirically determine good iteration numbers)
        for iteration in range(max_iter):
            new_eigenvector : np.ndarray[float] = B @ eigenvector
            new_eigenvector_norm = np.linalg.norm(new_eigenvector)
            new_eigenvector = new_eigenvector / new_eigenvector_norm

            # Termination condition, our eigenvalues did not significantly change
            if abs(np.dot(eigenvector, new_eigenvector)) > 1 - epsilon:
                if self.debug:
                    print(f"Terminated after {iteration + 1} iterations")
                return new_eigenvector
            
            eigenvector = new_eigenvector
    
        # If no termination was reached, return our best guess
        if self.debug:
            print(f"Failed to converge after {max_iter} iterations")

        return eigenvector

    # Big SVD Function
    def compute_SVD(self, A: np.ndarray[float], r: int = 6
    ) -> tuple[np.ndarray[float], np.ndarray[float]]:
        # Only extract first r singular values unless input is set to -1 (in that case extract as many as possible)
        rows : int = A.shape[0]
        cols : int = A.shape[1]
        dim : int = min(rows, cols)
        if r == -1 or r > dim:
            r = dim
        
        # Set up a matrix to decompose (we will remove components of eigenvectors from it)
        matrix_to_decompose : np.ndarray[float] = np.copy(A)

        # Lists to hold our decomposed values
        vs : List[np.ndarray[float]] = []
        us : List[np.ndarray[float]] = []

        # Extract the first r singular values one at a time (looping through the svd_1d subroutine)
        for __ in range(r):

            # Fill u or v depending on size of matrices
            if rows > cols:
                v = self.svd_1d(matrix_to_decompose)
                # Compute singular value
                u : np.ndarray[float] = A @ v
                singular_value : float = np.linalg.norm(u)
                u = u / singular_value

            else:
                u = self.svd_1d(matrix_to_decompose)
                # Compute singular value
                v : np.ndarray[float] = A.T @ u
                singular_value : float = np.linalg.norm(v)
                v = v / singular_value
            
            # Update our eigenvectors
            us.append(u)
            vs.append(v)

            # Update matrix by removing values that correspond to old singular values
            matrix_to_decompose -= singular_value * np.outer(u, v)
        
        # Compile all the values into an array
        us_arr : np.ndarray[float] = np.array(us)
        vs_arr : np.ndarray[float] = np.array(vs)

        return us_arr.T, vs_arr

class InsecureRobustWeights:
    pass