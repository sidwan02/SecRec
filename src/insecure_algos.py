# Insecure algorithms used for benchmarking performance tests

import numpy as np
import math
from scipy.sparse import linalg as slinalg
from typing import List, Tuple

class InsecureClip:
    def __init__(self):
        pass

    def sqrt(self, x: float, max_val: float = 8, max_iter: int = 4) -> float:
        # Scale x to interval (0, 1) to apply Wilkes' square root algorithm
        x *= (1 / max_val)

        # Setup variables for approximation
        a : float = x
        b : float = x - 1

        # max_iter is parameter that controls degree of approximation
        for _ in range(0, max_iter):
            a = a * (1 - (0.5 * b))
            b = (b ** 2) * ((b - 3) * 0.25)

        # Scale result back (note that m is plaintext so we may use the square root directly)
        return a * math.sqrt(max_val)

    def min(self, a: float, b: float, max_val: float = 8, max_iter: int = 4) -> float:
        # Uses fact that max of number is equal to average + half norm of difference
        average : float = (a + b) * 0.5
        difference : float = self.sqrt((a - b) ** 2, max_val=max_val ** 2, max_iter=max_iter) * 0.5
        return average - difference

    def max(self, a: float, b: float, max_val: float = 8, max_iter: int = 4) -> float:
        # Uses fact that max of number is equal to average + half norm of difference
        average : float = (a + b) * 0.5
        difference : float = self.sqrt((a - b) ** 2, max_val=max_val ** 2, max_iter=max_iter) * 0.5
        return average + difference

    # Actual clipping operation (easy with min and max)
    def clip(self, x : float, low: float, high: float, max_val : float = 8, max_iter = 4) -> float:
        return self.min(self.max(x, low, max_val=max_val, max_iter=max_iter), high, max_val=max_val, max_iter=max_iter)

class InsecureSciPySVD:
    def __init__(self):
        pass

    def compute_SVD(self, A: np.ndarray[float], r: int = 6) -> tuple[np.ndarray[float], np.ndarray[float]]:
        U, _, vT = slinalg.svds(A, k=r)
        return np.array(U), np.array(vT)

# Wrapper class for the eigenvalue extraction algorithm
class InsecureSVD1D:
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

class InsecureSVD:
    def __init__(self, debug : bool = True, svd_1d_wrapper = InsecureSVD1D):
        self.debug = debug
        self.svd_1d_wrapper = svd_1d_wrapper

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
                v = self.svd_1d_wrapper.svd_1d(matrix_to_decompose)
                # Compute singular value
                u : np.ndarray[float] = A @ v
                singular_value : float = np.linalg.norm(u)
                u = u / singular_value

            else:
                u = self.svd_1d_wrapper.svd_1d(matrix_to_decompose)
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

# Non-encrypted version of the gradient descent algorithm for computing weights
class InsecureRobustWeights:
    def __init__(self, svd_1d_wrapper : InsecureSVD1D, epochs : int = 10, sub_epochs : int = 5, epsilon : float = 1e-3, alpha : float = 0.5, debug : bool = True):
        self.debug = debug
        self.svd_1d_wrapper = svd_1d_wrapper
        self.alpha = alpha
        self.epochs = epochs
        self.sub_epochs = sub_epochs
        self.epsilon = epsilon

    def generate_block_matrices(self, B : np.ndarray[float]):
        self.B : np.ndarray[float] = np.block([
            [np.zeros((B.shape[0], B.shape[0])), B],
            [B.T, np.zeros((B.shape[1], B.shape[1]))]
        ])
        
        # Weights matrix to convert into block form
        weights : np.ndarray[float] = np.random.rand(B.shape[0], B.shape[1]) * B
        self.W : np.ndarray[float] = np.block([
            [np.zeros((B.shape[0], B.shape[0])), weights],
            [weights.T, np.zeros((B.shape[1], B.shape[1]))]
        ])

        self.J : np.ndarray[float] = np.block([
            [np.zeros((B.shape[0], B.shape[0])), np.ones(B.shape)],
            [np.ones(B.T.shape), np.zeros((B.shape[1], B.shape[1]))]
        ])

        # In the secure version we'd need to encrypt the matrices as this point

    # Computes loss of an inputted eigenvector
    def loss(self, v : np.ndarray[float]) -> float:
        loss : float = v.T @ self.W @ v
        # Note: in secure version the loss should be decrypted once computed
        return loss

    # Helper function to perform a gradient descent subroutine (on individual eigenvalues)
    def sub_gradient_descent(self, v : np.ndarray[float], curr_epoch : int):
        # Initialize loss for determining convergence
        old_loss = 0

        for sub_epoch in range(self.sub_epochs):
            # Compute loss
            new_loss : float = self.loss(v)

            # Compute gradient
            gradient : np.ndarray[float] = np.outer(v, v)

            # Apply gradient (with boolean mask to keep 0'd values 0)
            if new_loss > 0:
                self.W -= (self.alpha * gradient) * self.B
            elif new_loss < 0:
                self.W += (self.alpha * gradient) * self.B
            
            # Display loss information
            if self.debug:
                print(f"Iteration {curr_epoch}.{sub_epoch + 1}, Train loss: {abs(new_loss)}")
            
            # Exit early if loss does not substantially change
            if abs(abs(old_loss) - abs(new_loss)) < self.epsilon:
                return new_loss
                
            old_loss = new_loss
        
        # Return the most recent loss
        return old_loss

    def compute_weights(self, B : np.ndarray[float]) -> np.ndarray[float]:
        # Initialize weights
        self.generate_block_matrices(B)
        
        # Cache the computation computing the W - J matrix (we will add it back when we need to recover the weights)
        self.W -= self.J

        # In each epoch, we extract an eigenvector and run gradient descent using it
        old_loss = 0
        for epoch in range(self.epochs):
            # Extract eigenvector and run subroutine
            v = self.svd_1d_wrapper.svd_1d(self.W)
            new_loss = self.sub_gradient_descent(v, epoch + 1)

            # Determine convergence (measures difference in loss between subsequent subgradient descent steps)
            if abs(abs(old_loss) - abs(new_loss)) < self.epsilon:
                break

            old_loss = new_loss

        # Retrieve weights
        self.W += self.J
        print(B.shape)
        print(self.B.shape)
        return [row[len(B) : ] for row in self.W[0 : len(B)].tolist()]

# TODO: Add an insecure version of matrix completion (SGD) and robust matrix completion (SGD) ???