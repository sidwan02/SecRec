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

    def compute_SVD(self, A: np.ndarray[float], r: int = 6) -> Tuple[np.ndarray, np.ndarray[float]]:
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
    ) -> Tuple[np.ndarray[float], np.ndarray[float]]:
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
    def __init__(self, svd_1d_wrapper : InsecureSVD1D, epochs : int = 10, sub_epochs : int = 5, epsilon : float = 1e-3, alpha : float = 1, debug : bool = True):
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
                print(f"Iteration {curr_epoch}.{sub_epoch + 1}, Train loss: {new_loss}")
            
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
        return self.W[0 : B.shape[0], B.shape[0] : ]

# Insecure version of the matrix completion task (used in benchmarking for robust algos)
class InsecureMatrixCompletion:
    def __init__(
        self,
        r: int,
        epochs: int,
        alpha: float,
        insecure_svd_wrapper: InsecureSVD,
        loss_list = None
    ):
        # rank / no.of features
        self.r = r

        # iterations of SGD
        self.epochs = epochs

        # lr
        self.alpha: alpha

        # proportion of entries contributing to validation instead of training
        self.val_prop = 0.1

        # Insecure SVD wrapper
        self.insecure_svd_wrapper = insecure_svd_wrapper

        # List to accumulate losses
        self.loss_list = loss_list

    def prepare_data(
        self,
        ratings_mat: np.ndarray[float],
        is_filled_mat: np.ndarray[float],
    ):
        # m movies (cols), n users (rows)
        self.n, self.m = ratings_mat.shape[0], ratings_mat.shape[1]

        self.num_train_values: int = 0

        self.ratings_mat = ratings_mat
        self.is_filled_mat = is_filled_mat

        self.indices_mat = np.empty(self.ratings_mat.shape, dtype=object)
        for r in range(self.n):
            for c in range(self.m):
                self.indices_mat[r][c] = (r, c)
                self.num_train_values += self.is_filled_mat[r][c]

        U, vT = self.insecure_svd_wrapper.compute_SVD(self.ratings_mat, self.r)

        self.X = U
        self.Y = np.transpose(vT)

        assert self.X.shape == (self.n, self.r)
        assert self.Y.shape == (self.m, self.r)

    def shuffle_data(self):
        assert self.ratings_mat.shape == self.is_filled_mat.shape

        ratings_flat = np.array(self.ratings_mat).flatten()
        filled_flat = np.array(self.is_filled_mat).flatten()
        indices_flat = self.indices_mat.flatten()

        perm = np.random.permutation(len(ratings_flat))

        self.shuffled_rankings = ratings_flat[perm]
        self.shuffled_filled = filled_flat[perm]
        self.shuffled_indices = indices_flat[perm]

    def train(self):
        for cur_i in range(1, self.epochs + 1):
            # shuffle the train data
            self.shuffle_data()
            self.sgd()
            # err_val is unused since it is 0 right now
            err_train, err_val = self.error()
            print(f"Iteration {cur_i}, Train loss: {round(err_train, 4)}")

            # Accumulate losses for loss convergence testing
            if self.loss_list is not None:
                self.loss_list.append(round(err_train, 4))

        return self.compute_M_prime()

    def sgd(self):
        for s_no, (M_i_j, (i, j), is_filled) in enumerate(
            zip(self.shuffled_rankings, self.shuffled_indices, self.shuffled_filled)
        ):
            # there is no update to M if the value is not filled
            e_i_j = (M_i_j - self.pred_rating(i, j)) * is_filled

            for c in range(self.r):
                self.X[i, c] += self.alpha * 2 * e_i_j * self.Y[j, c]
                self.Y[j, c] += self.alpha * 2 * e_i_j * self.X[i, c]

    def pred_rating(self, i, j) -> float:
        val = self.X[i, :] @ self.Y[j, :].T

        # 0.5 <= rating <= 5
        val = np.clip(val, 0, 5)
        return val

    def error(self):
        error_train = 0
        error_val = 0
        M_prime = self.compute_M_prime()

        # Computing the MSE loss
        for s_no, (M_i_j, (i, j), is_filled) in enumerate(
            zip(self.shuffled_rankings, self.shuffled_indices, self.shuffled_filled)
        ):
            # ideally we want to maintain a train test split.
            error_train += (M_i_j - M_prime[i, j]) * (M_i_j - M_prime[i, j]) * is_filled

        return (
            error_train / self.num_train_values,
            0,
        )

    def compute_M_prime(self):
        M_prime = self.X @ self.Y.T
        M_prime = np.clip(M_prime, 0, 5)
        return M_prime

# Insecure implementation of robust matrix completion (for use in benchmarking performance)
class RobustInsecureMatrixCompletion(InsecureMatrixCompletion):
    def __init__(
        self,
        r: int,
        epochs: int,
        alpha: float,
        insecure_svd_wrapper: InsecureSVD,
        insecure_robust_weights_wrapper : InsecureRobustWeights,
        loss_list = None
    ):
        super().__init__(r, epochs, alpha, insecure_svd_wrapper, loss_list)
        self.insecure_robust_weights_wrapper = insecure_robust_weights_wrapper

    # Overwritten method to induce pre-processing weight computation
    def prepare_data(
        self,
        ratings_mat: np.ndarray[float],
        is_filled_mat: np.ndarray[float],
    ):
        # Only change in robust case is doing weight pre-processing on revealed entries
        self.weights_mat : np.ndarray[float] = self.insecure_robust_weights_wrapper.compute_weights(is_filled_mat)
        super().prepare_data(ratings_mat, self.weights_mat)