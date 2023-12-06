import numpy as np
from scipy.sparse import linalg as slinalg
from support import crypto, util
import tenseal as ts
from typing import List, Tuple
import time
import math


# Eventually, these wrapper classes (division, error reset, svd, clip) can have a privacy budget preventing the server from calling them for nefarious purposes (within reasonable estimates).
class SecureClearDivision:
    def __init__(self, secret_context: bytes):
        self.decrypt_sk = ts.context_from(secret_context)

    def compute_division(self, x: ts.CKKSVector, y: ts.CKKSVector) -> float:
        x.link_context(self.decrypt_sk)
        y.link_context(self.decrypt_sk)

        x_plain = round(x.decrypt()[0], 4)
        y_plain = round(y.decrypt()[0], 4)

        ans = x_plain / y_plain

        return ans


class SecureMatrixErrorReset:
    def __init__(self, public_context, secret_context):
        self.encrypt_pk = ts.context_from(public_context)
        self.decrypt_sk = ts.context_from(secret_context)

    def reset_error(self, M) -> List[List[ts.CKKSVector]]:
        decrypted_M = util.decrypt_ckks_mat(M, self.decrypt_sk)
        return util.encrypt_to_ckks_mat(decrypted_M, self.encrypt_pk)

# SVD implementation from SciPy (cannot be performed on server)
class SecureSciPySVD:
    def __init__(self, public_context: bytes, secret_context: bytes):
        self.encrypt_pk = ts.context_from(public_context)
        self.decrypt_sk = ts.context_from(secret_context)

    def compute_SVD(
        self, A: List[List[bytes]], r: int
    ) -> Tuple[List[List[ts.CKKSVector]], List[List[ts.CKKSVector]]]:
        ratings_mat = util.decrypt_ckks_mat(A, self.decrypt_sk)

        # This operation cannot be performed on the server
        U, _, vT = slinalg.svds(ratings_mat, k=r)

        return util.encrypt_to_ckks_mat(U, self.encrypt_pk), util.encrypt_to_ckks_mat(
            vT, self.encrypt_pk
        )

# SVD implementation using power iteration (easier to implement in FHE world)
class SecureSVD:
    def __init__(self, public_context: bytes, secret_context: bytes, debug : bool = True):
        self.encrypt_pk = ts.context_from(public_context)
        self.decrypt_sk = ts.context_from(secret_context)
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
    def svd_1d(self, A : List[List[ts.CKKSVector]], epsilon : float = 1e-1, max_iter = 100) -> np.ndarray[float]:
        rows : int = len(A)
        cols : int = len(A[0])
        dim : int = min(rows, cols)

        # Initialize starting values for iteration
        eigenvector : np.ndarray[float] = self.random_unit_vector(dim)

        # We need to do this multiplication to guarentee we have a square matrix
        if rows > cols:
            B : List[ts.CKKSVector] = A.T @ A
        elif rows < cols:
            B : List[ts.CKKSVector] = A @ A.T
        else:
            B : List[ts.CKKSVector] = A

         # Cap iterations at max_iter (can use this to empirically determine good iteration numbers)
        for iteration in range(max_iter):
            new_eigenvector : List[ts.CKKSVector] = B @ eigenvector

            # === Begin Client-side Operations === #

            # Decrypt and normalize the eigenvector
            new_eigenvector = np.array(util.decrypt_ckks_vec(new_eigenvector, self.decrypt_sk))
            new_eigenvector_norm = np.linalg.norm(new_eigenvector)
            new_eigenvector = new_eigenvector / new_eigenvector_norm

            # === End Client-side Operations === #

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
    def compute_SVD(self, A: List[List[ts.CKKSVector]], r: int = 6
    ) -> Tuple[List[List[ts.CKKSVector]], List[List[ts.CKKSVector]]]:
        # Only extract first r singular values unless input is set to -1 (in that case extract as many as possible)
        rows : int = len(A)
        cols : int = len(A[0])
        dim : int = min(rows, cols)
        if r == -1 or r > dim:
            r = dim
        
        # Set up a matrix to decompose (we will remove components of eigenvectors from it)
        matrix_to_decompose : List[ts.CKKSVector] = np.copy(A)

        # Lists to hold our decomposed values
        vs : List[np.ndarray[float]] = []
        us : List[np.ndarray[float]] = []

        # Extract the first r singular values one at a time (looping through the svd_1d subroutine)
        for __ in range(r):

            # Fill u or v depending on size of matrices
            if rows > cols:
                v = self.svd_1d(matrix_to_decompose)
                # Compute singular value
                u : List[ts.CKKSVector] = A @ v

                # === Begin Client-side Operations === #

                # Decrypt and normalize other eigenvector
                u = np.array(util.decrypt_ckks_vec(u, self.decrypt_sk))
                singular_value : float = np.linalg.norm(u)
                u = u / singular_value

                # === End Client-side Operations === #


            else:
                u = self.svd_1d(matrix_to_decompose)
                # Compute singular value
                v : np.ndarray[float] = A.T @ u

                # === Begin Client-side Operations === #

                # Decrypt and normalize other eigenvector
                v = np.array(util.decrypt_ckks_vec(v, self.decrypt_sk))
                singular_value : float = np.linalg.norm(v)
                v = v / singular_value

                # === End Client-side Operations === #
            
            # Update our eigenvectors
            us.append(u)
            vs.append(v)

            # Update matrix by removing values that correspond to old singular values
            matrix_to_decompose -= singular_value * np.outer(u, v)
        
        # Compile all the values into an array
        us_arr : np.ndarray[float] = np.array(us)
        vs_arr : np.ndarray[float] = np.array(vs)

        return util.encrypt_to_ckks_mat(us_arr.T, self.encrypt_pk), util.encrypt_to_ckks_mat(vs_arr, self.encrypt_pk)

# Secure clipping implementation that keeps computation completely over FHE space
class SecureFHEClip:
    def __init__(self, public_context: bytes, secret_context: bytes):
        self.encrypt_sk = ts.context_from(public_context)
        self.decrypt_sk = ts.context_from(secret_context)

    def sqrt(self, x: ts.CKKSVector, max_val: float = 8, max_iter: int = 4) -> ts.CKKSVector:
        # Scale x to interval (0, 1) to apply Wilkes' square root algorithm
        x *= (1 / max_val)

        # Setup variables for approximation
        a : ts.CKKSVector = x
        b : ts.CKKSVector = x - 1

        # max_iter is parameter that controls degree of approximation
        for _ in range(0, max_iter):
            a = a * (1 - (0.5 * b))
            b = (b ** 2) * ((b - 3) * 0.25)

        # Scale result back (note that m is plaintext so we may use the square root directly)
        return a * math.sqrt(max_val)

    def min(self, a: ts.CKKSVector, b: float, max_val: float = 8, max_iter: int = 4) -> ts.CKKSVector:
        # Uses fact that max of number is equal to average + half norm of difference
        average : ts.CKKSVector = (a + b) * 0.5
        difference : ts.CKKSVector = self.sqrt((a - b) ** 2, max_val=max_val ** 2, max_iter=max_iter) * 0.5
        return average - difference

    def max(self, a: ts.CKKSVector, b: float, max_val: float = 8, max_iter: int = 4) -> ts.CKKSVector:
        # Uses fact that max of number is equal to average + half norm of difference
        average : ts.CKKSVector = (a + b) * 0.5
        # Note: need to set this maximum value to the square of the original
        difference : ts.CKKSVector = self.sqrt((a - b) ** 2, max_val=max_val ** 2, max_iter=max_iter) * 0.5
        return average + difference

    # Actual clipping operation (easy with min and max)
    def clip(self, x : ts.CKKSVector, low: float, high: float, max_val : float = 8, max_iter = 4) -> float:
        return self.min(self.max(x, low, max_val=max_val, max_iter=max_iter), high, max_val=max_val, max_iter=max_iter)

# Clipping implementation that involves decrypting and re-encrypting
class SecureClip:
    def __init__(self, public_context: bytes, secret_context: bytes):
        self.encrypt_sk = ts.context_from(public_context)
        self.decrypt_sk = ts.context_from(secret_context)

    def clip(self, x: ts.CKKSVector, min_val: float, max_val: float) -> ts.CKKSVector:
        x.link_context(self.decrypt_sk)
        x_plain = round(x.decrypt()[0], 4)

        ans = max(min(x_plain, max_val), min_val)

        return ts.ckks_vector(self.encrypt_sk, [ans])


class SecureMatrixCompletion:
    def __init__(
        self,
        r: int,
        epochs: int,
        alpha: float,
        public_context: bytes,
        secure_matrix_error_reset_wrapper: SecureMatrixErrorReset,
        secure_svd_wrapper: SecureSVD,
        secure_clip_wrapper: SecureClip,
        secure_division_wrapper: SecureClearDivision,
    ):
        self.encrypt_pk = ts.context_from(public_context)

        # rank / no.of features
        self.r = r

        # iterations of SGD
        self.epochs = epochs

        # lr
        self.alpha: ts.CKKSVector = ts.ckks_vector(self.encrypt_pk, [alpha])

        # proportion of entries contributing to validation instead of training
        self.val_prop = 0.1

        self.secure_matrix_error_reset_wrapper = secure_matrix_error_reset_wrapper
        self.secure_clip_wrapper = secure_clip_wrapper
        self.secure_svd_wrapper = secure_svd_wrapper
        self.secure_division_wrapper = secure_division_wrapper

    def prepare_data(
        self,
        ratings_mat: List[List[bytes]],
        is_filled_mat: List[List[bytes]],
    ):
        # m movies (cols), n users (rows)
        self.n, self.m = len(ratings_mat), len(ratings_mat[0])

        self.num_train_values: ts.CKKSVector = ts.ckks_vector(self.encrypt_pk, [0])

        self.ratings_mat = np.array(
            util.convert_bytes_mat_to_ckks_mat(ratings_mat, self.encrypt_pk)
        )
        self.is_filled_mat = np.array(
            util.convert_bytes_mat_to_ckks_mat(is_filled_mat, self.encrypt_pk)
        )
        self.indices_mat = np.empty(self.ratings_mat.shape, dtype=object)
        for r in range(self.n):
            for c in range(self.m):
                self.indices_mat[r][c] = (r, c)
                self.num_train_values += self.is_filled_mat[r][c]

        U, vT = self.secure_svd_wrapper.compute_SVD(self.ratings_mat, self.r)

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

        return util.convert_ckks_mat_to_bytes_mat(self.compute_M_prime())

    def sgd(self):
        two_encrypted = ts.ckks_vector(self.encrypt_pk, [2])

        for s_no, (M_i_j, (i, j), is_filled) in enumerate(
            zip(self.shuffled_rankings, self.shuffled_indices, self.shuffled_filled)
        ):
            # there is no update to M if the value is not filled
            e_i_j = (M_i_j - self.pred_rating(i, j)) * is_filled

            self.X = self.secure_matrix_error_reset_wrapper.reset_error(self.X)
            self.Y = self.secure_matrix_error_reset_wrapper.reset_error(self.Y)
            for c in range(self.r):
                self.X[i, c] += self.alpha * two_encrypted * e_i_j * self.Y[j, c]
                self.Y[j, c] += self.alpha * two_encrypted * e_i_j * self.X[i, c]

    def pred_rating(self, i, j) -> ts.CKKSVector:
        val = self.X[i, :] @ self.Y[j, :].T

        # 0.5 <= rating <= 5
        val = self.secure_clip_wrapper.clip(val, 0, 5)
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
            self.secure_division_wrapper.compute_division(
                error_train, self.num_train_values
            ),
            0,
        )

    def compute_M_prime(self):
        M_prime = self.X @ self.Y.T
        M_prime = np.vectorize(lambda x: self.secure_clip_wrapper.clip(x, 0, 5))(
            M_prime
        )
        return M_prime
