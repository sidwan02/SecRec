# Algorithms implementing robust matrix completion (semi-random matrix completion)

import numpy as np
import tenseal as ts
from support import crypto, util

from typing import List
from secure_algos import SecureMatrixCompletion, SecureMatrixErrorReset, SecureSVD, SecureSVD1D, SecureClip, SecureClearDivision

class SecureRobustWeights:
    def __init__(self, public_context: bytes, secret_context: bytes, svd_1d_wrapper : SecureSVD1D, epochs : int = 10, sub_epochs : int = 5, epsilon : float = 1e-3, alpha : float = 0.5, debug : bool = True):
        self.encrypt_pk = ts.context_from(public_context)
        self.decrypt_sk = ts.context_from(secret_context)
        self.debug = debug
        self.svd_1d_wrapper = svd_1d_wrapper
        self.alpha = alpha
        self.epochs = epochs
        self.sub_epochs = sub_epochs
        self.epsilon = epsilon

    # Helper function to generate block matrices for encrypted matrices
    def block(self, M : List[List[ts.CKKSVector]]):
        new_mat : List[List[ts.CKKSVector]] = []

        encrypted_0 : ts.CKKSVector = ts.ckks_vector(self.encrypt_pk, np.zeros(1))
        m_rows : int = len(M)
        m_cols : int = len(M[0])

        # Create individual rows
        # Upper portion of the matrix
        for row in range(m_rows):
            new_row : List[ts.CKKSVector] = []
            for col in range(m_rows + m_cols):
                # Determine if we want the 0s or the inputted matrix
                if col < m_rows:
                    new_row.append(encrypted_0)
                else:
                    new_row.append(M[row][col - m_rows])
            new_mat.append(new_row)
        
        # Lower portion of the matrix
        for row in range(m_cols):
            new_row : List[ts.CKKSVector] = []
            for col in range(m_rows + m_cols):
                if col < m_rows:
                    new_row.append(M[col][row])
                else:
                    new_row.append(encrypted_0)
            new_mat.append(new_row)
        
        return new_mat

    def generate_block_matrices(self, B : List[List[bytes]]):
        b_rows : int = len(B)
        b_cols : int = len(B[0])

        encrypted_B : List[List[ts.CKKSVector]] = util.convert_bytes_mat_to_ckks_mat(B, self.encrypt_pk)

        # Note that the boolean matrix tracking filled entries and the weights matrix are encrypted (hidden)
        self.B : List[List[ts.CKKSVector]] = self.block(encrypted_B)
        
        # Weights matrix to convert into block form
        weights : List[List[ts.CKKSVector]] = np.random.rand(b_rows, b_cols) * encrypted_B
        self.W : List[List[ts.CKKSVector]] = self.block(weights)

        self.J : np.ndarray[float] = np.block([
            [np.zeros((b_rows, b_rows)), np.ones((b_rows, b_cols))],
            [np.ones((b_cols, b_rows)), np.zeros((b_cols, b_cols))]
        ])

    # Computes loss of an inputted eigenvector
    def loss(self, v : np.ndarray[float]) -> float:
        loss : ts.CKKSVector = v.T @ self.W @ v
        # Decrypt the loss once computed
        loss.link_context(self.decrypt_sk)
        return round(loss.decrypt()[0], 4)

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

    def compute_weights(self, B : List[List[bytes]]) -> List[List[bytes]]:
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
        # Convert to bytes before returning
        return util.convert_ckks_mat_to_bytes_mat([row[len(B) : ] for row in self.W[0 : len(B)]])

class RobustSecureMatrixCompletion(SecureMatrixCompletion):
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
        secure_robust_weights_wrapper : SecureRobustWeights
    ):
        super().__init__(r, epochs, alpha, public_context, secure_matrix_error_reset_wrapper, secure_svd_wrapper, secure_clip_wrapper, secure_division_wrapper)
        self.secure_robust_weights_wrapper = secure_robust_weights_wrapper

    # Overwritten method to induce pre-processing weight computation
    def prepare_data(
        self,
        ratings_mat: List[List[bytes]],
        is_filled_mat: List[List[bytes]],
    ):
        # Only change in robust case is doing weight pre-processing on revealed entries
        self.weights_mat : List[List[bytes]] = self.secure_robust_weights_wrapper.compute_weights(is_filled_mat)
        super().prepare_data(ratings_mat, self.weights_mat)