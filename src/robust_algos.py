# Algorithms implementing robust matrix completion (semi-random matrix completion)

from typing import List
from secure_algos import SecureMatrixCompletion, SecureMatrixErrorReset, SecureSVD, SecureSVD1D, SecureClip, SecureClearDivision

class SecureRobustWeights:
    pass

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
        # TODO: Add new class wrapper for weight computation (preprocessing step)
    ):
        super().__init__(r, epochs, alpha, public_context, secure_matrix_error_reset_wrapper, secure_svd_wrapper, secure_clip_wrapper, secure_division_wrapper)

    # Overwritten method to induce pre-processing weight computation
    def prepare_data(
        self,
        ratings_mat: List[List[bytes]],
        is_filled_mat: List[List[bytes]],
    ):
        # TODO: add methods for weight computation
        weights_mat : List[List[bytes]] = is_filled_mat
        super().prepare_data(ratings_mat, weights_mat)