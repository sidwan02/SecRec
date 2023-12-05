import numpy as np
from scipy.sparse import linalg as slinalg
from support import crypto, util
import tenseal as ts
from typing import List, Tuple
from support.util import (
    convert_bytes_mat_to_ckks_mat,
    encrypt_to_ckks_mat,
    decrypt_ckks_mat,
)
import time


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


class SecureSVD:
    def __init__(self, public_context: bytes, secret_context: bytes):
        self.encrypt_pk = ts.context_from(public_context)
        self.decrypt_sk = ts.context_from(secret_context)

    def compute_SVD(
        self, A: List[List[bytes]], r: int
    ) -> Tuple[List[List[ts.CKKSVector]], List[List[ts.CKKSVector]]]:
        ratings_mat = decrypt_ckks_mat(A, self.decrypt_sk)

        U, _, vT = slinalg.svds(ratings_mat, k=r)

        return encrypt_to_ckks_mat(U, self.encrypt_pk), encrypt_to_ckks_mat(
            vT, self.encrypt_pk
        )


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
        secret_context: bytes,
        secure_svd_wrapper: SecureSVD,
        secure_clip_wrapper: SecureClip,
        secure_division_wrapper: SecureClearDivision,
    ):
        self.encrypt_pk = ts.context_from(public_context)
        # TODO: remove the decrypt_pk later
        self.decrypt_sk = ts.context_from(secret_context)

        # rank / no.of features
        self.r = r

        # iterations of SGD
        self.epochs = epochs

        # lr
        self.alpha: ts.CKKSVector = ts.ckks_vector(self.encrypt_pk, [alpha])

        # proportion of entries contributing to validation instead of training
        self.val_prop = 0.1

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
            convert_bytes_mat_to_ckks_mat(ratings_mat, self.encrypt_pk)
        )
        self.is_filled_mat = np.array(
            convert_bytes_mat_to_ckks_mat(is_filled_mat, self.encrypt_pk)
        )
        self.indices_mat = np.empty(self.ratings_mat.shape, dtype=object)
        for r in range(self.n):
            for c in range(self.m):
                self.indices_mat[r][c] = (r, c)

                # TODO: maybe this being lazy is a problem
                self.num_train_values += self.is_filled_mat[r][c]

        # print(self.indices_mat)

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
            err_train, err_val = self.error()

            # print(f"Iteration {cur_i}, Train loss: {err_train}, Val loss: {err_val}")
            print(f"Iteration {cur_i}, Train loss: {round(err_train, 4)}")

        return util.convert_ckks_mat_to_bytes_mat(self.compute_M_prime())

    def sgd(self):
        two_encrypted = ts.ckks_vector(self.encrypt_pk, [2])

        for s_no, (M_i_j, (i, j), is_filled) in enumerate(
            zip(self.shuffled_rankings, self.shuffled_indices, self.shuffled_filled)
        ):
            start = time.time()
            # print(s_no)
            # there is no update to M if the value is not filled
            e_i_j = (M_i_j - self.pred_rating(i, j)) * is_filled

            # TODO: remove this later
            e_i_j.link_context(self.decrypt_sk)
            # print(f"e_i_j: {round(e_i_j.decrypt()[0], 4)}")

            # gradient update rules derived from the loss function relation, derivation in the pdf
            # recall that both self.X and self.Y have the same number of cols

            self.X = self.reset_matrix_error(self.X)
            self.Y = self.reset_matrix_error(self.Y)
            for c in range(self.r):
                # print(self.alpha * two_encrypted * e_i_j * self.Y[j, c])
                self.X[i, c] += self.alpha * two_encrypted * e_i_j * self.Y[j, c]
                self.Y[j, c] += self.alpha * two_encrypted * e_i_j * self.X[i, c]

            """
            self.X = self.reset_matrix_error(self.X)
            self.Y = self.reset_matrix_error(self.Y)
            # print(self.alpha * two_encrypted * e_i_j * self.Y[j, c])
            self.X[i, :] += self.alpha * two_encrypted * e_i_j * self.Y[j, :]
            self.Y[j, :] += self.alpha * two_encrypted * e_i_j * self.X[i, :]
            """

            # for debugging purposes to see how far we are into the sgd
            # if s_no % 100000 == 0:
            # print(f"Computing: {s_no / (self.n * self.m) * 100} %")

            end = time.time()
            # print(end - start)

    # TODO: this is a really bad way of reseting error. We need to find a better way
    def reset_matrix_error(self, M):
        decrypted_M = util.decrypt_ckks_mat(M, self.decrypt_sk)
        return util.encrypt_to_ckks_mat(decrypted_M, self.encrypt_pk)

    def pred_rating(self, i, j) -> ts.CKKSVector:
        self.X = self.reset_matrix_error(self.X)
        self.Y = self.reset_matrix_error(self.Y)

        val = self.X[i, :] @ self.Y[j, :].T

        # 0.5 <= rating <= 5
        val = self.secure_clip_wrapper.clip(val, 0.5, 5)
        return val

    def error(self):
        error_train = 0
        error_val = 0
        M_prime = self.compute_M_prime()

        # Computing the MSE loss
        for s_no, (M_i_j, (i, j), is_filled) in enumerate(
            zip(self.shuffled_rankings, self.shuffled_indices, self.shuffled_filled)
        ):
            # TODO: later on, split train set into train and val.
            error_train += (M_i_j - M_prime[i, j]) * (M_i_j - M_prime[i, j]) * is_filled

        # for (i, j), M_i_j in zip(self.indices_train, self.ratings_train):
        #     error_train += (M_i_j - M_prime[i, j]) ** 2

        # for (i, j), M_i_j in zip(self.indices_val, self.ratings_val):
        #     error_val += (M_i_j - M_prime[i, j]) ** 2

        # normalizing by the number of seen entries
        # return (
        #     error_train / (len(self.indices_train)),
        #     error_val / (len(self.indices_val)),
        # )
        return (
            self.secure_division_wrapper.compute_division(
                error_train, self.num_train_values
            ),
            0,
        )

    def compute_M_prime(self):
        M_prime = self.X @ self.Y.T
        M_prime = np.vectorize(lambda x: self.secure_clip_wrapper.clip(x, 0.5, 5))(
            M_prime
        )
        return M_prime