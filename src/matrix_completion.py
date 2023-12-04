import numpy as np
from scipy.sparse import linalg as slinalg
from support import crypto, util
import tenseal as ts
from typing import List


def decrypt_mat(
    A: List[List[ts.CKKSVector]], decrypt_sk: ts.Context
) -> List[List[float]]:
    plaintext_mat = np.empty((2, 4), dtype=float)

    for i in range(len(self.matrix)):
        for j in range(len(self.matrix[0])):
            m = self.matrix[i][j]
            m.link_context(decrypt_sk)
            plaintext_mat[i][j] = round(m.decrypt()[0], 4)

    print(plaintext_matrix)

    return plaintext_mat


def encrypt_mat(
    A: List[List[float]], encrypt_pk: ts.Context
) -> List[List[ts.CKKSVector]]:
    cipher_matrix = [[None for _ in range(len(A[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(A[0])):
            cipher_matrix[i][j] = ts.ckks_vector(encrypt_pk, [A[i][j]])

    cipher_matrix = np.array(cipher_matrix)

    return cipher_matrix


# TODO: make some tests for encrypt_mat and decrypt_mat and secure clip


class SecureSVD:
    def __init__(
        self, secret_context: bytes
    ) -> tuple(
        List[List[ts.CKKSVector]], List[List[ts.CKKSVector]], List[List[ts.CKKSVector]]
    ):
        self.decrypt_sk = ts.context_from(secret_context)

    def compute_SVD(
        self, A: List[List[ts.CKKSVector]], r: int
    ) -> tuple(List[List[ts.CKKSVector]], List[List[ts.CKKSVector]]):
        # same api as:
        # U, S, vT = slinalg.svds(ratings, self.r)
        # m movies, n users
        # U is n x r
        # vT is r x m

        # TODO: we will replace this with a secure SVD implementation later.

        ratings_mat = decrypt_mat(A, self.decrypt_sk)

        return slinalg.svds(ratings_mat, k=r)


class SecureClip:
    def __init__(self, self, secret_context: bytes, public_context: bytes):
        # Look into https://medium.com/optalysys/max-min-and-sort-functions-using-programmable-bootstrapping-in-concrete-fhe-ac4d9378f17d

        self.encrypt_sk = ts.context_from(public_context)
        self.decrypt_sk = ts.context_from(secret_context)

    def clip(self, x: ts.CKKSVector, min_val: float, max_val: float) -> ts.CKKSVector:
        x.link_context(self.decrypt_sk)

        x_plain = round(m.decrypt()[0], 4)

        ans = max(min(x_plain, max_val), min_val)

        return ts.ckks_vector(self.encrypt_sk, [ans])


class SecureMatrixCompletion:
    def __init__(
        self,
        ratings_mat: List[List[ts.CKKSVector]],
        is_filled_mat: List[List[ts.CKKSVector]],
        r: int,
        epochs: int,
        alpha: float,
        public_context: bytes,
        secure_svd_wrapper: SecureSVD,
        secure_clip_wrapper: SecureClip,
    ):
        self.encrypt_pk = ts.context_from(public_context)

        # m movies, n users
        self.n, self.m = len(ratings_mat[0]), len(ratings_mat)

        # TODO: all unpopulated ratings have encrypted value 0 (this should be the case before instantiating this class)
        self.ratings_mat = ratings_mat
        self.is_filled_mat = is_filled_mat
        self.indices_mat = np.empty(self.M.shape)
        for r in range(self.n):
            for c in range(self.m):
                self.indices_mat[r][c] = (r, c)

        # rank / no.of features
        self.r = r

        U, _, vT = secure_svd_wrapper.compute_SVD(self.ratings_mat, self.r)

        self.X = U
        self.Y = np.transpose(vT)

        assert self.X.shape == (self.n, self.r)
        assert self.Y.shape == (self.m, self.r)

        # iterations of SGD
        self.epochs = epochs

        # lr
        self.alpha: ts.CKKSVector = ts.ckks_vector(self.encrypt_pk, [alpha])

        # proportion of entries contributing to validation instead of training
        self.val_prop = 0.1

        self.secure_clip_wrapper = secure_clip_wrapper

    def shuffle_data(self):
        assert self.ratings_mat.shape == self.is_filled_mat.shape

        ratings_flat = np.array(self.ratings_mat).flatten()
        filled_flat = np.array(self.is_filled_mat).flatten()

        perm = np.random.permutation(len(self.ratings_flat))
        self.shuffled_rankings = ratings_flat[perm].reshape(self.ratings_mat.shape)
        self.shuffled_filled = filled_flat[perm].reshape(self.is_filled_mat.shape)
        self.shuffled_indices = self.indices_mat.flatten()[perm].reshape(
            self.indices_mat.shape
        )

    def train(self):
        for cur_i in range(1, self.epochs + 1):
            # shuffle the train data
            self.shuffle_data()
            self.sgd()
            err_train, err_val = self.error()

            print(f"Iteration {cur_i}, Train loss: {err_train}, Val loss: {err_val}")

    def sgd(self):
        two_encrypted = ts.ckks_vector(self.encrypt_pk, [2])

        for s_no, (M_i_j, (i, j), is_filled) in enumerate(
            zip(self.shuffled_rankings, self.shuffled_indices, self.shuffled_filled)
        ):
            # there is no update to M if the value is not filled
            e_i_j = (M_i_j - self.pred_rating(i, j)) * is_filled

            # gradient update rules derived from the loss function relation, derivation in the pdf
            # recall that both self.X and self.Y have the same number of cols
            for c in range(self.r):
                self.X[i, c] += self.alpha * two_encrypted * e_i_j * self.Y[j, c]
                self.Y[j, c] += self.alpha * two_encrypted * e_i_j * self.X[i, c]

            # for debugging purposes to see how far we are into the sgd
            if s_no % 100000 == 0:
                print(s_no / len(self.indices_train) * 100)

    def pred_rating(self, i, j) -> ts.CKKSVector:
        val = self.X[i, :] @ self.Y[j, :].T
        # 0.5 <= rating <= 5
        val = self.secure_clip_wrapper(val, 0.5, 5)
        return val

    def error(self):
        error_train = 0
        error_val = 0
        M_prime = self.compute_M_prime()

        # Computing the MSE loss
        for (i, j), M_i_j in zip(self.indices_train, self.ratings_train):
            error_train += (M_i_j - M_prime[i, j]) ** 2

        for (i, j), M_i_j in zip(self.indices_val, self.ratings_val):
            error_val += (M_i_j - M_prime[i, j]) ** 2

        # normalizing by the number of seen entries
        return (
            error_train / (len(self.indices_train)),
            error_val / (len(self.indices_val)),
        )

    def compute_M_prime(self):
        M_prime = self.X @ self.Y.T
        M_prime = np.clip(M_prime, 0.5, 5)
        return M_prime

    def generate_output(self):
        M_prime = self.compute_M_prime()

        f = open("mat_comp_ans", "w")
        for i, j in self.queries:
            f.write(f"{M_prime[i, j]}\n")
        f.close()

    # when the user writes they write the whole row, then everyone uses PIR to access it
    # singular key is ok. But see if there's


mf = MatrixFactorization(file_path="./mat_comp", r=30, epochs=10, alpha=1e-3)
print("starting train")
mf.train()
print(mf.M)
print(mf.compute_M_prime())
mf.generate_output()
