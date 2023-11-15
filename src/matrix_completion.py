import numpy as np
from scipy.sparse import linalg as slinalg
from support import crypto, util
import tenseal as ts
from typing import List


class SecureSVD:
    def __init__(
        self, A: List[List[ts.CKKSVector]], r: int
    ) -> tuple(
        List[List[ts.CKKSVector]], List[List[ts.CKKSVector]], List[List[ts.CKKSVector]]
    ):
        # same api as:
        # U, S, vT = slinalg.svds(ratings, self.r)
        # m movies, n users
        # U is n x r
        # vT is r x m
        pass


class SecureMatrixMultiplication:
    def __init__(self, A: List[List[ts.CKKSVector]], B: List[List[ts.CKKSVector]]):
        # there is likely already an implementation in CKKSTensor in tenseal
        # actually, since we're not using CKKStensors, we might be able to get away with broadcast working for the @ symbol! We should test this out
        pass


class SecureClip:
    def __init__(self, x: float, min: float, max: float):
        pass


class SecureMatrixCompletion:
    def __init__(
        self,
        ratings_mat: List[List[ts.CKKSVector]],
        is_filled_mat: List[List[ts.CKKSVector]],
        r: int,
        epochs: int,
        alpha: float,
        public_context: bytes,
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

        # TODO: we need an SVD that supports FHE
        U, _, vT = slinalg.svds(ratings_mat, k=r)

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
        # val = max(min(val, 5), 0.5)
        val = SecureClip(val, 0.5, 5)
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
