import numpy as np
from scipy.sparse import linalg as slinalg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class MatrixFactorization:
    def __init__(self, file_path, r, epochs, alpha):

        file1 = open(file_path, "r")
        # k ratings, m movies, n users
        self.n, self.m, k = list(map(int, file1.readline().split(" ")))

        # initialize user-movie rating matrix with 0s (missing results remain 0 as M is populated)
        self.M = np.zeros((self.n, self.m))

        # we build the indices and ratings directly when reading the file since identifying them from sparse M each time is computationally wasteful
        indices = []
        ratings = []

        while k > 0:
            line = file1.readline()
            if not line:
                print("improperly formatted file")
            i, j, M_i_j = line.split(" ")
            # convert 1 indexing to 0 indexing
            self.M[int(i) - 1, int(j) - 1] = float(M_i_j)
            indices.append((int(i) - 1, int(j) - 1))
            ratings.append(float(M_i_j))

            k -= 1

        q = int(file1.readline())

        # store the queries for later
        self.queries = []

        while q > 0:
            line = file1.readline()
            if not line:
                print("improperly formatted file")
            i, j = list(map(int, line.split(" ")))
            # convert 1 indexing to 0 indexing
            self.queries.append([i - 1, j - 1])

            q -= 1

        # make sure we've read everything/file is properly formatted
        assert not file1.readline()

        file1.close()

        # rank / no.of features
        self.r = r

        U, S, vT = slinalg.svds(self.M, k=r)

        self.X = U
        self.Y = np.transpose(vT)

        assert self.X.shape == (self.n, self.r)
        assert self.Y.shape == (self.m, self.r)

        # iterations of SGD
        self.epochs = epochs

        # lr
        self.alpha = alpha

        # create train/validation split
        (
            self.indices_train,
            self.indices_val,
            self.ratings_train,
            self.ratings_val,
        ) = train_test_split(indices, ratings, test_size=0.1)

    def train(self):
        for cur_i in range(1, self.epochs + 1):
            # shuffle the train data
            self.indices_train, self.ratings_train = shuffle(
                self.indices_train, self.ratings_train
            )
            self.sgd()
            err_train, err_val = self.error()

            print(f"Iteration {cur_i}, Train loss: {err_train}, Val loss: {err_val}")

    def sgd(self):
        for s_no, ((i, j), M_i_j) in enumerate(
            zip(self.indices_train, self.ratings_train)
        ):
            e_i_j = M_i_j - self.get_rating(i, j)

            # gradient update rules derived from the loss function relation, derivation in the pdf
            self.X[i, :] += self.alpha * 2 * e_i_j * self.Y[j, :]
            self.Y[j, :] += self.alpha * 2 * e_i_j * self.X[i, :]

            # for debugging purposes to see how far we are into the sgd
            if s_no % 100000 == 0:
                print(s_no / len(self.indices_train) * 100)

    def get_rating(self, i, j):
        val = self.X[i, :] @ self.Y[j, :].T
        # 0.5 <= rating <= 5
        val = max(min(val, 5), 0.5)
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


mf = MatrixFactorization(file_path="./mat_comp", r=30, epochs=10, alpha=1e-3)
print("starting train")
mf.train()
print(mf.M)
print(mf.compute_M_prime())
mf.generate_output()

