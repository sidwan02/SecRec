import secure_algos
import tenseal as ts
from typing import Tuple, List


class Server:
    def __init__(
        self,
        public_context: bytes,
        secure_matrix_error_reset_wrapper: secure_algos.SecureMatrixErrorReset,
        secure_svd_wrapper: secure_algos.SecureSVD,
        secure_clip_wrapper: secure_algos.SecureClip,
        secure_division_wrapper: secure_algos.SecureClearDivision,
    ):
        self.storage = {}

        # Both ratings and is_filled are matrices of bytes. This allows for really nice serialization and ultimately storage, if we were to go in that direction.

        # N x M matrix with N users and M movies
        self.ratings = [[]]

        # N x M matrix with N users and M movies
        # The values are either 0 (value not filled) or 1 (value has been filled)
        # encryption and signature of this is maintained separately by the server

        # One disadvantage of having a matrix of bools:
        # ideally we wanna generate train test splits of indices + we need the ability to perform encryptions with the encryption key so we would be able to determine which entries are 0 and 1 anyways by manually checking against each row.
        self.is_filled = [[]]

        self.users_added_init = 0
        self.movies_added_init = 0

        self.init_complete = False

        self.encrypt_pk = ts.context_from(public_context)
        self.zero_bytes = ts.ckks_vector(self.encrypt_pk, [0]).serialize()

        self.secure_matrix_completion_wrapper = secure_algos.SecureMatrixCompletion(
            1,
            20,
            1e-2,
            public_context,
            secure_matrix_error_reset_wrapper,
            secure_svd_wrapper,
            secure_clip_wrapper,
            secure_division_wrapper,
        )

    # adds a new col
    def add_movie(self):
        # The default rating is 0 (invalid). Valid ratings are between 0.5 and 5.0

        if not self.init_complete:
            if self.users_added_init == 0:
                self.movies_added_init += 1
            else:
                assert self.users_added_init == 1
                self.init_complete = True
                self.ratings = [
                    [self.zero_bytes for _ in range(self.movies_added_init)]
                ]
                self.is_filled = [
                    [self.zero_bytes for _ in range(self.movies_added_init)]
                ]
            return

        for user_ratings in self.ratings:
            user_ratings.append(self.zero_bytes)
        for filled_row in self.is_filled:
            filled_row.append(self.zero_bytes)

    # adds a new row
    def add_user(self):
        if not self.init_complete:
            if self.movies_added_init == 0:
                self.users_added_init += 1
            else:
                assert self.movies_added_init == 1
                self.init_complete = True
                self.ratings = [
                    [self.zero_bytes] for _ in range(self.movies_added_init)
                ]
                self.is_filled = [
                    [self.zero_bytes] for _ in range(self.movies_added_init)
                ]
            return

        num_movies = len(self.ratings[0])

        # The default rating is 0 (invalid). Valid ratings are between 0.5 and 5.0
        self.ratings.append([self.zero_bytes for _ in range(num_movies)])
        self.is_filled.append([self.zero_bytes for _ in range(num_movies)])

    def modify_rating(self, rating: bytes, r: int, c: int):
        self.ratings[r][c] = rating

    def modify_filled(self, filled_status: bytes, r: int, c: int):
        self.is_filled[r][c] = filled_status

    def matrix_completion(self):
        self.secure_matrix_completion_wrapper.prepare_data(self.ratings, self.is_filled)
        return self.secure_matrix_completion_wrapper.train()

    def receive_rating(self, r: int, c: int) -> bytes:
        predicated_ratings = self.matrix_completion()

        return predicated_ratings[r][c]
