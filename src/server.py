from matrix_completion import SecureMatrixCompletion


class Server:
    def __init__(
        self,
        public_context: bytes,
        secure_svd_wrapper: SecureSVD,
        secure_clip_wrapper: SecureClip,
    ):
        self.storage = {}

        # N x M matrix with N users and M movies
        # encryption and signature of this is maintained separately by the server
        self.ratings = [[]]

        # N x M matrix with N users and M movies
        # The values are either 0 (value not filled) or 1 (value has been filled)
        # encryption and signature of this is maintained separately by the server

        # One disadvantage of having a matrix of bools:
        # ideally we wanna generate train test splits of indices + we need the ability to perform encryptions with the encryption key so we would be able to determine which entries are 0 and 1 anyways by manually checking against each row.
        self.is_filled = [[]]

        self.secure_matrix_completion_wrapper = SecureMatrixCompletion(
            10, 50, 1e-1, public_context, secure_svd_wrapper, secure_clip_wrapper
        )

    # adds a new col
    def add_movie(self):
        # The default rating is 0 (invalid). Valid ratings are between 0.5 and 5.0
        for user_ratings in self.ratings:
            user_ratings.append(0)

    # adds a new row
    def add_user(self):
        num_movies = len(self.ratings[0])

        # The default rating is 0 (invalid). Valid ratings are between 0.5 and 5.0
        self.ratings.append([0 for _ in range(num_movies)])

    def modify_rating(self, rating: bytes, r: int, c: int):
        self.ratings[r][c] = rating

    def modify_filled(self, filled_status: bytes, r: int, c: int):
        self.is_filled[r][c] = filled_status

    def matrix_completion(self):
        self.secure_matrix_completion_wrapper.prepare_data(self.ratings, self.is_filled)
        self.secure_matrix_completion_wrapper.train()

    # TODO: be able to generate statistics
    # def generate_statistic(self):
    #     self.matrix_completion()

    def receive_rating(self, r: int, c: int) -> bytes:
        # TODO: optimization is that if the rating already exists then there isn't a need to recompute the matrix completion. This could be in the form of a flag sent by the combiner.
        self.matrix_completion()
        return self.ratings[r][c]
