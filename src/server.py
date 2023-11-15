class Server:
    def __init__(self):
        self.storage = {}

        # N x M matrix with N users and M movies
        # encryption and signature of this is maintained separately by the server
        self.ratings = [[]]

        # N x M matrix with N users and M movies
        # The values are either 0 (value not filled) or 1 (value has been filled)
        # encryption and signature of this is maintained separately by the server
        self.is_filled = [[]]

        # it does not make sense for us to have a matrix of bools where each bool is a 0 or 1 since ideally we wanna generate train test splits of indices + we need the ability to perform encryptions with the encryption key so we would be able to determine which entries are 0 and 1 anyways by manually checking against each row.

    # adds a new col
    def add_movie(self):
        # TODO: looking at the matrix completion assignment from 1952q, figure out what the default value should be
        for user_ratings in self.ratings:
            user_ratings.append(None)

    # adds a new row
    def add_user(self):
        num_movies = len(self.ratings[0])

        # TODO: looking at the matrix completion assignment from 1952q, figure out what the default value should be
        self.ratings.append([None for _ in range(num_movies)])

    def modify_rating(self, rating: bytes, r: int, c: int):
        self.ratings[r][c] = rating

    def modify_filled(self, filled_status: bytes, r: int, c: int):
        self.is_filled[r][c] = filled_status

    def matrix_completion(self):
        pass

    def generate_statistic(self):
        self.matrix_completion()

        pass
