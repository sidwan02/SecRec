class Server:
    def __init__(self):
        self.storage = {}
        
        # N x M matrix with N users and M movies
        # encryption and signature of this is maintained separately by the server
        self.ratings = [[]]
        
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
        
    def add_rating(self, rating: float, r: int, c: int):
        pass
    
    def matrix_completion(self):
        pass
    
    def generate_statistic(self):
        self.matrix_completion()
        
        pass