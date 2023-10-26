# The combiner does not have the ability to store information (if it did, then the problem becomes to easy). It must store all information with the server.


class Combiner:
    def __init__(self):
        # right now, this is the same for all users. Later on, with TFHE, it will be different.
        Combiner.publish_sk()

    def handle_rating(self, movie: str, rating: float):
        Combiner.handle_rating(movie, rating, self.username)

    def receive_rating(self, movie: str, username: str) -> Union[None, float]:
        pass
