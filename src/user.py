from combiner import Combiner
from typing import Union


class User:
    # for now, we only deal with having one private and public key
    def __init__(self, username: str, combiner: Combiner, global_pk: bytes):
        self.username = username
        self.combiner = combiner
        self.global_pk = global_pk

        # right now, this is the same for all users. Later on, with TFHE, it will be different.
        Combiner.publish_sk()

    def send_rating(self, movie: str, rating: float):
        Combiner.handle_rating(movie, rating, self.username)

    def receive_rating(self, movie: str) -> Union[None, float]:
        Combiner.receive_rating(movie, self.username)
