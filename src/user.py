from combiner import Combiner
from typing import Union


class User:
    # for now, we only deal with having one private and public key
    def __init__(self, username: str, combiner: Combiner, global_pk: bytes, private_sk: bytes):
        self.username = username
        self.combiner = combiner
        self.global_pk = global_pk
        self.private_sk = private_sk

        # For tast 3, we will not have the global pk and private sk created before User creation. Instead, we will publish the created SEAL keys to the combiner which will then get the global pk, and then that will be set for all users
        # Combiner.publish_sk()

    def send_rating(self, movie: str, rating: float):
        Combiner.handle_rating(movie, rating, self.username)

    def receive_rating(self, movie: str) -> Union[None, float]:
        Combiner.receive_rating(movie, self.username)
