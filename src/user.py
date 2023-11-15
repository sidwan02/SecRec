from combiner import Combiner
from typing import Union
import tenseal as ts


class User:
    # for now, we only deal with having one private and public key
    def __init__(
        self,
        username: str,
        combiner: Combiner,
        public_context: bytes,
        secret_context: bytes,
    ):
        self.username = username
        self.combiner = combiner
        self.encrypt_pk = ts.context_from(public_context)
        self.decrypt_sk = ts.context_from(secret_context)

        # For tast 3, we will not have the global pk and private sk created before User creation. Instead, we will publish the created SEAL keys to the combiner which will then get the global pk, and then that will be set for all users
        # Combiner.publish_sk()

    def send_rating(self, movie: str, rating: float):
        encrypted_rating = ts.ckks_vector(self.encrypt_pk, [rating])
        Combiner.handle_rating(movie, encrypted_rating.serialize(), self.username)

    def receive_rating(self, movie: str) -> Union[None, float]:
        Combiner.receive_rating(movie, self.username)
