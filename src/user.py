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
        # self.combiner.publish_sk()

    def send_rating(self, movie: str, rating: float):
        encrypted_rating = ts.ckks_vector(self.encrypt_pk, [rating])
        self.combiner.handle_rating(movie, encrypted_rating.serialize(), self.username)

    def receive_rating(self, movie: str) -> float:
        print("RECEIVE RATING")
        self.combiner.test_print_clear_server_storage(self.encrypt_pk, self.decrypt_sk)

        rating_bytes: Union[bytes, None] = self.combiner.receive_rating(
            movie, self.username
        )

        self.combiner.test_print_clear_server_storage(self.encrypt_pk, self.decrypt_sk)

        if rating_bytes is None:
            return 0.0

        m = ts.lazy_ckks_vector_from(rating_bytes)
        m.link_context(self.decrypt_sk)

        rating_plain = round(m.decrypt()[0], 4)
        print(rating_plain)

        return rating_plain
