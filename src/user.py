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
        demo_user: bool,
    ):
        self.username = username
        self.combiner = combiner
        self.encrypt_pk = ts.context_from(public_context)
        self.decrypt_sk = ts.context_from(secret_context)
        self.demo = demo_user

    def send_rating(self, movie: str, rating: float):
        encrypted_rating = ts.ckks_vector(self.encrypt_pk, [rating])
        self.combiner.handle_rating(movie, encrypted_rating.serialize(), self.username)

    def receive_rating(self, movie: str) -> float:
        if self.demo:
            print("real ratings:")
            self.combiner.pretty_print_server_matrix(
                "ratings", self.encrypt_pk, self.decrypt_sk
            )
            print("bool user-filled:")
            self.combiner.pretty_print_server_matrix(
                "is_filled", self.encrypt_pk, self.decrypt_sk
            )

        rating_bytes: Union[bytes, None] = self.combiner.receive_rating(
            movie, self.username, self.demo
        )

        if self.demo:
            print("predicted ratings:")
            self.combiner.pretty_print_server_matrix(
                "predicted", self.encrypt_pk, self.decrypt_sk
            )

        if rating_bytes is None:
            return 0.0

        m = ts.lazy_ckks_vector_from(rating_bytes)
        m.link_context(self.decrypt_sk)

        rating_plain = round(m.decrypt()[0], 4)

        return rating_plain
