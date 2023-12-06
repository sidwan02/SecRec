from combiner import Combiner
from user import User
from server import Server
import secure_algos
import support.crypto as crypto
import tenseal as ts
import base64
from support.util import tenseal_util_test
import math
import random
import numpy as np


def setup_contexts():
    # controls precision of the fractional part
    bits_scale = 26

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[
            31,
            bits_scale,
            bits_scale,
            bits_scale,
            bits_scale,
            bits_scale,
            bits_scale,
            31,
        ],
    )
    context.generate_galois_keys()
    context.global_scale = 2**bits_scale

    secret_context = context.serialize(save_secret_key=True)

    context.make_context_public()
    public_context = context.serialize()

    return public_context, secret_context


def test_combiner_rating_logic(server, combiner, public_context, secret_context):
    encrypt_pk = ts.context_from(public_context)
    decrypt_sk = ts.context_from(secret_context)

    # Tests for handle_rating
    user1 = User("Bob", combiner, public_context, secret_context, True)
    user2 = User("Alice", combiner, public_context, secret_context, True)
    user3 = User("Jason", combiner, public_context, secret_context, True)

    combiner.handle_rating("movie1", ts.ckks_vector(encrypt_pk, [1]).serialize(), "Bob")
    combiner.handle_rating("movie1", ts.ckks_vector(encrypt_pk, [3]).serialize(), "Bob")
    combiner.handle_rating(
        "movie1", ts.ckks_vector(encrypt_pk, [2]).serialize(), "Jason"
    )
    combiner.handle_rating("movie2", ts.ckks_vector(encrypt_pk, [2]).serialize(), "Bob")
    combiner.handle_rating(
        "movie3", ts.ckks_vector(encrypt_pk, [2]).serialize(), "Alice"
    )
    combiner.handle_rating(
        "movie1", ts.ckks_vector(encrypt_pk, [4.3]).serialize(), "Alice"
    )

    # Tests for recieve_rating

    print(user1.receive_rating("movie1"))
    # assert math.isclose(user2.receive_rating("movie1"), 4.3)


def demo(server, combiner, public_context, secret_context):
    encrypt_pk = ts.context_from(public_context)
    decrypt_sk = ts.context_from(secret_context)

    num_users = 5
    num_movies = 5

    users = []
    for i in range(1, num_users + 1):
        users.append(User(f"User {i}", combiner, public_context, secret_context, True))

    movies = [f"Movie {i}" for i in range(1, num_movies + 1)]

    for i in range(len(movies) * len(users)):
        add_rating = np.random.rand() < 0.3

        if add_rating:
            user = users[random.randint(0, num_users - 1)]
            movie = movies[random.randint(0, num_movies - 1)]

            rating = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            user.send_rating(movie, rating)

    # Ratings test
    print(users[0].receive_rating(f"Movie {num_movies}"))


def reset_state():
    public_context, secret_context = setup_contexts()

    server = Server(
        public_context,
        secure_algos.SecureMatrixErrorReset(public_context, secret_context),
        secure_algos.SecureSVD(public_context, secret_context),
        secure_algos.SecureClip(public_context, secret_context),
        secure_algos.SecureClearDivision(secret_context),
    )
    combiner = Combiner(server, public_context)

    return public_context, secret_context, server, combiner


if __name__ == "__main__":
    public_context, secret_context, server, combiner = reset_state()
    user1 = User("Bob", combiner, public_context, secret_context, True)

    ######### TESTS START #########
    tenseal_util_test()
    combiner.test_server_storage()
    test_combiner_rating_logic(server, combiner, public_context, secret_context)
    ######### TESTS END #########

    public_context, secret_context, server, combiner = reset_state()

    ######### DEMO ###########
    demo(server, combiner, public_context, secret_context)
    ######### DEMO END #######
