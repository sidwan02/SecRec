from combiner import Combiner
from user import User
from server import Server
from matrix_completion import SecureSVD, SecureClip, SecureClearDivision
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


def demo(server, combiner, public_context, secret_context):
    encrypt_pk = ts.context_from(public_context)
    decrypt_sk = ts.context_from(secret_context)

    num_users = 10
    num_movies = 10

    users = []
    for i in range(1, num_users + 1):
        users.append(User(f"User {i}", combiner, public_context, secret_context))

    movies = [f"Movie {i}" for i in range(1, num_movies + 1)]

    for i in range(len(movies) * len(users)):
        add_rating = np.random.rand() < 0.8

        if add_rating:
            user = users[random.randint(0, num_users - 1)]
            movie = movies[random.randint(0, num_movies - 1)]

            rating = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            user.send_rating(movie, rating)

    # Ratings test
    print(users[0].receive_rating("Movie 3"))


if __name__ == "__main__":
    # Create all the users. Also, create the FHE key pair.
    public_context, secret_context = setup_contexts()

    server = Server(
        public_context,
        secret_context,
        SecureSVD(public_context, secret_context),
        SecureClip(public_context, secret_context),
        SecureClearDivision(secret_context),
    )
    combiner = Combiner(server, public_context)

    demo(server, combiner, public_context, secret_context)
