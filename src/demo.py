from combiner import Combiner
from user import User
from server import Server
from secure_algos import SecureSVD, SecureClip, SecureClearDivision
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
