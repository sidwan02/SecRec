from combiner import Combiner
from user import User
from server import Server
from matrix_completion import SecureSVD, SecureClip, SecureClearDivision
import support.crypto as crypto
import tenseal as ts
import base64
from support.util import tenseal_util_test


def setup_contexts():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.generate_galois_keys()
    context.global_scale = 2**40

    secret_context = context.serialize(save_secret_key=True)

    context.make_context_public()
    public_context = context.serialize()

    return public_context, secret_context


def test_handle_rating(server, combiner, public_context, secret_context):
    # TODO: modularize this into a helper function called in multiple tests + other places
    encrypt_pk = ts.context_from(public_context)
    decrypt_sk = ts.context_from(secret_context)

    combiner.handle_rating("movie1", ts.ckks_vector(encrypt_pk, [1]).serialize(), "Bob")
    # TODO: make all util helpers used by util.f rather than f in all files (make it consistent)
    combiner.test_print_clear_server_storage(decrypt_sk)
    combiner.handle_rating("movie1", ts.ckks_vector(encrypt_pk, [3]).serialize(), "Bob")
    combiner.test_print_clear_server_storage(decrypt_sk)
    combiner.handle_rating(
        "movie1", ts.ckks_vector(encrypt_pk, [2]).serialize(), "Jason"
    )
    combiner.test_print_clear_server_storage(decrypt_sk)
    combiner.handle_rating("movie2", ts.ckks_vector(encrypt_pk, [2]).serialize(), "Bob")
    combiner.test_print_clear_server_storage(decrypt_sk)
    combiner.handle_rating(
        "movie10", ts.ckks_vector(encrypt_pk, [2]).serialize(), "Alice"
    )
    combiner.test_print_clear_server_storage(decrypt_sk)
    combiner.handle_rating(
        "movie1", ts.ckks_vector(encrypt_pk, [4.3]).serialize(), "Alice"
    )
    combiner.test_print_clear_server_storage(decrypt_sk)


if __name__ == "__main__":
    # generate the global pk (this is naively done as a POC measure)
    # global_pk, global_sk = crypto.AsymmetricKeyGen()

    # Create all the users. Also, create the FHE key pair.
    public_context, secret_context = setup_contexts()

    server = Server(
        public_context,
        SecureSVD(public_context, secret_context),
        SecureClip(public_context, secret_context),
        SecureClearDivision(secret_context),
    )
    combiner = Combiner(server, public_context)

    user1 = User("Bob", combiner, public_context, secret_context)

    ######### TESTS START #########
    # combiner.test_server_storage()
    # tenseal_util_test()
    test_handle_rating(server, combiner, public_context, secret_context)
    ######### TESTS END #########
