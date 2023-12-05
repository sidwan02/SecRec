from combiner import Combiner
from user import User
from server import Server
from matrix_completion import SecureSVD, SecureClip, SecureClearDivision
import support.crypto as crypto
import tenseal as ts
import base64
from support.util import tenseal_util_test

if __name__ == "__main__":
    # generate the global pk (this is naively done as a POC measure)
    # global_pk, global_sk = crypto.AsymmetricKeyGen()

    # Create all the users. Also, create the FHE key pair.
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

    server = Server(
        public_context,
        SecureSVD(public_context, secret_context),
        SecureClip(public_context, secret_context),
        SecureClearDivision(secret_context),
    )
    combiner = Combiner(server, public_context)

    user1 = User("Bob", combiner, public_context, secret_context)

    ######### TESTS START #########
    combiner.test_server_storage()
    tenseal_util_test()
    ######### TESTS END #########
