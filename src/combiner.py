from server import Server
import support.crypto as crypto
from typing import List, Union
from support import util
import tenseal as ts

# The combiner does not have the ability to store information (if it did, then the problem becomes to easy). It must store all information with the server.


class KeyDB:
    def __init__(self):
        # description: str -> key: bytes
        self.keys = {}


class Combiner:
    def __init__(self, server: Server, public_context: bytes):
        self.server = server
        self.keyDB = KeyDB()
        self.encrypt_pk = ts.context_from(public_context)

        # {user: row_id}
        self.setup_server_storage({}, "user-ownership")
        # {movie: col_id}
        self.setup_server_storage({}, "movie-ownership")

    def setup_server_storage(self, data, description):
        plaintext = util.ObjectToBytes(data)

        data_encryptk, data_decryptk = crypto.AsymmetricKeyGen()
        data_verifyk, data_signk = crypto.SignatureKeyGen()

        # sign the plaintext
        data_sign = crypto.SignatureSign(data_signk, plaintext)

        # concat the sign to the plaintext, then encrypt
        # ciphertext = crypto.AsymmetricEncrypt(data_encryptk, plaintext + data_sign)
        ciphertext = crypto.AsymmetricEncrypt(data_encryptk, plaintext)

        # TODO: see later on if there is a way to concat the sign to the plaintext and not need to store the sign separately
        self.server.storage[description] = ciphertext
        self.server.storage[f"{description}-sign"] = data_sign

        self.keyDB.keys[f"{description}-encryptk"] = data_encryptk
        self.keyDB.keys[f"{description}-decryptk"] = data_decryptk
        self.keyDB.keys[f"{description}-signk"] = data_signk
        self.keyDB.keys[f"{description}-verifyk"] = data_verifyk

    def server_store(self, data, description):
        plaintext = util.ObjectToBytes(data)

        # sign the plaintext
        data_sign = crypto.SignatureSign(
            self.keyDB.keys[f"{description}-signk"], plaintext
        )

        # concat the sign to the plaintext, then encrypt
        ciphertext = crypto.AsymmetricEncrypt(
            self.keyDB.keys[f"{description}-encryptk"], plaintext
        )

        self.server.storage[description] = ciphertext
        self.server.storage[f"{description}-sign"] = data_sign

    def retrieve_server_storage(self, description: str):
        plaintext = crypto.AsymmetricDecrypt(
            self.keyDB.keys[f"{description}-decryptk"], self.server.storage[description]
        )

        # TODO: get the signature separately.
        signature = self.server.storage[f"{description}-sign"]
        # plaintext, signature = concat_plaintext[:-512], concat_plaintext[-512:]

        if not crypto.SignatureVerify(
            self.keyDB.keys[f"{description}-verifyk"], plaintext, signature
        ):
            print("Plaintext has been corrupted!")
            return None

        return util.BytesToObject(plaintext)

    def handle_rating(self, movie: str, rating: bytes, user: str):
        movie_ownership_map = self.retrieve_server_storage("movie-ownership")
        user_ownership_map = self.retrieve_server_storage("user-ownership")

        if movie not in movie_ownership_map:
            self.server.add_movie()
            movie_ownership_map[movie] = len(movie_ownership_map)
            self.server_store(movie_ownership_map, "movie-ownership")

        if user not in user_ownership_map:
            self.server.add_user()
            user_ownership_map[user] = len(user_ownership_map)
            self.server_store(user_ownership_map, "user-ownership")

        one_encrypted = ts.ckks_vector(self.encrypt_pk, [1])

        self.server.modify_filled(
            one_encrypted.serialize(),
            user_ownership_map[user],
            movie_ownership_map[movie],
        )
        self.server.modify_rating(
            rating, user_ownership_map[user], movie_ownership_map[movie]
        )

    def receive_rating(self, movie: str, username: str) -> bytes:
        movie_ownership_map = self.retrieve_server_storage("movie-ownership")
        user_ownership_map = self.retrieve_server_storage("user-ownership")

        rating_bytes: bytes = Server.receive_rating(
            user_ownership_map[user], movie_ownership_map[movie]
        )

        return rating_bytes

    def test_server_storage(self):
        a = {"hi": 1}
        self.setup_server_storage(a, "test 1")
        assert a == self.retrieve_server_storage("test 1")

        b = {"bye": 100}
        self.server_store(b, "test 1")
        assert b == self.retrieve_server_storage("test 1")
