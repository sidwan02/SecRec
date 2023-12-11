from server import Server
import support.crypto as crypto
from typing import List, Union
from support import util
import tenseal as ts
import numpy as np
import pandas as pd
from tabulate import tabulate


class KeyDB:
    def __init__(self):
        # description: str -> key: bytes
        self.keys = {}


class Combiner:
    def __init__(
        self, 
        server: Server, 
        public_context: bytes,
        do_pir : bool = False
    ):
        self.server = server
        self.keyDB = KeyDB()
        self.encrypt_pk = ts.context_from(public_context)
        self.do_pir = do_pir

        # {user: row_id}
        self.setup_server_storage({}, "user-ownership")
        # {movie: col_id}
        self.setup_server_storage({}, "movie-ownership")

    # data key is a symmetric key
    # we must perform this hybrid encryption since RSA (asymmetric encryption) has an upper bound plaintext size, while symmetric AES uses block chaining and can scale with plaintest size.
    def setup_key_storage(self, data_key, description):
        key_verifyk, key_signk = crypto.SignatureKeyGen()

        # sign the plaintext
        key_sign = crypto.SignatureSign(key_signk, data_key)

        # generate the keys for
        key_encryptk, key_decryptk = crypto.AsymmetricKeyGen()

        ciphertext_key = crypto.AsymmetricEncrypt(key_encryptk, data_key)

        self.server.storage[f"{description}"] = ciphertext_key
        self.server.storage[f"{description}-sign"] = key_sign

        self.keyDB.keys[f"{description}-encryptk"] = key_encryptk
        self.keyDB.keys[f"{description}-decryptk"] = key_decryptk

        self.keyDB.keys[f"{description}-signk"] = key_signk
        self.keyDB.keys[f"{description}-verifyk"] = key_verifyk

    def setup_server_storage(self, data, description):
        plaintext = util.ObjectToBytes(data)

        data_key = crypto.PasswordKDF(
            description, crypto.ZERO_SALT, 128 // crypto.BITS_IN_BYTE
        )

        data_verifyk, data_signk = crypto.SignatureKeyGen()

        # sign the plaintext
        data_sign = crypto.SignatureSign(data_signk, plaintext)

        # later, could also have a random iv (we have the api for it in support.crypto)
        ciphertext = crypto.SymmetricEncrypt(data_key, crypto.ZERO_SALT, plaintext)

        self.server.storage[description] = ciphertext
        self.server.storage[f"{description}-sign"] = data_sign

        self.setup_key_storage(data_key, f"{description}-key")

        # right now, we're storing both the sign and verify keys in the trusted key storage. Later, ideally, only the public keys should be stored here and the private keys should be stored on the server.
        self.keyDB.keys[f"{description}-signk"] = data_signk
        self.keyDB.keys[f"{description}-verifyk"] = data_verifyk

    def server_store(self, data, description):
        plaintext = util.ObjectToBytes(data)

        # sign the plaintext
        data_sign = crypto.SignatureSign(
            self.keyDB.keys[f"{description}-signk"], plaintext
        )

        data_key = self.retrieve_server_storage_key(f"{description}-key")

        ciphertext = crypto.SymmetricEncrypt(data_key, crypto.ZERO_SALT, plaintext)

        self.server.storage[description] = ciphertext
        self.server.storage[f"{description}-sign"] = data_sign

    def retrieve_server_storage_key(self, description: str) -> bytes:
        plaintext = crypto.AsymmetricDecrypt(
            self.keyDB.keys[f"{description}-decryptk"], self.server.storage[description]
        )

        signature = self.server.storage[f"{description}-sign"]

        if not crypto.SignatureVerify(
            self.keyDB.keys[f"{description}-verifyk"], plaintext, signature
        ):
            print("Key has been corrupted!")
            return None

        return plaintext

    def retrieve_server_storage(self, description: str):
        data_key = self.retrieve_server_storage_key(f"{description}-key")

        plaintext = crypto.SymmetricDecrypt(data_key, self.server.storage[description])

        signature = self.server.storage[f"{description}-sign"]

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

    def receive_rating(self, movie: str, user: str, demo) -> Union[None, bytes]:
        movie_ownership_map = self.retrieve_server_storage("movie-ownership")
        user_ownership_map = self.retrieve_server_storage("user-ownership")

        if movie not in movie_ownership_map or user not in user_ownership_map:
            print("Invalid user or movie!")
            return None

        if self.do_pir:
            user_one_hot = [1 if curr_user == user else 0 for curr_user, _ in user_ownership_map.items()]
            user_one_hot_encrypted = ts.ckks_vector(self.encrypt_pk, user_one_hot)
            movie_one_hot = [1 if curr_movie == movie else 0 for curr_movie, _ in movie_ownership_map.items()]
            movie_one_hot_encrypted = ts.ckks_vector(self.encrypt_pk, movie_one_hot)

            rating_bytes = self.server.receive_rating_pir(
                user_one_hot_encrypted, movie_one_hot_encrypted, demo
            )

            return rating_bytes
        else:
            rating_bytes = self.server.receive_rating(
                user_ownership_map[user], movie_ownership_map[movie], demo
            )

            return rating_bytes

    def test_server_storage(self):
        a = {"hi": 1}
        self.setup_server_storage(a, "test 1")
        assert a == self.retrieve_server_storage("test 1")

        b = {"bye": 100}
        self.server_store(b, "test 1")
        assert b == self.retrieve_server_storage("test 1")

    def pretty_print_server_matrix(
        self, description: str, encrypt_pk: ts.Context, decrypt_sk: ts.Context
    ):
        decrypted_ratings = np.array(
            util.decrypt_ckks_mat(
                util.convert_bytes_mat_to_ckks_mat(
                    self.server.matrices[description], encrypt_pk
                ),
                decrypt_sk,
            )
        )

        movie_ownership_map = self.retrieve_server_storage("movie-ownership")
        user_ownership_map = self.retrieve_server_storage("user-ownership")

        column_header = [None for _ in range(len(movie_ownership_map))]
        for movie, col_id in movie_ownership_map.items():
            column_header[col_id] = movie

        row_header = [None for _ in range(len(user_ownership_map))]
        for user, row_id in user_ownership_map.items():
            row_header[row_id] = user

        df_ratings = pd.DataFrame(
            decrypted_ratings, columns=column_header, index=row_header
        )
        print(tabulate(df_ratings, headers="keys", tablefmt="psql"))
