from server import Server
import support.crypto as crypto
from typing import List, Union
from support import util
import tenseal as ts
import numpy as np
import pandas as pd
from tabulate import tabulate

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

    def setup_key_storage(self, sym_data_key, description):
        key_verifyk, key_signk = crypto.SignatureKeyGen()

        # sign the plaintext
        key_sign = crypto.SignatureSign(data_signk, sym_data_key)

        key_encryptk, key_decryptk = crypto.AsymmetricKeyGen()

        # concat the sign to the plaintext, then encrypt
        # ciphertext = crypto.AsymmetricEncrypt(data_encryptk, plaintext + data_sign)

        encrypted_sym_data_key = crypto.AsymmetricEncrypt(key_encryptk, data_key)

        # TODO: see later on if there is a way to concat the sign to the plaintext and not need to store the sign separately
        self.server.storage[f"{description}-key"] = encrypted_sym_data_key
        self.server.storage[f"{description}-key-sign"] = data_sign

        self.setup_server_storage(
            encrypted_key,
        )

        self.keyDB.keys[f"{description}-key-encryptk"] = key_encryptk
        self.keyDB.keys[f"{description}-key-decryptk"] = key_decryptk
        self.keyDB.keys[f"{description}-key-signk"] = data_signk
        self.keyDB.keys[f"{description}-key-verifyk"] = data_verifyk

    def setup_server_storage(self, data, description):
        if type(data) != bytes:
            plaintext = util.ObjectToBytes(data)
        else:
            plaintext = data

        data_key = crypto.PasswordKDF(
            description, crypto.ZERO_SALT, 128 // crypto.BITS_IN_BYTE
        )

        data_verifyk, data_signk = crypto.SignatureKeyGen()

        # sign the plaintext
        data_sign = crypto.SignatureSign(data_signk, plaintext)

        # concat the sign to the plaintext, then encrypt
        # ciphertext = crypto.AsymmetricEncrypt(data_encryptk, plaintext + data_sign)

        data_iv: bytes = crypto.SecureRandom(128 // BITS_IN_BYTE)
        ciphertext = crypto.SymmetricEncrypt(data_key, data_iv, plaintext)

        # TODO: see later on if there is a way to concat the sign to the plaintext and not need to store the sign separately
        self.server.storage[description] = ciphertext
        self.server.storage[f"{description}-sign"] = data_sign

        self.setup_key_storage(encrypted_key, description)

        self.keyDB.keys[f"{description}-key_encryptk"] = key_encryptk
        self.keyDB.keys[f"{description}-key_decryptk"] = key_decryptk
        self.keyDB.keys[f"{description}-signk"] = data_signk
        self.keyDB.keys[f"{description}-verifyk"] = data_verifyk

    def server_store(self, data, description):
        # print(data)
        # TODO: if the plaintext gets too large, the asymmetric encryption just failes for some reason.
        plaintext = util.ObjectToBytes(data)

        # sign the plaintext
        data_sign = crypto.SignatureSign(
            self.keyDB.keys[f"{description}-signk"], plaintext
        )

        # print(self.keyDB.keys[f"{description}-encryptk"])

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
            # print("setting movie ownership map")
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

    def receive_rating(self, movie: str, user: str) -> Union[None, bytes]:
        movie_ownership_map = self.retrieve_server_storage("movie-ownership")
        user_ownership_map = self.retrieve_server_storage("user-ownership")

        if movie not in movie_ownership_map or user not in user_ownership_map:
            print("Invalid user or movie!")
            return None

        rating_bytes: bytes = self.server.receive_rating(
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

    def test_print_clear_server_storage(
        self, encrypt_pk: ts.Context, decrypt_sk: ts.Context
    ):
        decrypted_ratings = np.array(
            util.decrypt_ckks_mat(
                util.convert_bytes_mat_to_ckks_mat(self.server.ratings, encrypt_pk),
                decrypt_sk,
            )
        )

        decrypted_is_filled = np.array(
            util.decrypt_ckks_mat(
                util.convert_bytes_mat_to_ckks_mat(self.server.is_filled, encrypt_pk),
                decrypt_sk,
            ),
        )

        movie_ownership_map = self.retrieve_server_storage("movie-ownership")
        user_ownership_map = self.retrieve_server_storage("user-ownership")

        """
        column_header = [[None for _ in range(len(movie_ownership_map))]]
        for movie, col_id in movie_ownership_map.items():
            column_header[0][col_id] = movie
        # print(f"column_header: {column_header}")

        # the +1 is for the top left cell which is empty (intersection between column names and row names)
        row_header = [[""] for _ in range(len(user_ownership_map) + 1)]
        for user, row_id in user_ownership_map.items():
            row_header[row_id + 1][0] = user
        # print(f"row_header: {row_header}")

        decrypted_ratings = np.r_[column_header, decrypted_ratings]
        decrypted_is_filled = np.r_[column_header, decrypted_is_filled]

        # print(decrypted_ratings)
        # print(decrypted_is_filled)

        decrypted_ratings = np.c_[row_header, decrypted_ratings]
        decrypted_is_filled = np.c_[row_header, decrypted_is_filled]

        print(" | ", end="")
        for name in columns:
            print(name.center(10, " "), end="")
            print(" | ", end="")
        print("")
        """

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

        # print(decrypted_is_filled)

        df_is_filled = pd.DataFrame(
            decrypted_is_filled, columns=column_header, index=row_header
        )
        # print(tabulate(df_is_filled, headers="keys", tablefmt="psql"))
