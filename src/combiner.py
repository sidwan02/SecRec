from server import Server
import support.crypto as crypto
from user import User
from typing import List, Union
from support import util

# The combiner does not have the ability to store information (if it did, then the problem becomes to easy). It must store all information with the server.

class KeyDB:
    def __init__(self):
        # description: str -> key: bytes
        self.keys = {}


class Combiner:
    def __init__(self, server: Server):
        self.server = server
        self.keyDB = KeyDB
        
        # {user: row_id}
        self.setup_server_storage({}, "user-ownership")
        # {movie: col_id}
        self.setup_server_storage({}, "movie-ownership")
        
    def setup_server_storage(self, data, description):
        palintext = util.ObjectToBytes(data)
        
        data_encryptk, data_decryptk = crypto.AsymmetricKeyGen()
        data_verifyk, data_signk = crypto.SignatureKeyGen()
        
        # sign the plaintext
        data_sign = crypto.SignatureSign(data_signk, palintext)
        
        # concat the sign to the plaintext, then encrypt
        ciphertext = crypto.AsymmetricEncrypt(data_encryptk, data + data_sign)
        
        self.server.storage[description] = ciphertext
        
        self.keyDB.keys[f"{description}-encryptk"] = data_encryptk
        self.keyDB.keys[f"{description}-decryptk"] = data_decryptk
        self.keyDB.keys[f"{description}-signk"] = data_signk
        self.keyDB.keys[f"{description}-verifyk"] = data_verifyk
        
    def server_store(self, data, description):
        palintext = util.ObjectToBytes(data)
        
        # sign the plaintext
        data_sign = crypto.SignatureSign(self.keyDB.keys[f"{description}-signk"], palintext)
        
        # concat the sign to the plaintext, then encrypt
        ciphertext = crypto.AsymmetricEncrypt(self.keyDB.keys[f"{description}-encryptk"], data + data_sign)
        
        self.server.storage[description] = ciphertext
        
    def retrieve_server_storage(self, description: str):
        concat_plaintext = crypto.AsymmetricDecrypt(self.keyDB.keys[f"{description}-decryptk"], self.server.storage[description])
        plaintext, signature = concat_plaintext[:-512], concat_plaintext[-512:]
        
        if not crypto.SignatureVerify(self.keyDB.keys[f"{description}-verifyk"], plaintext, signature):
            pass
        
        return util.BytesToObject(plaintext)

    def handle_rating(self, movie: str, rating: float, user: str):
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
        
        self.server.add_rating(rating, user_ownership_map[user], movie_ownership_map[movie])

    def receive_rating(self, movie: str, username: str) -> Union[None, float]:
        pass
