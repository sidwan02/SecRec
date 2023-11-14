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
        # {user: set(filled entries: tuple(int, int))}
        self.setup_server_storage({}, "owned-entries")
        # N x M matrix with N users and M movies
        self.setup_server_storage({}, "ratings")
        
    def setup_server_storage(self, data, description):
        palintext = util.ObjectToBytes(data)
        
        data_encryptk, data_decryptk = crypto.AsymmetricKeyGen()
        data_verifyk, data_signk = crypto.SignatureKeyGen()
        
        # sign the plaintext
        data_sign = crypto.SignatureSign(data_signk, palintext)
        
        # concat the sign to the plaintext, then encrypt
        ciphertext = crypto.AsymmetricEncrypt(data_encryptk, data + data_sign)
        
        self.server.storage[description] = ciphertext
        

    def handle_rating(self, movie: str, rating: float):
        
        Combiner.handle_rating(movie, rating, self.username)

    def receive_rating(self, movie: str, username: str) -> Union[None, float]:
        pass
