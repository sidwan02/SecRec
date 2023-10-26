from combiner import Combiner
from user import User
import support.crypto as crypto

if __name__ == "__main__":
    # generate the global pk (this is naively done as a POC measure)
    global_pk, global_sk = crypto.AsymmetricKeyGen()
