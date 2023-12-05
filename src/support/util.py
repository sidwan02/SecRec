##
## util.py: Dropbox @ CSCI1660 (Spring 2021)
##
## This file contains the util API.
##
## DO NOT MODIFY THE CONTENTS OF THIS FILE.
##
## ---
##
## Author: wschor
##
import code
import json
import base64
import tenseal as ts
from typing import List
import numpy as np


def _print_bytes(b: bytes) -> None:
    """
    A helper function to print bytes as base64.
    """
    print(base64.b64encode(b).decode("utf-8"))


def __bytes_to_b64(b: bytes) -> str:
    """
    A helper function that gives a base64 string representation of bytes.
    You probably do not need to use this directly.
    """
    return base64.b64encode(b).decode()


def __b64_to_bytes(b64: str) -> bytes:
    """
    A helper function that returns the bytes given by base64 string.
    You probably do not need to use this directly.
    """
    return base64.b64decode(b64)


def __detect_tags(s: str):
    return s[:3] == "^^^" and s[-3:] == "$$$"


def _prepare_bytes(o):
    """
    A helper funtion for ObjectToBytes
    """
    if isinstance(o, dict):
        result = {}
        for key, value in o.items():
            if isinstance(key, bytes):
                key = "^^^" + __bytes_to_b64(key) + "$$$"
            if isinstance(value, bytes):
                value = "^^^" + __bytes_to_b64(value) + "$$$"
            elif isinstance(value, dict) or isinstance(value, list):
                value = _prepare_bytes(value)
            result[key] = value
        return result

    if isinstance(o, list):
        result = []
        for item in o:
            if isinstance(item, bytes):
                item = "^^^" + __bytes_to_b64(item) + "$$$"
            elif isinstance(item, dict) or isinstance(item, list):
                item = _prepare_bytes(item)
            result.append(item)
        return result

    if isinstance(o, bytes):
        return "^^^" + __bytes_to_b64(o) + "$$$"

    elif isinstance(o, (int, str, float, bool)) or o is None:
        return o
    else:
        print(
            f"ERROR: Unserializable type {type(o)} detected! Valid types are [dict, list, int, str, float, bool, NoneType]"
        )
        raise ValueError


def _repair_bytes(o):
    """
    A helper funtion for ObjectToBytes
    """
    if isinstance(o, dict):
        result = {}
        for key, value in o.items():
            if isinstance(key, str):
                if __detect_tags(key):
                    key = __b64_to_bytes(key[3:-3])
            if isinstance(value, str):
                if __detect_tags(value):
                    value = __b64_to_bytes(value[3:-3])

            elif isinstance(value, dict) or isinstance(value, list):
                value = _repair_bytes(value)
            result[key] = value
        return result

    if isinstance(o, list):
        result = []
        for item in o:
            if isinstance(item, str):
                if __detect_tags(item):
                    item = __b64_to_bytes(item[3:-3])
            elif isinstance(item, dict) or isinstance(item, list):
                item = _repair_bytes(item)
            result.append(item)
        return result

    if isinstance(o, str):
        if __detect_tags(o):
            return __b64_to_bytes(o[3:-3])
        else:
            return o

    elif isinstance(o, (int, str, float, bool)) or o is None:
        return o
    else:
        print(
            f"ERROR: Undeserializable type {type(o)} detected! Valid types are [dict, list, int, str, float, bool, NoneType]"
        )
        raise ValueError


def ObjectToBytes(o: object) -> bytes:
    """
    A helper function that will serialize objects to bytes using JSON.
    It can serialize arbitrary nestings of lists and dictionaries containing ints, floats, booleans, strs, Nones, and bytes.

    A note on bytes and strings:
    This function encodes all bytes as base64 strings in order to be json compliant.
    The complimentary function, BytesToObject, will decode everything it detects to be a base64 string
    back to bytes. If you store a base64 formatted string, it would also be decoded to bytes.

    To alleviate this, the base64 string are prefixed with "^^^" and suffixed with "$$$", and the function
    checks for those tags instead.

    In the (unlikely) event you store a string with this format it will be decoded to bytes!
    """
    o = _prepare_bytes(o)
    return json.dumps(o).encode()


def BytesToObject(b: bytes) -> object:
    """
    A helper function that will deserialize bytes to an object using JSON. See caveats in ObjectToBytes().
    """
    obj = json.loads(b.decode())
    return _repair_bytes(obj)


## Custom Exceptions
class DropboxError(Exception):
    def __init__(self, msg="DROPBOX ERROR", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)


############# Tenseal Util #############
def convert_bytes_mat_to_ckks_mat(
    A: List[List[bytes]], encrypt_pk: ts.Context
) -> List[List[ts.CKKSVector]]:
    # return [
    #     [ts.lazy_ckks_vector_from(A[i][j]) for j in range(len(A[0]))]
    #     for i in range(len(A))
    # ]
    matrix_ckks = [[None for _ in range(len(A[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(A[0])):
            m = ts.lazy_ckks_vector_from(A[i][j])
            m.link_context(encrypt_pk)
            matrix_ckks[i][j] = m

    return matrix_ckks


def convert_ckks_mat_to_bytes_mat(A: List[List[ts.CKKSVector]]) -> List[List[bytes]]:
    return [[A[i][j].serialize() for j in range(len(A[0]))] for i in range(len(A))]


def decrypt_ckks_mat(
    A: List[List[ts.CKKSVector]], decrypt_sk: ts.Context
) -> List[List[float]]:
    plaintext_mat = np.empty((len(A), len(A[0])), dtype=float)

    for i in range(len(A)):
        for j in range(len(A[0])):
            m = A[i][j]
            m.link_context(decrypt_sk)
            plaintext_mat[i][j] = round(m.decrypt()[0], 4)

    return plaintext_mat


def encrypt_to_ckks_mat(
    A: List[List[float]], encrypt_pk: ts.Context
) -> List[List[ts.CKKSVector]]:
    cipher_matrix = [[None for _ in range(len(A[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(A[0])):
            cipher_matrix[i][j] = ts.ckks_vector(encrypt_pk, [A[i][j]])

    cipher_matrix = np.array(cipher_matrix)

    return cipher_matrix


def tenseal_util_test():
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

    encrypt_pk = ts.context_from(public_context)
    decrypt_sk = ts.context_from(secret_context)

    a = [[1, 2], [3, 4], [5, 6]]
    encrypted_mat = encrypt_to_ckks_mat(a, encrypt_pk)

    assert np.array_equal(decrypt_ckks_mat(encrypted_mat, decrypt_sk), a)


############# Tenseal Util End #############
