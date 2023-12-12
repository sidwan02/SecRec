import tenseal as ts
import numpy as np
import time

# controls precision of the fractional part
bits_scale = 26

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=32768,
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

public_key = ts.context_from(public_context)
secret_key = ts.context_from(secret_context)

# these can be lists
a_encrypted = ts.ckks_vector(public_key, np.array([1, 2, 3, 4]))

b_encrypted = ts.ckks_vector(public_key, np.array([1, 2, 3, 4]))

ans_encrypted = a_encrypted * b_encrypted
m1 = ts.lazy_ckks_vector_from(ans_encrypted.serialize())
m1.link_context(secret_key)
print(m1.decrypt())

ans2_encrypted = a_encrypted + b_encrypted
m2 = ts.lazy_ckks_vector_from(ans2_encrypted.serialize())
m2.link_context(secret_key)
print(m2.decrypt())

# working with matrices. When storing the matrix always store the elemnts as bytes by serializing.
cipher_matrix_bytes = [[None for _ in range(4)] for _ in range(2)]

for i in range(2):
    for j in range(4):
        a = ts.ckks_vector(public_key, [i + j]).serialize()
        cipher_matrix_bytes[i][j] = a

cipher_matrix_bytes = np.array(cipher_matrix_bytes)
print(
    "type: ", cipher_matrix_bytes.dtype
)  # we can't just initialize the np array with dtype bytes since the type is actually something else quite strange "  |S334566"

plaintext_matrix = np.empty((2, 4), dtype=float)

# when getting the ckks vector/tensor back from the bytes, we need to link to context. if public key, can perform FHE operations on it. if secret key, can decrypt the ciphertext
for i in range(2):
    for j in range(4):
        m2 = ts.lazy_ckks_vector_from(cipher_matrix_bytes[i][j])
        m2.link_context(secret_key)
        plaintext_matrix[i][j] = round(m2.decrypt()[0], 4)

print(plaintext_matrix)

# notice how here we link the public key to the tensors since we wanna perform computations with this mat later on for broadcast testing
# another reminder: FHE operations can only be performed on the ckks vector/tensor, not on the bytes
cipher_matrix_ckks = [[None for _ in range(4)] for _ in range(2)]
for i in range(2):
    for j in range(4):
        m = ts.lazy_ckks_vector_from(cipher_matrix_bytes[i, j])
        m.link_context(public_key)
        cipher_matrix_ckks[i][j] = m

cipher_matrix_ckks = np.array(cipher_matrix_ckks)

# numpy's broadcast operations work quite well
ans = cipher_matrix_ckks + cipher_matrix_ckks

plaintext_matrix = np.empty((2, 4), dtype=float)

for i in range(2):
    for j in range(4):
        m2 = ans[i, j]
        m2.link_context(secret_key)
        plaintext_matrix[i][j] = round(m2.decrypt()[0], 4)

print(plaintext_matrix)


# as do more complex broadcast, splicing, and assignments
cipher_matrix_ckks[1, :] += ans[0, :]

plaintext_matrix = np.empty((2, 4), dtype=float)

for i in range(2):
    for j in range(4):
        m2 = cipher_matrix_ckks[i, j]
        m2.link_context(secret_key)
        plaintext_matrix[i][j] = round(m2.decrypt()[0], 4)

print(plaintext_matrix)


start = time.time()

a = np.array(
    [
        [
            ts.ckks_vector(public_key, np.array([1])),
            ts.ckks_vector(public_key, np.array([2])),
            ts.ckks_vector(public_key, np.array([3])),
        ]
    ]
)
b = np.array(
    [
        [ts.ckks_vector(public_key, np.array([1]))],
        [ts.ckks_vector(public_key, np.array([2]))],
        [ts.ckks_vector(public_key, np.array([3]))],
    ]
)


c = a @ b

val = c[0, 0]
val.link_context(secret_key)
print(round(val.decrypt()[0], 4))
print(type(val))

end = time.time()

print(end - start)

# Multiplication scaling limit
enc_one = ts.ckks_vector(public_key, np.array([1]))

ans = enc_one

for i in range(100):
    ans.link_context(secret_key)
    decrypted = round(ans.decrypt()[0], 4)
    ans = ts.ckks_vector(public_key, np.array([decrypted]))
    ans *= enc_one

# Multiplication scaling limit
plain_one = ts.plain_tensor(np.array([1]))

ans = enc_one

for i in range(100):
    print(i)
    # ans.link_context(secret_key)
    # decrypted = round(ans.decrypt()[0], 4)
    # ans = ts.ckks_vector(public_key, np.array([decrypted]))
    ans += enc_one

ans.link_context(secret_key)
print(ans.decrypt())

print(type(enc_one))

# Testing matrix multiplication and dot product
one_hot_encrypted = np.array([
    ts.ckks_vector(public_key, np.array([0])),
    ts.ckks_vector(public_key, np.array([1])),
    ts.ckks_vector(public_key, np.array([0])),
    ts.ckks_vector(public_key, np.array([0])),
])
one_hot_encrypted2 = np.array([
    ts.ckks_vector(public_key, np.array([0])),
    ts.ckks_vector(public_key, np.array([1])),
    ts.ckks_vector(public_key, np.array([0]))
])
print(type(one_hot_encrypted))

mat_encrypted = np.array(
    [
        [
            ts.ckks_vector(public_key, [1]),
            ts.ckks_vector(public_key, [2]),
            ts.ckks_vector(public_key, [3]),
            ts.ckks_vector(public_key, [4]),
        ], 
        [
            ts.ckks_vector(public_key, [5]),
            ts.ckks_vector(public_key, [6]),
            ts.ckks_vector(public_key, [7]),
            ts.ckks_vector(public_key, [8]),
        ], 
        [
            ts.ckks_vector(public_key, [9]),
            ts.ckks_vector(public_key, [10]),
            ts.ckks_vector(public_key, [11]),
            ts.ckks_vector(public_key, [12]),
        ], 
    ]
)
print(type(mat_encrypted))

matmul_dot_encrypted = (mat_encrypted @ one_hot_encrypted).dot(one_hot_encrypted2)

matmul_dot_res = ts.lazy_ckks_vector_from(matmul_dot_encrypted.serialize())
matmul_dot_res.link_context(secret_key)
print(matmul_dot_res.decrypt()[0])

# Test matrix multiplication with a plain matrix
mat_unencrypted = np.array(
    [
        [1, 2, 3, 4], 
        [5, 6, 7, 8], 
        [9, 10, 11, 12], 
    ]
)
plain_matmul_dot_encrypted = (mat_unencrypted @ one_hot_encrypted).dot(one_hot_encrypted2)

plain_matmul_dot_res = ts.lazy_ckks_vector_from(plain_matmul_dot_encrypted.serialize())
plain_matmul_dot_res.link_context(secret_key)
print(plain_matmul_dot_res.decrypt()[0])

# Testing with bytes and serialization
print(type(plain_matmul_dot_encrypted.serialize()))
print(type(ts.lazy_ckks_vector_from(plain_matmul_dot_encrypted.serialize())))
# print(print(matmul_dot_encrypted.__dir__()))

matmul_dot_res2 = ts.lazy_ckks_vector_from(matmul_dot_encrypted.serialize())
matmul_dot_res2.link_context(matmul_dot_encrypted.context())
matmul_dot_res_square = matmul_dot_res2 * matmul_dot_res2
matmul_dot_res_square.link_context(secret_key)
print(matmul_dot_res_square.decrypt()[0])