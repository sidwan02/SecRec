import tenseal as ts
import numpy as np

context = ts.context(
    ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40

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
