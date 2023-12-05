# B1. Context and key setup
# ---------------------------
import numpy as np
from Pyfhel import Pyfhel
import time

# Feel free to change this number!
n_mults = 8

HE = Pyfhel(
    key_gen=True,
    context_params={
        "scheme": "CKKS",
        "n": 2**14,  # For CKKS, n/2 values can be encoded in a single ciphertext.
        "scale": 2**30,  # Each multiplication grows the final scale
        "qi_sizes": [60]
        + [30] * n_mults
        + [60]  # Number of bits of each prime in the chain.
        # Intermediate prime sizes should be close to log2(scale).
        # One per multiplication! More/higher qi_sizes means bigger
        #  ciphertexts and slower ops.
    },
)
HE.relinKeyGen()
print("\nB1. CKKS context generation")
print(f"\t{HE}")

#
# B2. CKKS Array Encoding & Encryption
# ----------------------------------------
arr_x = np.array([1], dtype=np.float64)
arr_y = np.array([2], dtype=np.float64)

ctxt_x = HE.encryptFrac(arr_x)
ctxt_y = HE.encryptFrac(arr_y)

print("\nB2. Fixed-point Encoding & Encryption, ")
print("->\tarr_x ", arr_x, "\n\t==> ctxt_x ", ctxt_x)
print("->\tarr_y ", arr_y, "\n\t==> ctxt_y ", ctxt_y)

#
# B3. Multiply n_mult times!
# -----------------------------
# Besides rescaling, we also need to perform rescaling & mod switching. Luckily
# Pyfhel does it for us by calling HE.align_mod_n_scale() before each operation.

_r = lambda x: np.round(x, decimals=6)[:4]
print(f"B3. Securely multiplying {n_mults} times!")
for step in range(1, n_mults + 1):
    ctxt_x *= ctxt_y  # Multiply in-place --> implicit align_mod_n_scale()
    ctxt_x = ~(ctxt_x)  # Always relinearize after each multiplication!
    print(f"\tStep {step}:  res {_r(HE.decryptFrac(ctxt_x))[0]}")
try:
    ctxt_x *= ctxt_y
except ValueError as e:
    assert str(e) == "scale out of bounds"
    print(f"If we multiply further we get: {str(e)}")
print("---------------------------------------")


start = time.time()

a = np.array(
    [
        [
            HE.encryptFrac(np.array([1], dtype=np.float64)),
            HE.encryptFrac(np.array([1], dtype=np.float64)),
            HE.encryptFrac(np.array([1], dtype=np.float64)),
        ]
    ]
)
b = np.array(
    [
        [HE.encryptFrac(np.array([1], dtype=np.float64))],
        [HE.encryptFrac(np.array([1], dtype=np.float64))],
        [HE.encryptFrac(np.array([1], dtype=np.float64))],
    ]
)


c = a @ b


plain_mat = np.vectorize(lambda c: _r(HE.decryptFrac(c))[0])(c)

end = time.time()

print(plain_mat)
print(end - start)

start = time.time()

a = np.array(
    [
        [
            1,
            1,
            1,
        ]
    ]
)
b = np.array(
    [
        [1],
        [1],
        [1],
    ]
)

c = a @ b
print(c)

end = time.time()

print(end - start)
