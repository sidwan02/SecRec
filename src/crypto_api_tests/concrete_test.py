from concrete import fhe


def add(x, y):
    return x + y


compiler = fhe.Compiler(add, {"x": "encrypted", "y": "clear"})

inputset = [(2, 3), (0, 0), (1, 6), (7, 7), (7, 1)]
circuit = compiler.compile(inputset)

x = 4
y = 4

clear_evaluation = add(x, y)
homomorphic_evaluation = circuit.encrypt_run_decrypt(x, y)

print(x, "+", y, "=", clear_evaluation, "=", homomorphic_evaluation)
