import benchmark
import repl
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Run benchmarks for loss convergence
    no_fhe_losses, fhe_losses = benchmark.secure_insecure_matrix_completion_runtime_benchmark()
    plt.plot(no_fhe_losses)
    plt.plot(fhe_losses)
    plt.show()

    # Run benchmarks for FHE runtime of matrix completion algorithms

    # Run benchmarks for FHE + PIR runtime of weight retrieval