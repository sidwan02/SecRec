import benchmark
import repl
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Run benchmarks for loss convergence
    no_fhe_losses, fhe_losses = benchmark.secure_insecure_matrix_completion_runtime_benchmark()
    plt.plot(no_fhe_losses)
    plt.plot(fhe_losses)
    plt.savefig("loss_curves.png")

    no_fhe_robust_losses, fhe_robust_losses = benchmark.secure_insecure_robust_matrix_completion_loss_convergence_benchmark()
    plt.figure()
    plt.plot(no_fhe_robust_losses)
    plt.plot(fhe_robust_losses)
    plt.savefig("robust_loss_curves.png")

    # Run benchmarks for FHE runtime of matrix completion algorithms

    # Run benchmarks for FHE + PIR runtime of weight retrieval