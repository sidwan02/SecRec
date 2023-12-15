import benchmark
import repl
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Run benchmarks for loss convergence
    no_fhe_losses, fhe_losses = benchmark.secure_insecure_matrix_completion_runtime_benchmark()
    plt.plot(no_fhe_losses)
    plt.plot(fhe_losses)
    plt.savefig("loss_curves.pdf", format="pdf")

    no_fhe_robust_losses, fhe_robust_losses = benchmark.secure_insecure_robust_matrix_completion_loss_convergence_benchmark()
    plt.figure()
    plt.plot(no_fhe_robust_losses)
    plt.plot(fhe_robust_losses)
    plt.savefig("robust_loss_curves.pdf", format="pdf")

    # Run benchmarks for FHE runtime of matrix completion algorithms (test size 5, 10, 20, and 40 if time)
    no_fhe_time5, fhe_time5 = benchmark.secure_insecure_matrix_completion_runtime_benchmark(rows=5, cols=5)
    print("---------------------- Time ----------------------")
    print(no_fhe_time5, fhe_time5) # 0.014873504638671875 457.088915348053
    no_fhe_time10, fhe_time10 = benchmark.secure_insecure_matrix_completion_runtime_benchmark(rows=10, cols=10)
    print("---------------------- Time ----------------------")
    print(no_fhe_time10, fhe_time10) # 0.0466914176940918 3000.4046161174774
    no_fhe_time20, fhe_time20 = benchmark.secure_insecure_matrix_completion_runtime_benchmark(rows=20, cols=20)
    print("---------------------- Time ----------------------")
    print(no_fhe_time20, fhe_time20)
    # no_fhe_time40, fhe_time40 = benchmark.secure_insecure_matrix_completion_runtime_benchmark(rows=40, cols=40)
    # print("---------------------- Time ----------------------")
    # print(no_fhe_time40, fhe_time40)

    no_fhe_robust_time5, fhe_robust_time5 = benchmark.secure_insecure_robust_matrix_completion_runtime_benchmark(rows=5, cols=5)
    print("---------------------- Time ----------------------")
    print(no_fhe_robust_time5, fhe_robust_time5)
    no_fhe_robust_time10, fhe_robust_time10 = benchmark.secure_insecure_robust_matrix_completion_runtime_benchmark(rows=10, cols=10)
    print("---------------------- Time ----------------------")
    print(no_fhe_robust_time10, fhe_robust_time10)
    no_fhe_robust_time20, fhe_robust_time20 = benchmark.secure_insecure_robust_matrix_completion_runtime_benchmark(rows=20, cols=20)
    print("---------------------- Time ----------------------")
    print(no_fhe_robust_time20, fhe_robust_time20)
    # no_fhe_robust_time40, fhe_robust_time40 = benchmark.secure_insecure_robust_matrix_completion_runtime_benchmark(rows=40, cols=40)
    # print("---------------------- Time ----------------------")
    # print(no_fhe_robust_time40, fhe_robust_time40)

    # Run benchmarks for FHE + PIR runtime of weight retrieval