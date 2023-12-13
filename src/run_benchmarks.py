import benchmark
import repl
import matplotlib.pyplot as plt
import numpy as np

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

    # Run benchmarks for FHE runtime of matrix completion algorithms (test size 5, 10, 20, and 40 if time)
    no_fhe_time5, fhe_time5 = benchmark.secure_insecure_matrix_completion_runtime_benchmark(rows=5, cols=5)
    no_fhe_time10, fhe_time10 = benchmark.secure_insecure_matrix_completion_runtime_benchmark(rows=10, cols=10)
    no_fhe_time20, fhe_time20 = benchmark.secure_insecure_matrix_completion_runtime_benchmark(rows=20, cols=20)

    no_fhe_robust_time5, fhe_robust_time5 = benchmark.secure_insecure_robust_matrix_completion_runtime_benchmark(rows=5, cols=5)
    no_fhe_robust_time10, fhe_robust_time10 = benchmark.secure_insecure_robust_matrix_completion_runtime_benchmark(rows=10, cols=10)
    no_fhe_robust_time20, fhe_robust_time20 = benchmark.secure_insecure_robust_matrix_completion_runtime_benchmark(rows=20, cols=20)

    runtime_variations = ("5x5", "10x10", "20x20", "Robust 5x5", "Robust 10x10", "Robust 20x20")
    runtime_data = {
        'Without FHE': (
            no_fhe_time5, no_fhe_time10, no_fhe_time20, 
            no_fhe_robust_time5, no_fhe_robust_time10, no_fhe_robust_time20
        ),
        'With FHE': (
            fhe_time5, fhe_time10, fhe_time20, 
            fhe_robust_time5, fhe_robust_time10, fhe_robust_time20
        ),
    }

    x = np.arange(len(runtime_variations))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in runtime_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Runtime (sec)')
    ax.set_xticks(x + width, runtime_variations)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 250)

    plt.savefig("runtime_bar_chart.png")

    # Run benchmarks for FHE + PIR runtime of weight retrieval