### File for containing tests to benchmark performance of robust and secure matrix completion algorithms

import matplotlib
import numpy as np
import time
import tenseal as ts

import insecure_algos
import secure_algos
import robust_algos
from support import util
from repl import setup_contexts
from scipy.sparse.linalg import svds

from typing import List, Tuple

# =========================================================================================================================
# ============================================ BEGIN HELPER FUNCTIONS =====================================================
# =========================================================================================================================

# Generates an arbitrary matrix of movie ratings for benchmarking accuracy
# Min and max control range of entries, step controls how far away these entries are from each other (step size of 0.5 means you can have min, min + 0.5, min + 1, etc...)
def generate_ratings_matrix(min : int = 0, max : int = 5, rows : int = 100, cols : int = 100, step = 0.5):
    matrix = np.empty((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            random_num : float = np.random.rand()
            
            if step is None or step == 0:
                matrix[i][j] = (random_num * (max - min)) + min

            # Sort into (max - min) / step + 1 discrete buckets
            else:
                buckets = ((max - min) / step) + 1
                random_val = ((random_num * buckets) // 1) * step
                matrix[i][j] = random_val
    
    return matrix

# Helper func to generate boolean matrices
# Generates two matrices:
# The true matrix (where data is revealed with probability p), and noised matrix, where data has an additional epsilon chance of being revealed
def generate_boolean_matrices(epsilon : float = 0.1, p : float = 0.2, rows : int = 100, cols : int = 100) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    # Create ground truth matrix
    true_boolean_matrix = np.zeros((rows, cols))
    noised_boolean_matrix = np.zeros((rows, cols))

    # Entries are added to the true boolean matrix (and noised) with probability p, and to noised only with probability epsilon
    for i in range(rows):
        for j in range(cols):
            random_num : float = np.random.rand()

            if random_num < p:
                true_boolean_matrix[i][j] = 1
                noised_boolean_matrix[i][j] = 1
            elif random_num < (p + epsilon):
                true_boolean_matrix[i][j] = 0
                noised_boolean_matrix[i][j] = 1
            else:
                true_boolean_matrix[i][j] = 0
                noised_boolean_matrix[i][j] = 0
    
    return true_boolean_matrix, noised_boolean_matrix

# Helper function to generate the insecure weight generator
# Inputted parameters are inherited (set) by those passed in to benchmarking functions (see below)
def generate_insecure_weights(epochs : int = 10, sub_epochs : int = 5, epsilon : float = 1e-3, alpha : float = 1, debug : bool = True) -> insecure_algos.InsecureRobustWeights:
    svd_1d_impl = insecure_algos.InsecureSVD1D(debug=debug)
    weight_calculator = insecure_algos.InsecureRobustWeights(svd_1d_impl, epochs=epochs, sub_epochs=sub_epochs, epsilon=epsilon, alpha=alpha)
    return weight_calculator

# Helper function to generate the secure weight generator
# Inputted parameters are inherited (set) by those passed in to benchmarking functions (see below)
def generate_secure_weights(public_context : bytes, secret_context : bytes, epochs : int = 10, sub_epochs : int = 5, epsilon : float = 1e-3, alpha : float = 1, debug : bool = True) -> robust_algos.SecureRobustWeights:
    svd_1d_impl = secure_algos.SecureSVD1D(public_context=public_context, secret_context=secret_context, debug=debug)
    weight_calculator = robust_algos.SecureRobustWeights(public_context=public_context, secret_context=secret_context, svd_1d_wrapper=svd_1d_impl, epochs=epochs, sub_epochs=sub_epochs, epsilon=epsilon, alpha=alpha, debug=debug)
    return weight_calculator

# Helper function to generate insecure matrix completion
# Inputted parameters are inherited (set) by those passed in to benchmarking functions (see below)
def generate_insecure_matrix_completion(r = 5, epochs = 20, alpha = 1e-2, loss_list = None) -> insecure_algos.InsecureMatrixCompletion:
    # Create matrix completion instances
    svd_1d_impl = insecure_algos.InsecureSVD1D(debug=False)
    svd_impl = insecure_algos.InsecureSVD(debug = False, svd_1d_wrapper=svd_1d_impl)

    insecure_completer = insecure_algos.InsecureMatrixCompletion(r=r, epochs=epochs, alpha=alpha, insecure_svd_wrapper=svd_impl, loss_list=loss_list)
    return insecure_completer

# Helper function to generate insecure robust matrix completion
# Inputted parameters are inherited (set) by those passed in to benchmarking functions (see below)
def generate_insecure_robust_matrix_completion(r = 5, epochs = 20, alpha = 1e-2, w_epochs = 10, w_sub_epochs = 5, loss_list = None) -> insecure_algos.RobustInsecureMatrixCompletion:
    svd_1d_impl = insecure_algos.InsecureSVD1D(debug=False)
    weight_calculator = insecure_algos.InsecureRobustWeights(svd_1d_impl, epochs=w_epochs, sub_epochs=w_sub_epochs)
    svd_impl = insecure_algos.InsecureSVD(debug = False, svd_1d_wrapper=svd_1d_impl)
    robust_completer = insecure_algos.RobustInsecureMatrixCompletion(r=r, epochs=epochs, alpha=alpha, insecure_svd_wrapper=svd_impl, insecure_robust_weights_wrapper=weight_calculator, loss_list=loss_list)
    return robust_completer

# Helper function to generate secure matrix completion
# Inputted parameters are inherited (set) by those passed in to benchmarking functions (see below)
def generate_secure_matrix_completion(public_context : bytes, secret_context : bytes, r : int = 1, epochs : int = 20, alpha : float = 1e-2, loss_list = None) -> secure_algos.SecureMatrixCompletion:
    reset_impl = secure_algos.SecureMatrixErrorReset(public_context, secret_context)
    svd_1d_impl = secure_algos.SecureSVD1D(public_context, secret_context, False)
    svd_impl = secure_algos.SecureSVD(public_context, secret_context, debug=False)
    clip_impl = secure_algos.SecureClip(public_context, secret_context)
    division_impl = secure_algos.SecureClearDivision(secret_context)

    secure_completer = secure_algos.SecureMatrixCompletion(r=r, epochs=epochs, alpha=alpha, public_context=public_context, secure_matrix_error_reset_wrapper=reset_impl, secure_svd_wrapper=svd_impl, secure_clip_wrapper=clip_impl, secure_division_wrapper=division_impl, loss_list=loss_list)
    return secure_completer

# Helper function to generate robust secure matrix completion
# Inputted parameters are inherited (set) by those passed in to benchmarking functions (see below)
def generate_secure_robust_matrix_completion(public_context : bytes, secret_context : bytes, r : int = 1, epochs : int = 20, alpha : float = 1e-2, w_epochs = 10, w_sub_epochs = 5, loss_list = None) -> robust_algos.RobustSecureMatrixCompletion:
    reset_impl = secure_algos.SecureMatrixErrorReset(public_context, secret_context)
    svd_1d_impl = secure_algos.SecureSVD1D(public_context, secret_context, False)
    svd_impl = secure_algos.SecureSVD(public_context, secret_context, svd_1d_impl, False)
    clip_impl = secure_algos.SecureClip(public_context, secret_context)
    division_impl = secure_algos.SecureClearDivision(secret_context)
    weight_calculator = robust_algos.SecureRobustWeights(public_context=public_context, secret_context=secret_context, svd_1d_wrapper=svd_1d_impl, epochs=w_epochs, sub_epochs=w_sub_epochs)
    
    robust_completer = robust_algos.RobustSecureMatrixCompletion(r=r, epochs=epochs, alpha=alpha, public_context=public_context, secure_matrix_error_reset_wrapper=reset_impl, secure_svd_wrapper=svd_impl, secure_clip_wrapper=clip_impl, secure_division_wrapper=division_impl, secure_robust_weights_wrapper=weight_calculator, loss_list=loss_list)
    return robust_completer

# =========================================================================================================================
# ========================================= BEGIN BENCHMARKING FUNCTIONS ==================================================
# =========================================================================================================================

# =========================================================================================================================
# ======================================= WEIGHT COMPUTATION BENCHMARKING =================================================
# =========================================================================================================================

# Benchmark for accuracy of matrix weights (norm between weighted entries and true entries)
# Returns Tuple of Frobenius norms:
# 1. Norm of true revealed entries with all 1's matrix
# 2. Norms of weights with all 1's matrix (computed using robust pre-processing step)
def weight_accuracy_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 100, cols : int = 100, epochs = 10, sub_epochs = 5) -> Tuple[float, float]:
    # Create ground truth matrix
    true_boolean_matrix, noised_boolean_matrix = generate_boolean_matrices(epsilon, p, rows, cols)
    
    # We have constructed the matrix, run tests
    weight_calculator = generate_insecure_weights(epochs = epochs, sub_epochs=sub_epochs, epsilon=epsilon, debug=False)
    weights = weight_calculator.compute_weights(noised_boolean_matrix)

    # Determine difference in spectral norm between
    ones_mat = np.ones(rows, cols)
    true_norm = np.norm(true_boolean_matrix - ones_mat)
    weighted_norm = np.norm(weights - ones_mat)

    return true_norm, weighted_norm

# Variant of above function but with encrypted booleans
# Returns Tuple of Frobenius norms:
# 1. Norm of true revealed entries with all 1's matrix
# 2. Norms of weights with all 1's matrix (computed using robust pre-processing step)
def secure_weight_accuracy_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 10, cols : int = 10, epochs = 10, sub_epochs = 5) -> Tuple[float, float]:
# Create encrypted ground truth matrix
    public_context, secret_context = setup_contexts()
    true_boolean_matrix, noised_boolean_matrix = generate_boolean_matrices(epsilon, p, rows, cols)
    noised_boolean_matrix = util.encrypt_to_ckks_mat(noised_boolean_matrix, public_context)
    noised_boolean_matrix = util.convert_ckks_mat_to_bytes_mat(noised_boolean_matrix)
    
    # We have constructed the matrix, run tests
    weight_calculator = generate_secure_weights(public_context, secret_context, epochs, sub_epochs, epsilon, debug=False)
    weights = weight_calculator.compute_weights(noised_boolean_matrix)
    weights = util.convert_bytes_mat_to_ckks_mat(weights, public_context)
    weights = util.decrypt_ckks_mat(weights, secret_context)

    # Determine difference in spectral norm between
    ones_mat = np.ones(rows, cols)
    true_norm = np.norm(true_boolean_matrix - ones_mat)
    weighted_norm = np.norm(weights - ones_mat)

    return true_norm, weighted_norm

# Benchmark for runtime of matrix weights (norm between weighted entries and true entries)
# Returns a float:
# 1. Time indicating how long it took the secure weights implementation to perform the weights preprocessing step
def secure_weight_runtime_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 10, cols : int = 10, epochs = 10, sub_epochs = 5) -> float:
    # Create encrypted ground truth matrix
    public_context, secret_context = setup_contexts()
    _, noised_boolean_matrix = generate_boolean_matrices(epsilon, p, rows, cols)
    noised_boolean_matrix = util.encrypt_to_ckks_mat(noised_boolean_matrix, public_context)
    noised_boolean_matrix = util.convert_ckks_mat_to_bytes_mat(noised_boolean_matrix)

    # We have constructed the matrix, run tests
    weight_calculator = generate_secure_weights(public_context, secret_context, epochs, sub_epochs, epsilon, debug=False)

    runtime_start = time.time()
    _ = weight_calculator.compute_weights(noised_boolean_matrix)
    runtime_end = time.time()

    return runtime_end - runtime_start

# Runtime analysis for secure weight computation (pre-processing robust step)
# Returns a float:
# 1. Time indicating how long it took the plaintext weights implementation to perform the weights preprocessing step
def weight_runtime_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 100, cols : int = 100, epochs = 10, sub_epochs = 5) -> float:
    # Create ground truth matrix
    _, noised_boolean_matrix = generate_boolean_matrices(epsilon, p, rows, cols)
    
    # We have constructed the matrix, run tests
    weight_calculator = generate_insecure_weights(epochs = epochs, sub_epochs=sub_epochs, epsilon=epsilon, debug=False)
    runtime_start = time.time()
    _ = weight_calculator.compute_weights(noised_boolean_matrix)
    runtime_end = time.time()

    return runtime_end - runtime_start

# =========================================================================================================================
# ==================================== MATRIX COMPLETION RUNTIME BENCHMARKING =============================================
# =========================================================================================================================

# Benchmark runtime of robust precomputation relative to input size and epsilon
# Evaluates runtime of robust and non-robust matrix completion on noised and non-noised data, respectively
# Returns tuple of floats:
# 1. Time taken to perform matrix completion on non-robust implementation using non-noised data
# 2. Time taken to perform matrix completion on robust implementation using noised data
def noised_matrix_completion_runtime_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 100, cols : int = 100, min : int = 0, max : int = 5, step : float = 0.5, w_epochs = 10, w_sub_epochs = 5, r = 5, epochs = 20, alpha = 1e-2) -> Tuple[float, float]:
    true_ratings_matrix = generate_ratings_matrix(rows = rows, cols = cols, min=min, max=max, step=step)
    true_boolean_matrix, noised_boolean_matrix = generate_boolean_matrices(epsilon, p, rows = rows, cols = cols)
    true_revealed_ratings = true_ratings_matrix * true_boolean_matrix
    noised_revealed_ratings = true_ratings_matrix * noised_boolean_matrix

    # Create matrix completion instances
    insecure_completer = generate_insecure_matrix_completion(r=r, epochs=epochs, alpha=alpha)
    robust_completer = generate_insecure_robust_matrix_completion(w_epochs=w_epochs, w_sub_epochs=w_sub_epochs, r=r, epochs=epochs, alpha=alpha)

    # Run training algorithm on both implementations
    true_start = time.time()
    insecure_completer.prepare_data(true_revealed_ratings, true_boolean_matrix)
    insecure_complete_matrix = insecure_completer.train()
    true_end = time.time()

    robust_start = time.time()
    robust_completer.prepare_data(noised_revealed_ratings, noised_boolean_matrix)
    robust_complete_matrix = robust_completer.train()
    robust_end = time.time()

    return true_end - true_start, robust_end - robust_start

# Benchmark runtime of robust precomputation relative to input size and epsilon
# Evaluates runtime of robust and non-robust matrix completion, both on non-noised data
# Returns tuple of floats:
# 1. Time taken to perform matrix completion on non-robust implementation using non-noised data
# 2. Time taken to perform matrix completion on robust implementation using non-noised data
def matrix_completion_runtime_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 100, cols : int = 100, min : int = 0, max : int = 5, step : float = 0.5, w_epochs = 10, w_sub_epochs = 5, r = 5, epochs = 20, alpha = 1e-2) -> Tuple[float, float]:
    true_ratings_matrix = generate_ratings_matrix(rows = rows, cols = cols, min=min, max=max, step=step)
    true_boolean_matrix, _ = generate_boolean_matrices(epsilon, p, rows = rows, cols = cols)
    true_revealed_ratings = true_ratings_matrix * true_boolean_matrix

    # Create matrix completion instances
    insecure_completer = generate_insecure_matrix_completion(r=r, epochs=epochs, alpha=alpha)
    robust_completer = generate_insecure_robust_matrix_completion(w_epochs=w_epochs, w_sub_epochs=w_sub_epochs, r=r, epochs=epochs, alpha=alpha)

    # Run training algorithm on both implementations
    true_start = time.time()
    insecure_completer.prepare_data(true_revealed_ratings, true_boolean_matrix)
    insecure_complete_matrix = insecure_completer.train()
    true_end = time.time()

    robust_start = time.time()
    robust_completer.prepare_data(true_revealed_ratings, true_boolean_matrix)
    robust_complete_matrix = robust_completer.train()
    robust_end = time.time()

    return true_end - true_start, robust_end - robust_start

# Secure version of comparing runtime between robust and non-robust implementation on same datasets
# Returns tuple of floats:
# 1. Time taken to perform matrix completion on non-robust implementation using non-noised data
# 2. Time taken to perform matrix completion on robust implementation using non-noised data
def secure_matrix_completion_runtime_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 10, cols : int = 10, min : int = 0, max : int = 5, step : float = 0.5, w_epochs = 10, w_sub_epochs = 5, r = 5, epochs = 20, alpha = 1e-2) -> Tuple[float, float]:
    public_context, secret_context = setup_contexts()

    ratings_matrix = generate_ratings_matrix(rows = rows, cols = cols, min=min, max=max, step=step)
    true_boolean_matrix, noised_boolean_matrix = generate_boolean_matrices(epsilon, p, rows = rows, cols = cols)
    true_revealed_ratings = ratings_matrix * true_boolean_matrix

    true_ratings_matrix = util.encrypt_to_ckks_mat(ratings_matrix, public_context)
    true_ratings_matrix = util.convert_ckks_mat_to_bytes_mat(true_ratings_matrix)

    true_boolean_matrix = util.encrypt_to_ckks_mat(true_boolean_matrix, public_context)
    true_boolean_matrix = util.convert_ckks_mat_to_bytes_mat(true_boolean_matrix)

    secure_completer = generate_secure_matrix_completion(public_context, secret_context, r=r, epochs=epochs, alpha=alpha)
    robust_completer = generate_secure_robust_matrix_completion(public_context, secret_context, w_epochs=w_epochs, w_sub_epochs=w_sub_epochs, r=r, epochs=epochs, alpha=alpha)

    # Run training algorithm on both implementations
    true_start = time.time()
    secure_completer.prepare_data(true_revealed_ratings, true_boolean_matrix)
    insecure_complete_matrix = secure_completer.train()
    true_end = time.time()

    robust_start = time.time()
    robust_completer.prepare_data(true_revealed_ratings, true_boolean_matrix)
    robust_complete_matrix = robust_completer.train()
    robust_end = time.time()

    return true_end - true_start, robust_end - robust_start

# Secure version of comparing runtime between robust and non-robust implementation on different datasets
# Evaluates runtime of robust and non-robust matrix completion, both on non-noised data
# Returns tuple of floats:
# 1. Time taken to perform matrix completion on non-robust implementation using non-noised data
# 2. Time taken to perform matrix completion on robust implementation using noised data
def noised_secure_matrix_completion_runtime_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 10, cols : int = 10, min : int = 0, max : int = 5, step : float = 0.5, w_epochs = 10, w_sub_epochs = 5, r = 5, epochs = 20, alpha = 1e-2) -> Tuple[float, float]:
    public_context, secret_context = setup_contexts()

    ratings_matrix = generate_ratings_matrix(rows = rows, cols = cols, min=min, max=max, step=step)
    true_boolean_matrix, noised_boolean_matrix = generate_boolean_matrices(epsilon, p, rows = rows, cols = cols)
    true_revealed_ratings = ratings_matrix * true_boolean_matrix
    noised_revealed_ratings = ratings_matrix * noised_boolean_matrix

    true_ratings_matrix = util.encrypt_to_ckks_mat(ratings_matrix, public_context)
    true_ratings_matrix = util.convert_ckks_mat_to_bytes_mat(true_ratings_matrix)

    true_boolean_matrix = util.encrypt_to_ckks_mat(true_boolean_matrix, public_context)
    true_boolean_matrix = util.convert_ckks_mat_to_bytes_mat(true_boolean_matrix)

    noised_ratings_matrix = util.encrypt_to_ckks_mat(noised_ratings_matrix, public_context)
    noised_ratings_matrix = util.convert_ckks_mat_to_bytes_mat(noised_ratings_matrix)

    noised_boolean_matrix = util.encrypt_to_ckks_mat(noised_boolean_matrix, public_context)
    noised_boolean_matrix = util.convert_ckks_mat_to_bytes_mat(noised_boolean_matrix)

    secure_completer = generate_secure_matrix_completion(public_context, secret_context, r=r, epochs=epochs, alpha=alpha)
    robust_completer = generate_secure_robust_matrix_completion(public_context, secret_context, w_epochs=w_epochs, w_sub_epochs=w_sub_epochs, r=r, epochs=epochs, alpha=alpha)

    # Run training algorithm on both implementations
    true_start = time.time()
    secure_completer.prepare_data(true_revealed_ratings, true_boolean_matrix)
    insecure_complete_matrix = secure_completer.train()
    true_end = time.time()

    robust_start = time.time()
    robust_completer.prepare_data(noised_revealed_ratings, noised_boolean_matrix)
    robust_complete_matrix = robust_completer.train()
    robust_end = time.time()

    return true_end - true_start, robust_end - robust_start

# Measuring runtime for matrix completion algorithm on encrypted/non-encrypted data
# Returns tuple of floats:
# 1. Runtime taken for matrix completion algorithm on non-encrypted data
# 2. Runtime taken for matrix completion algorithm on encrypted data
def secure_insecure_matrix_completion_runtime_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 10, cols : int = 10, min : int = 0, max : int = 5, step : float = 0.5, w_epochs = 10, w_sub_epochs = 5, r = 5, epochs = 20, alpha = 1e-2) -> Tuple[float, float]:
    public_context, secret_context = setup_contexts()
    public_key = ts.context_from(public_context)
    secret_key = ts.context_from(secret_context)
    
    true_ratings_matrix = generate_ratings_matrix(rows = rows, cols = cols, min=min, max=max, step=step)
    true_boolean_matrix, _ = generate_boolean_matrices(epsilon, p, rows = rows, cols = cols)
    true_revealed_ratings = true_ratings_matrix * true_boolean_matrix

    encrypted_true_ratings_matrix = util.encrypt_to_ckks_mat(true_ratings_matrix, public_key)
    encrypted_true_ratings_matrix = util.convert_ckks_mat_to_bytes_mat(encrypted_true_ratings_matrix)

    encrypted_true_boolean_matrix = util.encrypt_to_ckks_mat(true_boolean_matrix, public_key)
    encrypted_true_boolean_matrix = util.convert_ckks_mat_to_bytes_mat(encrypted_true_boolean_matrix)

    insecure_completer = generate_insecure_matrix_completion(r=r, epochs=epochs, alpha=alpha)
    secure_completer = generate_secure_matrix_completion(public_context, secret_context, r=r, epochs=epochs, alpha=alpha)

    # Run training algorithm on both implementations

    insecure_start = time.time()
    insecure_completer.prepare_data(true_revealed_ratings, true_boolean_matrix)
    insecure_complete_matrix = insecure_completer.train()
    insecure_end = time.time()

    secure_start = time.time()
    secure_completer.prepare_data(encrypted_true_ratings_matrix, encrypted_true_boolean_matrix)
    scure_complete_matrix = secure_completer.train()
    secure_end = time.time()

    return insecure_end - insecure_start, secure_end - secure_start

# =========================================================================================================================
# ==================================== MATRIX COMPLETION ACCURACY BENCHMARKING ============================================
# =========================================================================================================================

# Implement versions of the above algorithms but on homomorphically encrypted (Secure) data
# Secure version of checking accuracy of robust implementation on noised data
# Returns tuple of Frobenius norms:
# 1. Norm of completed matrix (compared with ground truth matrix) using non-noised data and non-robust completion
# 2. Norm of completed matrix (compared with ground truth matrix) using noised data and robust-completion
def secure_robust_matrix_completion_accuracy_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 10, cols : int = 10, min : int = 0, max : int = 5, step : float = 0.5, w_epochs = 10, w_sub_epochs = 5, r = 5, epochs = 20, alpha = 1e-2) -> Tuple[float, float]:
    public_context, secret_context = setup_contexts()

    ratings_matrix = generate_ratings_matrix(rows = rows, cols = cols, min=min, max=max, step=step)
    true_boolean_matrix, noised_boolean_matrix = generate_boolean_matrices(epsilon, p, rows = rows, cols = cols)
    true_revealed_ratings = ratings_matrix * true_boolean_matrix
    noised_revealed_ratings = ratings_matrix * noised_boolean_matrix

    true_ratings_matrix = util.encrypt_to_ckks_mat(ratings_matrix, public_context)
    true_ratings_matrix = util.convert_ckks_mat_to_bytes_mat(true_ratings_matrix)

    true_boolean_matrix = util.encrypt_to_ckks_mat(true_boolean_matrix, public_context)
    true_boolean_matrix = util.convert_ckks_mat_to_bytes_mat(true_boolean_matrix)

    noised_ratings_matrix = util.encrypt_to_ckks_mat(noised_ratings_matrix, public_context)
    noised_ratings_matrix = util.convert_ckks_mat_to_bytes_mat(noised_ratings_matrix)

    noised_boolean_matrix = util.encrypt_to_ckks_mat(noised_boolean_matrix, public_context)
    noised_boolean_matrix = util.convert_ckks_mat_to_bytes_mat(noised_boolean_matrix)

    secure_completer = generate_secure_matrix_completion(public_context, secret_context, r=r, epochs=epochs, alpha=alpha)
    robust_completer = generate_secure_robust_matrix_completion(public_context, secret_context, w_epochs=w_epochs, w_sub_epochs=w_sub_epochs, r=r, epochs=epochs, alpha=alpha)

    # Run training algorithm on both implementations
    secure_completer.prepare_data(true_revealed_ratings, true_boolean_matrix)
    secure_complete_matrix = secure_completer.train()
    secure_complete_matrix = util.convert_bytes_mat_to_ckks_mat(secure_complete_matrix, public_context)
    secure_complete_matrix = util.decrypt_ckks_mat(secure_complete_matrix, secret_context)

    robust_completer.prepare_data(noised_revealed_ratings, noised_boolean_matrix)
    robust_complete_matrix = robust_completer.train()
    robust_complete_matrix = util.convert_bytes_mat_to_ckks_mat(robust_complete_matrix, public_context)
    robust_complete_matrix = util.decrypt_ckks_mat(robust_complete_matrix, secret_context)

    true_norm = np.norm(secure_complete_matrix - ratings_matrix)
    weighted_norm = np.norm(robust_complete_matrix - ratings_matrix)
    return true_norm, weighted_norm

# Secure version of checking comparison between robust and non-robust implementation on noised data
# Returns tuple of Frobenius norms:
# 1. Norm of completed matrix (compared with ground truth matrix) using noised data and non-robust completion
# 2. Norm of completed matrix (compared with ground truth matrix) using noised data and robust-completion
def noised_secure_noised_matrix_completion_accuracy_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 10, cols : int = 10, min : int = 0, max : int = 5, step : float = 0.5, w_epochs = 10, w_sub_epochs = 5, r = 5, epochs = 20, alpha = 1e-2) -> Tuple[float, float]:
    public_context, secret_context = setup_contexts()
    
    true_ratings_matrix = generate_ratings_matrix(rows = rows, cols = cols, min=min, max=max, step=step)
    _, noised_boolean_matrix = generate_boolean_matrices(epsilon, p, rows = rows, cols = cols)
    noised_revealed_ratings = true_ratings_matrix * noised_boolean_matrix

    noised_ratings_matrix = util.encrypt_to_ckks_mat(noised_ratings_matrix, public_context)
    noised_ratings_matrix = util.convert_ckks_mat_to_bytes_mat(noised_ratings_matrix)

    noised_boolean_matrix = util.encrypt_to_ckks_mat(noised_boolean_matrix, public_context)
    noised_boolean_matrix = util.convert_ckks_mat_to_bytes_mat(noised_boolean_matrix)

    # Create matrix completion instances
    secure_completer = generate_secure_matrix_completion(public_context, secret_context, r=r, epochs=epochs, alpha=alpha)
    robust_completer = generate_secure_robust_matrix_completion(public_context, secret_context, w_epochs=w_epochs, w_sub_epochs=w_sub_epochs, r=r, epochs=epochs, alpha=alpha)

    # Run training algorithm on both implementations
    secure_completer.prepare_data(noised_revealed_ratings, noised_boolean_matrix)
    secure_complete_matrix = secure_completer.train()
    secure_complete_matrix = util.convert_bytes_mat_to_ckks_mat(secure_complete_matrix, public_context)
    secure_complete_matrix = util.decrypt_ckks_mat(secure_complete_matrix, secret_context)

    robust_completer.prepare_data(noised_revealed_ratings, noised_boolean_matrix)
    robust_complete_matrix = robust_completer.train()
    robust_complete_matrix = util.convert_bytes_mat_to_ckks_mat(robust_complete_matrix, public_context)
    robust_complete_matrix = util.decrypt_ckks_mat(robust_complete_matrix, secret_context)

    regular_norm = np.norm(secure_complete_matrix - true_ratings_matrix)
    weighted_norm = np.norm(robust_complete_matrix - true_ratings_matrix)
    return regular_norm, weighted_norm

# Benchmark for performance of robust matrix completion (as opposed to standard matrix completion) using norm compared to true data
# Checks the completed matrix from noised data with completed matrix from unoised data
# Returns tuple of Frobenius norms:
# 1. Norm of completed matrix (compared with ground truth matrix) using non-noised data and non-robust completion
# 2. Norm of completed matrix (compared with ground truth matrix) using noised data and robust-completion
def robust_matrix_completion_accuracy_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 100, cols : int = 100, min : int = 0, max : int = 5, step : float = 0.5, w_epochs = 10, w_sub_epochs = 5, r = 5, epochs = 20, alpha = 1e-2) -> Tuple[float, float]:
    true_ratings_matrix = generate_ratings_matrix(rows = rows, cols = cols, min=min, max=max, step=step)
    true_boolean_matrix, noised_boolean_matrix = generate_boolean_matrices(epsilon, p, rows = rows, cols = cols)
    true_revealed_ratings = true_ratings_matrix * true_boolean_matrix
    noised_revealed_ratings = true_ratings_matrix * noised_boolean_matrix

    insecure_completer = generate_insecure_matrix_completion(r=r, epochs=epochs, alpha=alpha)
    robust_completer = generate_insecure_robust_matrix_completion(w_epochs=w_epochs, w_sub_epochs=w_sub_epochs, r=r, epochs=epochs, alpha=alpha)

    # Run training algorithm on both implementations
    insecure_completer.prepare_data(true_revealed_ratings, true_boolean_matrix)
    insecure_complete_matrix = insecure_completer.train()

    robust_completer.prepare_data(noised_revealed_ratings, noised_boolean_matrix)
    robust_complete_matrix = robust_completer.train()

    true_norm = np.norm(insecure_complete_matrix - true_ratings_matrix)
    weighted_norm = np.norm(robust_complete_matrix - true_ratings_matrix)
    return true_norm, weighted_norm

# Benchmark for testing performance of robust matrix completion with epsilon error compared to normal implementation
# Checks the robustly completed matrix from noised data with regularly completed matrix from same noised data
# Returns tuple of Frobenius norms:
# 1. Norm of completed matrix (compared with ground truth matrix) using noised data and non-robust completion
# 2. Norm of completed matrix (compared with ground truth matrix) using noised data and robust-completion
def noised_matrix_completion_accuracy_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 100, cols : int = 100, min : int = 0, max : int = 5, step : float = 0.5, w_epochs = 10, w_sub_epochs = 5, r = 5, epochs = 20, alpha = 1e-2) -> Tuple[float, float]:
    true_ratings_matrix = generate_ratings_matrix(rows = rows, cols = cols, min=min, max=max, step=step)
    _, noised_boolean_matrix = generate_boolean_matrices(epsilon, p, rows = rows, cols = cols)
    noised_revealed_ratings = true_ratings_matrix * noised_boolean_matrix

    # Create matrix completion instances
    insecure_completer = generate_insecure_matrix_completion(r=r, epochs=epochs, alpha=alpha)
    robust_completer = generate_insecure_robust_matrix_completion(w_epochs=w_epochs, w_sub_epochs=w_sub_epochs, r=r, epochs=epochs, alpha=alpha)

    # Run training algorithm on both implementations
    insecure_completer.prepare_data(noised_revealed_ratings, noised_boolean_matrix)
    insecure_complete_matrix = insecure_completer.train()

    robust_completer.prepare_data(noised_revealed_ratings, noised_boolean_matrix)
    robust_complete_matrix = robust_completer.train()

    regular_norm = np.norm(insecure_complete_matrix - true_ratings_matrix)
    weighted_norm = np.norm(robust_complete_matrix - true_ratings_matrix)
    return regular_norm, weighted_norm

# =========================================================================================================================
# ========================================= SVD IMPLEMENTATION BENCHMARKING ===============================================
# =========================================================================================================================

# NOTE: Currently missing accuracy tests because the singular value portion of the SVD benchmarking got removed

# SVD Runtime analysis of our implementation versus scipy on plaintext data
# Returns tuple of floats
# 1. Time taken to perform SVD using our implementation
# 2. Time taken to perform SVD using scipy implementation
def insecure_svd_runtime_benchmark(r : int = 6, epsilon : float = 0.1, p : float = 0.2, rows : int = 10, cols : int = 10, min : int = 0, max : int = 5, step : float = 0.5) -> Tuple[float, float]:
    ratings_matrix = generate_ratings_matrix(rows = rows, cols = cols, min=min, max=max, step=step)
    true_boolean_matrix, _ = generate_boolean_matrices(epsilon, p, rows = rows, cols = cols)
    true_revealed_ratings = ratings_matrix * true_boolean_matrix
    
    svd_1d_impl = insecure_algos.InsecureSVD1D(debug=False)
    svd_impl = insecure_algos.InsecureSVD(debug = False, svd_1d_wrapper=svd_1d_impl)

    our_impl_start = time.time()
    svd_impl.compute_SVD(true_revealed_ratings, r)
    our_impl_end = time.time()

    scipy_impl_start = time.time()
    svds(true_boolean_matrix, k=r)
    scipy_impl_end = time.time()

    return our_impl_end - our_impl_start, scipy_impl_end - scipy_impl_start

# SVD Runtime analysis of our implementation versus our implementation using FHE
# Returns tuple of floats
# 1. Time taken to perform SVD using our implementation (no FHE)
# 2. Time taken to perform SVD using our implementation (FHE)
def secure_svd_runtime_benchmark(r : int = 6, epsilon : float = 0.1, p : float = 0.2, rows : int = 10, cols : int = 10, min : int = 0, max : int = 5, step : float = 0.5) -> Tuple[float, float]:
    public_context, secret_context = setup_contexts()
    
    ratings_matrix = generate_ratings_matrix(rows = rows, cols = cols, min=min, max=max, step=step)
    true_boolean_matrix, _ = generate_boolean_matrices(epsilon, p, rows = rows, cols = cols)
    true_revealed_ratings = ratings_matrix * true_boolean_matrix

    encrypted_true_ratings_matrix = util.encrypt_to_ckks_mat(true_revealed_ratings, public_context)
    encrypted_true_ratings_matrix = util.convert_ckks_mat_to_bytes_mat(encrypted_true_ratings_matrix)

    svd_1d_impl = insecure_algos.InsecureSVD1D(debug=False)
    insecure_svd_impl = insecure_algos.InsecureSVD(debug = False, svd_1d_wrapper=svd_1d_impl)

    secure_svd_impl = secure_algos.SecureSVD(public_context, secret_context, secure_algos.SecureSVD1D, False)

    insecure_impl_start = time.time()
    insecure_svd_impl.compute_SVD(true_revealed_ratings, r)
    insecure_impl_end = time.time()

    secure_impl_start = time.time()
    secure_svd_impl.compute_SVD(encrypted_true_ratings_matrix, r)
    secure_impl_end = time.time()

    return insecure_impl_end - insecure_impl_start, secure_impl_end - secure_impl_start

# =========================================================================================================================
# ============================================== LOSS CONVERGENCE BENCHMARKING ============================================
# =========================================================================================================================

# Measuring loss convergence in matrix completion between standard matrix completion instances using FHE and non-FHE on same data
# Returns tuple of lists:
# 1. Losses from training non-robust implementation using unecnrypted non-noised data
# 2. Losses from training non-robust implementation using encrypted non-noised data
def secure_insecure_matrix_completion_loss_convergence_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 10, cols : int = 10, min : int = 0, max : int = 5, step : float = 0.5, w_epochs = 10, w_sub_epochs = 5, r = 5, epochs = 20, alpha = 1e-2) -> Tuple[List[float], List[float]]:
    public_context, secret_context = setup_contexts()
    
    true_ratings_matrix = generate_ratings_matrix(rows = rows, cols = cols, min=min, max=max, step=step)
    true_boolean_matrix, _ = generate_boolean_matrices(epsilon, p, rows = rows, cols = cols)
    true_revealed_ratings = true_ratings_matrix * true_boolean_matrix

    encrypted_true_ratings_matrix = util.encrypt_to_ckks_mat(true_ratings_matrix, public_context)
    encrypted_true_ratings_matrix = util.convert_ckks_mat_to_bytes_mat(encrypted_true_ratings_matrix)

    encrypted_true_boolean_matrix = util.encrypt_to_ckks_mat(true_boolean_matrix, public_context)
    encrypted_true_boolean_matrix = util.convert_ckks_mat_to_bytes_mat(encrypted_true_boolean_matrix)

    # Create matrix completion instances
    unencrypted_losses = []
    encrypted_losses = []

    insecure_completer = generate_insecure_matrix_completion(r=r, epochs=epochs, alpha=alpha, loss_list=unencrypted_losses)
    secure_completer = generate_secure_matrix_completion(public_context, secret_context, r=r, epochs=epochs, alpha=alpha, loss_list=encrypted_losses)

    # Run training algorithm on both implementations
    insecure_completer.prepare_data(true_revealed_ratings, true_boolean_matrix)
    insecure_complete_matrix = insecure_completer.train()

    secure_completer.prepare_data(encrypted_true_ratings_matrix, encrypted_true_boolean_matrix)
    scure_complete_matrix = secure_completer.train()

    return unencrypted_losses, encrypted_losses

# Measuring loss convergence in matrix completion between standard matrix completion and robust matrix completion
# Loss convergence for insecure completion on same, noised data
# Returns tuple of lists:
# 1. Losses from training non-robust implementation using noised data
# 2. Losses from training robust implementation using noised data
def noised_matrix_completion_loss_convergence_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 100, cols : int = 100, min : int = 0, max : int = 5, step : float = 0.5, w_epochs = 10, w_sub_epochs = 5, r = 5, epochs = 20, alpha = 1e-2) -> Tuple[List[float], List[float]]:
    true_ratings_matrix = generate_ratings_matrix(rows = rows, cols = cols, min=min, max=max, step=step)
    _, noised_boolean_matrix = generate_boolean_matrices(epsilon, p, rows = rows, cols = cols)
    noised_revealed_ratings = true_ratings_matrix * noised_boolean_matrix

    # Create matrix completion instances
    true_losses = []
    robust_losses = []
    insecure_completer = generate_insecure_matrix_completion(r=r, epochs=epochs, alpha=alpha, loss_list=true_losses)
    robust_completer = generate_insecure_robust_matrix_completion(w_epochs=w_epochs, w_sub_epochs=w_sub_epochs, r=r, epochs=epochs, alpha=alpha, loss_list=robust_losses)

    # Run training algorithm on both implementations
    insecure_completer.prepare_data(noised_revealed_ratings, noised_boolean_matrix)
    insecure_complete_matrix = insecure_completer.train()

    robust_completer.prepare_data(noised_revealed_ratings, noised_boolean_matrix)
    robust_complete_matrix = robust_completer.train()

    return true_losses, robust_losses

# Loss convergence for insecure completion on noised data (with robust implementation) and non-noised data (standard implementation)
# Returns tuple of lists:
# 1. Losses from training non-robust implementation using non-noised data
# 2. Losses from training robust implementation using noised data
def robust_matrix_completion_loss_convergence_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 100, cols : int = 100, min : int = 0, max : int = 5, step : float = 0.5, w_epochs = 10, w_sub_epochs = 5, r = 5, epochs = 20, alpha = 1e-2) -> Tuple[List[float], List[float]]:
    true_ratings_matrix = generate_ratings_matrix(rows = rows, cols = cols, min=min, max=max, step=step)
    true_boolean_matrix, noised_boolean_matrix = generate_boolean_matrices(epsilon, p, rows = rows, cols = cols)
    true_revealed_ratings = true_ratings_matrix * true_boolean_matrix
    noised_revealed_ratings = true_ratings_matrix * noised_boolean_matrix

    true_losses = []
    robust_losses = []

    insecure_completer = generate_insecure_matrix_completion(r=r, epochs=epochs, alpha=alpha)
    robust_completer = generate_insecure_robust_matrix_completion(w_epochs=w_epochs, w_sub_epochs=w_sub_epochs, r=r, epochs=epochs, alpha=alpha)

    # Run training algorithm on both implementations
    insecure_completer.prepare_data(true_revealed_ratings, true_boolean_matrix)
    insecure_complete_matrix = insecure_completer.train()

    robust_completer.prepare_data(noised_revealed_ratings, noised_boolean_matrix)
    robust_complete_matrix = robust_completer.train()

    return true_losses, robust_losses

# Loss convergence for secure completion on noised data (both robust and non-robust implementation using same noised dataset)
# Returns tuple of lists:
# 1. Losses from training non-robust implementation using noised data
# 2. Losses from training robust implementation using noised data
def secure_noised_matrix_completion_loss_convergence_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 100, cols : int = 100, min : int = 0, max : int = 5, step : float = 0.5, w_epochs = 10, w_sub_epochs = 5, r = 5, epochs = 20, alpha = 1e-2) -> Tuple[List[float], List[float]]:
    public_context, secret_context = setup_contexts()
    
    true_ratings_matrix = generate_ratings_matrix(rows = rows, cols = cols, min=min, max=max, step=step)
    _, noised_boolean_matrix = generate_boolean_matrices(epsilon, p, rows = rows, cols = cols)
    noised_revealed_ratings = true_ratings_matrix * noised_boolean_matrix

    noised_ratings_matrix = util.encrypt_to_ckks_mat(noised_ratings_matrix, public_context)
    noised_ratings_matrix = util.convert_ckks_mat_to_bytes_mat(noised_ratings_matrix)

    noised_boolean_matrix = util.encrypt_to_ckks_mat(noised_boolean_matrix, public_context)
    noised_boolean_matrix = util.convert_ckks_mat_to_bytes_mat(noised_boolean_matrix)

    # Create matrix completion instances
    true_losses = []
    robust_losses = []

    secure_completer = generate_secure_matrix_completion(public_context, secret_context, r=r, epochs=epochs, alpha=alpha, loss_list=true_losses)
    robust_completer = generate_secure_robust_matrix_completion(public_context, secret_context, w_epochs=w_epochs, w_sub_epochs=w_sub_epochs, r=r, epochs=epochs, alpha=alpha, loss_list=robust_losses)

    # Run training algorithm on both implementations
    secure_completer.prepare_data(noised_revealed_ratings, noised_boolean_matrix)
    secure_complete_matrix = secure_completer.train()

    robust_completer.prepare_data(noised_revealed_ratings, noised_boolean_matrix)
    robust_complete_matrix = robust_completer.train()

    return true_losses, robust_losses

# Loss convergence for secure completion on noised data (with robust implementation) and non-noised data (standard implementation)
# Returns tuple of lists:
# 1. Losses from training non-robust implementation using non-noised data
# 2. Losses from training robust implementation using noised data
def secure_robust_matrix_completion_loss_convergence_benchmark(epsilon : float = 0.1, p : float = 0.2, rows : int = 100, cols : int = 100, min : int = 0, max : int = 5, step : float = 0.5, w_epochs = 10, w_sub_epochs = 5, r = 5, epochs = 20, alpha = 1e-2) -> Tuple[List[float], List[float]]:
    public_context, secret_context = setup_contexts()

    ratings_matrix = generate_ratings_matrix(rows = rows, cols = cols, min=min, max=max, step=step)
    true_boolean_matrix, noised_boolean_matrix = generate_boolean_matrices(epsilon, p, rows = rows, cols = cols)
    true_revealed_ratings = ratings_matrix * true_boolean_matrix
    noised_revealed_ratings = ratings_matrix * noised_boolean_matrix

    true_ratings_matrix = util.encrypt_to_ckks_mat(ratings_matrix, public_context)
    true_ratings_matrix = util.convert_ckks_mat_to_bytes_mat(true_ratings_matrix)

    true_boolean_matrix = util.encrypt_to_ckks_mat(true_boolean_matrix, public_context)
    true_boolean_matrix = util.convert_ckks_mat_to_bytes_mat(true_boolean_matrix)

    noised_ratings_matrix = util.encrypt_to_ckks_mat(noised_ratings_matrix, public_context)
    noised_ratings_matrix = util.convert_ckks_mat_to_bytes_mat(noised_ratings_matrix)

    noised_boolean_matrix = util.encrypt_to_ckks_mat(noised_boolean_matrix, public_context)
    noised_boolean_matrix = util.convert_ckks_mat_to_bytes_mat(noised_boolean_matrix)

    true_losses = []
    robust_losses = []

    secure_completer = generate_secure_matrix_completion(public_context, secret_context, r=r, epochs=epochs, alpha=alpha, loss_list=true_losses)
    robust_completer = generate_secure_robust_matrix_completion(public_context, secret_context, w_epochs=w_epochs, w_sub_epochs=w_sub_epochs, r=r, epochs=epochs, alpha=alpha, loss_list=robust_losses)

    # Run training algorithm on both implementations
    secure_completer.prepare_data(true_revealed_ratings, true_boolean_matrix)
    secure_complete_matrix = secure_completer.train()

    robust_completer.prepare_data(noised_revealed_ratings, noised_boolean_matrix)
    robust_complete_matrix = robust_completer.train()

    return true_losses, robust_losses
