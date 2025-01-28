import time
import numpy as np

# Initializing the problem
A = np.array([[1, 2, 3], [1, -1, 1], [1, -1, 0]])
b = np.array([[2, 1, 0]]).T

# Converting matrix A to a Positive Definite and Symmetric matrix
eps = np.finfo(np.float64).eps
A = A.T @ A
I = np.eye(A.shape[0])
A = A.T @ A + eps * I

# Setting the hyper-parameters
x_0 = np.array([[1, 1, 1]]).T
treshold = 1e-6

# Gradient-Descent algorithm
k = 0
x_k = x_0
start_time = time.time()
while True:
    r_k = b - A @ x_k
    a_k = (r_k.T @ r_k) / (r_k.T @ A @ r_k)
    x_k_plus_1 = x_k + a_k * r_k
    k+=1
    if np.linalg.norm(x_k_plus_1 - x_k, ord=2) <= treshold:
        break
    x_k = x_k_plus_1
elapsed_time = time.time() - start_time

# Print the results
print(f"Number of iterations: {k}")
print(f"Time elapsed: {round(elapsed_time, 4)} seconds")
print(f"Treshold: {treshold}")
print(f"norm_2(Ax-b) = {np.round(np.linalg.norm(A @ x_k_plus_1 - b, ord=2), 4)}")
