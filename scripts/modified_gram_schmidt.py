import numpy as np


def gram_schmidt(A):
    # Validate input
    if not isinstance(A, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    m, n = A.shape
    if m < n:
        raise ValueError(
            "Number of rows must be greater than or equal to the number of columns.")

    # Initialize Q and R
    Q = np.zeros_like(A)
    R = np.zeros((n, n))

    for j in range(n):
        # Start with the j-th column of A
        Q[:, j] = A[:, j]

        # Normalize q_j
        R[j, j] = np.linalg.norm(Q[:, j])
        if R[j, j] == 0:
            raise ValueError("Matrix A contains linearly dependent columns.")

        Q[:, j] /= R[j, j]

        # Subtract the projections onto the previous q_i's
        for i in range(j+1, n):
            R[i, j] = np.dot(Q[:, i], Q[:, j])
            Q[:, j] -= R[i, j] * Q[:, i]

    return Q, R


if __name__ == "__main__":
    # Define a matrix A
    A = np.array([[2, 4, -4], [1, 5, -5], [2, 10, 5]], dtype=float)
    # A = np.array([[1, 2, -1], [1, -1, 2], [1, -1, 2], [-1, 1, 1]], dtype=float)

    # Perform QR decomposition
    Q, R = gram_schmidt(A)

    # Output the results
    print(f"A =\n{A}\n\nQ =\n{Q}\n\nR =\n{R}")

    # Verify that A = QR (Should print True)
    print(f"\nVerification (A = QR): {np.allclose(A, Q @ R)}")