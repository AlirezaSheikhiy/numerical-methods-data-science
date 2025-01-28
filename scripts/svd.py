import numpy as np
import pandas as pd


# Singular Values Decomposition method
def SVD(A):
    B = A.T @ A
    C = A @ A.T

    # eigenvalues_B, V = np.linalg.eig(B)
    # eigenvalues_C, U = np.linalg.eig(C)

    eigenvalues_B, V = np.linalg.eigh(B)
    eigenvalues_C, U = np.linalg.eigh(C)

    """ Normalizing U and V matrices
    for i in range(len(V)):
        V[:, i] = V[:, i] / np.linalg.norm(V[:, i], ord=2)

    for i in range(len(U)):
        U[:, i] = U[:, i] / np.linalg.norm(U[:, i], ord=2)
    """

    sorted_indices_B = np.argsort(eigenvalues_B)[::-1]
    eigenvalues_B = eigenvalues_B[sorted_indices_B]
    V = V[:, sorted_indices_B]

    singularvalues = np.sqrt(eigenvalues_B)
    # singularvalues = np.sqrt(np.maximum(eigenvalues_B, 0))

    Sigma = np.zeros(A.shape)
    np.fill_diagonal(Sigma, singularvalues)

    sorted_indices_C = np.argsort(eigenvalues_C)[::-1]
    eigenvalues_C = eigenvalues_C[sorted_indices_C]
    U = U[:, sorted_indices_C]

    """ Adjust signs of U and V to ensure consistency
    for i in range(len(singularvalues)):
        if U[0, i] < 0:
            U[:, i] = -U[:, i]
        if V[i, 0] < 0:
            V[:, i] = -V[:, i]
    """

    return U, Sigma, V.T


# A method to calculate matrix rank using SVD
def rank(A, threshold=0):
    _, S, _ = SVD(A)
    return np.sum(np.diag(S) > threshold)


# Driver code
if __name__ == "__main__":
    # Initialize matrix 'A'
    A = np.array([[0.96, 1.72], [2.28, 0.96]])
    # A = np.array([[0, -1.6, 0.6], [0, 1.2, 0.8], [0, 0, 0], [0, 0, 0]])
    # A = np.array([[-2, 1, 2], [6, 6, 2]])

    # Using SVD() method to extract values
    U, S, VT = SVD(A)

    # Reconstruct the original matrix from U, Sigma, and VT
    A_hat = U @ S @ VT

    # Calculate the norm_2 of the difference between A and its reconstruction
    norm2 = np.linalg.norm((A - A_hat), ord=2)

    # Print the results
    print("U =")
    print(pd.DataFrame(U.round(4)).to_string(index=False, header=False))

    print("\nSigma =")
    print(pd.DataFrame(S.round(4)).to_string(index=False, header=False))

    print("\nV.T =")
    print(pd.DataFrame(VT.round(4)).to_string(index=False, header=False))

    print("\nA =")
    print(pd.DataFrame(A.round(4)).to_string(index=False, header=False))

    print("\nA_hat =")
    print(pd.DataFrame(A_hat.round(4)).to_string(index=False, header=False))

    print(f"\n\nRank = {rank(A)}")
    print(f"norm2(A - A_hat) = {norm2}")
