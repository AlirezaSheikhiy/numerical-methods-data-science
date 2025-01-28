from svd import SVD, rank
import numpy as np
import pandas as pd


# Moore-Penrose Inverse method
def moore_penrose_inv(A):
    U, S, VT = SVD(A)
    S_plus = S.T
    for i in range(rank(A)):
        S_plus[i, i] = 1 / S_plus[i, i]
    # S_plus = np.linalg.inv(S)
    A_plus = VT.T @ S_plus @ U.T
    return A_plus, S_plus


# Driver code
if __name__ == "__main__":
    # Initialize matrix 'A'
    A = np.array([[0.96, 1.72], [2.28, 0.96]])
    # A = np.array([[0, -1.6, 0.6], [0, 1.2, 0.8], [0, 0, 0], [0, 0, 0]])
    # A = np.array([[-2, 1, 2], [6, 6, 2]])

    # Using moore_penrose_inv() method to calculate inverse
    A_plus, S_plus = moore_penrose_inv(A)

    # Print the results
    print("A_plus =")
    print(pd.DataFrame(A_plus.round(4)).to_string(index=False, header=False))
    print("\nS_plus =")
    print(pd.DataFrame(S_plus.round(4)).to_string(index=False, header=False))
