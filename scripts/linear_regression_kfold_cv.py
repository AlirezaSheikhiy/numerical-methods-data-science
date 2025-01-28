import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# Data Generator Method
def linear_data_generator(size=(1, 1), noise=0.1, seed=100):
    n = size[0]
    p = size[1]
    np.random.seed(seed=seed)
    X = np.random.rand(n, p)
    coefficients = np.random.rand(p, 1)
    y = X.dot(coefficients) + noise * np.random.randn(n, 1)
    return X, y


# Moore-Penrose Pseudo Inverse Method
def moore_penrose_inv(A):
    U, s, VT = np.linalg.svd(A)
    S = np.zeros(A.shape)
    np.fill_diagonal(S, s)
    S_plus = S.T
    for i in range(np.linalg.matrix_rank(A)):
        S_plus[i, i] = 1 / S_plus[i, i]
    A_plus = VT.T @ S_plus @ U.T
    return A_plus


# Generating matrix `X` and `y`
X_bar, y_bar = linear_data_generator(size=(100, 5), noise=0.1, seed=77)
X = X_bar.T @ X_bar
y = X_bar.T @ y_bar

# Constructing coefficients matrix
A = np.insert(X, 0, np.ones(len(X)), axis=1)

# Initializing KFold object
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=77)

# K-Fold Cross-Validation
mse_list = []
norm_list = []

for train_index, test_index in kf.split(A):
    X_train, X_test = A[train_index], A[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train_plus = moore_penrose_inv(X_train)

    beta_hat = X_train_plus @ y_train

    y_pred = X_test @ beta_hat

    error = np.linalg.norm(y_test - y_pred, ord=2)
    mse = mean_squared_error(y_test, y_pred)

    norm_list.append(error)
    mse_list.append(mse)

# Printing the results
df = pd.DataFrame(data={"Folds": range(1, k+1),
                        "Norms Error": norm_list,
                        "MSE": mse_list})
print("The norms error and mse table for each fold:\n")
print(df.to_string(index=False), end="\n\n")
print("Average Norms:", np.mean(norm_list))
print("Average Mean Squared Error:", np.mean(mse_list))
