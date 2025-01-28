import numpy as np
from sklearn.model_selection import train_test_split
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

# Use `train_test_split` to split the data
X_train, X_test, y_train, y_test = train_test_split(
    A, y, test_size=0.2, random_state=77)

# Calculating the pseudo inverse of `X_train` matrix
X_train_plus = moore_penrose_inv(X_train)

# Calculating the Linear Regression coefficients using `train` data
beta_hat = X_train_plus @ y_train

# Predicting the `y` vector for the train and test data
y_pred_train = X_train @ beta_hat
y_pred_test = X_test @ beta_hat

# Calculating the Errors for both train and test data
norm_train = np.linalg.norm(y_train - y_pred_train, ord=2)
norm_test = np.linalg.norm(y_test - y_pred_test, ord=2)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

# Printing the results
print(f"Norm Error for Train data: {norm_train}")
print(f"Norm Error for Test data: {norm_test}")
print(f"Mean Squared Error for Train data: {mse_train}")
print(f"Mean Squared Error for Test data: {mse_test}")
