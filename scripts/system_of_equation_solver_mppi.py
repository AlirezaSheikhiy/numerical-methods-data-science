from mppseudoinv import moore_penrose_inv
import numpy as np
import pandas as pd

# Initialize the system of equations
A = np.array([[1, 2], [2, 1], [-3, 1], [-1, -3]])
b = np.array([[3, 3, -1, -4]])

# Calculate the pseudoinverse
A_plus, _ = moore_penrose_inv(A)

# Solve the system of equations
x = A_plus @ b.T

# Print the results
print("x =")
print(pd.DataFrame(x.round(4)).to_string(index=False, header=False))
