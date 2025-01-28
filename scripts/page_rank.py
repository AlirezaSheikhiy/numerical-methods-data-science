import pandas as pd
import numpy as np

# Initializing a network as a `dictionary` object
net = {
    "1": [2, 3],
    "2": [],
    "3": [1, 2, 5],
    "4": [5, 6],
    "5": [4, 6],
    "6": [4]
}

# Get the network size
n = len(net)

# Constructing the **Hyperlinks** Matrix
H = np.zeros(shape=(n, n))

for i in range(n):
    node_links = net[str(i+1)]
    for j in range(len(node_links)):
        H[i, node_links[j]-1] = 1/len(node_links)

# Constructing the **Stochastic** Matrix
S = H.copy()

for i in range(n):
    node_links = net[str(i+1)]
    if len(node_links) == 0:
        S[i, :] = 1/n

# Set the **Scaling Parameter**
alpha = 0.85

# Constructing the **Teleportation** Matrix
E = (1/n) + np.ones((n, n))

# Constructing the **Google** Matrix
G = alpha * S + (1-alpha) * E

# Initializing the **PageRank** vector
P = np.ones((n, 1)) * (1/n)

# Performing Google's Adjusted PageRank Method
k = 10
for i in range(k):
    V = P.T @ G
    P = V.T.copy()

# Printing the results table
df = pd.DataFrame(data=P, columns=["Ranks"])
df.index = range(1, n+1)
df.index.name = "Page"
print(df)
