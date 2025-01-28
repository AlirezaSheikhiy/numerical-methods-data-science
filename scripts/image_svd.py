import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import euclidean


# Image reader method
def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# Read the image file
path = "../images/peppers.jpg"
image = read_image(path)

# Convert the image to a matrix
A = np.array(image)

# Implementing SVD method
U, Sigma, VT = np.linalg.svd(A)
S = np.zeros(A.shape)
np.fill_diagonal(S, Sigma)

# Elbow Method (Choosing the best value of k as the number of singular values)
x = np.arange(1, len(Sigma) + 1)
y = np.log(Sigma)

p1 = np.array([x[0], Sigma[0]])
p2 = np.array([x[-1], Sigma[-1]])

distances = []

for i in range(len(Sigma)-1):
    p = np.array([x[i], Sigma[i]])
    d = np.abs(np.cross(p2 - p1, p1 - p)) / euclidean(p1, p2)
    distances.append(d)

elbow_index = np.argmax(distances)
elbow_value = np.max(distances)

plt.figure(figsize=(10, 6))
plt.plot(x, y, marker=".", label="Singular Values", linewidth=1, zorder=1)
plt.scatter(x[elbow_index], y[elbow_index], marker="o", color='red',
            label=f"Elbow Point (k={elbow_index + 1})", linewidths=3, zorder=2)
plt.xlabel("Index of Singular Values")
plt.ylabel("Singular Values Magnitude (Log Scale)")
plt.title("Singular Values and Elbow Method (Logarithmic Scale)")
plt.grid(which="both", linestyle="--", linewidth=0.5)

plt.legend()
plt.show()

print("*"*80)
print(f"The best number of components (k) by Elbow Method: {elbow_index + 1}")
print(f"The distance between these two singular values is {elbow_value}")
print("*"*80)

# Reconstructing the primary matrix
k = elbow_index + 1

B = U @ S @ VT
C = U[:, :k] @ np.diag(Sigma[:k]) @ VT[:k, :]

fig, axes = plt.subplots(1, 3, figsize=(10, 6))

axes[0].imshow(image, cmap='gray')
axes[0].axis('off')
axes[0].set_title("Original Image")

axes[1].imshow(C, cmap='gray')
axes[1].axis('off')
axes[1].set_title(f"Reconstructed Image\nwith k={k} singular values")

axes[2].imshow(B, cmap='gray')
axes[2].axis('off')
axes[2].set_title("Reconstructed Image\nwith all singular values")

plt.tight_layout()
plt.show()

# Showing the extracted images as result
fig, axes = plt.subplots(1, 3, figsize=(10, 6))

axes[0].imshow(abs(U * 255).astype("uint8"), cmap='gray')
axes[0].axis('off')
axes[0].set_title('U Matrix')

axes[1].imshow(abs(S * 255).astype("uint8"), cmap='gray')
axes[1].axis('off')
axes[1].set_title('Singular Values Matrix')

axes[2].imshow(abs(VT * 255).astype("uint8"), cmap='gray')
axes[2].axis('off')
axes[2].set_title('V Transposed Matrix')

plt.tight_layout()
plt.show()

# More matrix information about this image
print("Extra Information:")
print(f"dim(A) = {A.shape}")
print(f"dim(U) = {U.shape}")
print(f"dim(S) = {S.shape}")
print(f"dim(VT) = {VT.shape}")
print(f"rank(A) = {np.linalg.matrix_rank(A)}")
print(f"norm_2(A - A_k) = {np.linalg.norm(A-C, ord=2)}")
print("*"*80)
