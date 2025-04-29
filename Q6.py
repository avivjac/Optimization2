import numpy as np
import matplotlib.pyplot as plt

# Question 6 
# Part a

# Create a grid of points
x = np.linspace(-1, 1, 400)
y = np.linspace(-1, 1, 400)
X, Y = np.meshgrid(x, y)

# function definition
# f(x, y) = ax² + by²
def f(X, Y, a, b):
    return a * X**2 + b * Y**2

# (a=1, b=2)
Z1 = f(X, Y, 1, 2)
plt.figure(figsize=(8, 6))
contour1 = plt.contourf(X, Y, Z1, levels=50, cmap='viridis')
plt.colorbar(contour1)
plt.title('Contour plot of f(x, y) = x² + 2y²')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# (a=1, b=0.5)
Z2 = f(X, Y, 1, 0.5)
plt.figure(figsize=(8, 6))
contour2 = plt.contourf(X, Y, Z2, levels=50, cmap='plasma')
plt.colorbar(contour2)
plt.title('Contour plot of f(x, y) = x² + 0.5y²')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Part b
# Proof in the notebook

# Part 

# Draw the contour plot of f(x, y) = x² + 2y² after rotation θ=π/8
a = 1
b = 2
D = np.diag([a, b])
theta = np.pi / 8
U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

# Calculate the rotated coordinates
XY = np.stack([X.ravel(), Y.ravel()], axis=0) 
XY_rotated = U.T @ XY

# The rotated coordinates are now in XY_rotated
X_rot = XY_rotated[0, :].reshape(X.shape)
Y_rot = XY_rotated[1, :].reshape(Y.shape)
Z_rotated = a * X_rot**2 + b * Y_rot**2

# Calculate the rotated coordinates using x^TUDU^Tx
Z_direct = np.einsum('ij,jk,ik->i', XY.T, U @ D @ U.T, XY.T).reshape(X.shape)

# Plot the rotated contour plot
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
contour1 = plt.contourf(X, Y, Z_rotated, levels=50, cmap='coolwarm')
plt.colorbar(contour1)
plt.title('By rotating coordinates (using x\')')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')

plt.subplot(1,2,2)
contour2 = plt.contourf(X, Y, Z_direct, levels=50, cmap='coolwarm')
plt.colorbar(contour2)
plt.title('By direct formula xᵀUDUᵀx')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')

plt.tight_layout()
plt.show()

# Check the difference between the two methods
print("Max difference between methods:", np.max(np.abs(Z_rotated - Z_direct)))
print("We can see that the diffrance ius very small an dprobably due to a numerical error that occur because of the computer limits.")

# Part d

