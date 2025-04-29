import numpy as np



# Define the matrix A
A = np.array([
    [1, 2, 3, 4],
    [2, 4, -4, 8],
    [-5, 4, 1, 5],
    [5, 0, -3, -7]
])


# -------------------- 1.b --------------------

# Compute the SVD
U, S, Vt = np.linalg.svd(A)

# The largest singular value is S[0]
sigma_max = S[0]

# The corresponding right singular vector (maximizing x) is Vt[0]
x_max = Vt[0]

# Normalize x
x_max_normalized = x_max / np.linalg.norm(x_max)

# Print results
print("Largest singular value (sigma_max):", sigma_max)
print("Maximizing vector x (right singular vector corresponding to sigma_max):")
print(x_max_normalized)

# -------------------- 1.c --------------------

# Spectral radius: max of absolute values of eigenvalues
eigenvalues = np.linalg.eigvals(A)
spectral_radius = max(abs(eigenvalues))

# Norms
norm_1 = np.linalg.norm(A, 1)
norm_inf = np.linalg.norm(A, np.inf) 
norm_2 = np.linalg.norm(A, 2)  # Largest singular value

# Print comparison
print("Spectral Radius:        ", spectral_radius)
print("Induced 1-Norm:          ", norm_1)
print("Induced 2-Norm (sigma_max): ", norm_2)
print("Induced inf-Norm:          ", norm_inf)