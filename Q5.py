import numpy as np

# Define the matrix
A = np.array([
    [5, 6, 7, 8],
    [1, 3, 5, 4],
    [1, 0.5, 4, 2],
    [3, 4, 3, 1]
], dtype=float)

# Compute full SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Rank-1 approximation
A1 = S[0] * np.outer(U[:, 0], Vt[0, :])

# Rank-2 approximation
A2 = (
    S[0] * np.outer(U[:, 0], Vt[0, :]) +
    S[1] * np.outer(U[:, 1], Vt[1, :])
)

# Print results
print("Rank-1 Approximation of A:\n", A1)
print("Rank-2 Approximation of A:\n", A2)

# Optional: Compute errors
error_1 = np.linalg.norm(A - A1, 'fro')
error_2 = np.linalg.norm(A - A2, 'fro')
print("Frobenius error (rank-1):", error_1)
print("Frobenius error (rank-2):", error_2)