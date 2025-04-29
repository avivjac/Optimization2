import numpy as np

# --------------------- 3.a ---------------------

# regular gram schmidt algorithm implementation
def gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        q = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            q = q - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(q)
        Q[:, j] = q / R[j, j]
    return Q, R

def modified_gram_schmidt(A):
    m, n = A.shape
    Q = np.array(A, dtype=float)  # clone to avoid modifying A: changes happen in place
    R = np.zeros((n, n))
    
    for i in range(n):
        R[i, i] = np.linalg.norm(Q[:, i])
        Q[:, i] = Q[:, i] / R[i, i]
        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, i], Q[:, j])
            Q[:, j] = Q[:, j] - R[i, j] * Q[:, i]
    return Q, R

# --------------------- 3.b ---------------------

def test_qr(epsilon, algorithm):
    A = np.array([
        [1, 1, 1],
        [epsilon, 0, 0],
        [0, epsilon, 0],
        [0, 0, epsilon]
    ], dtype=float)

    Q, R = algorithm(A)
    orth_error = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), ord='fro')
    return orth_error

epsilons = [1, 1e-10]

print("Orthogonality errors:")
for eps in epsilons:
    gs_err = test_qr(eps, gram_schmidt)
    mgs_err = test_qr(eps, modified_gram_schmidt)
    print(f"epsilon = {eps:.0e} , Gram-Schmidt: {gs_err:.2e}, Modified Gram-Schmidt: {mgs_err:.2e}")