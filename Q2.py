import numpy as np


# --------------------- 2.a ---------------------

def FwdSub(L, b):
    n = len(b)
    x = np.zeros_like(b, dtype=np.float64)
    for i in range(n):
        sum_ = sum(L[i, j] * x[j] for j in range(i))
        x[i] = (b[i] - sum_) / L[i, i]
    return x


def BwdSub(U, b):
    n = len(b)
    x = np.zeros_like(b, dtype=np.float64)
    for i in reversed(range(n)):
        sum_ = sum(U[i, j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - sum_) / U[i, i]
    return x

# --------------------- 2.b ---------------------

# defining given matrices and the normal equation
A = np.array([
    [2, 1, 2],
    [1, -2, 1],
    [1, 2, 3],
    [1, 1, 1]
], dtype = np.float64)

b = np.array([6, 1, 5, 2], dtype = np.float64)

AtA = A.T @ A
Atb = A.T @ b

# Cholesky decomposition
L = np.linalg.cholesky(AtA)

# using forward substitution solve (L @ y = Atb)
y = FwdSub(L, Atb)

# using backwards substitution solve (L.T @ x = y)
x = BwdSub(L.T, y)

#print solution
print("LS solution x:\n", x)


# --------------------- 2.c ---------------------

# QR factorization:
Q, R = np.linalg.qr(A)
Qtb = Q.T @ b
x_qr = BwdSub(R, Qtb)
print("Solution using QR:\n", x_qr)


# SVD factoriztion:
U, S, Vt = np.linalg.svd(A)
y = U.T @ b
z = y[:len(S)] / S  # Solve Σz = y, only for nonzero singular values
x_svd = Vt.T @ z
print("Solution using SVD:\n", x_svd)