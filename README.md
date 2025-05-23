# Optimization Methods – Homework 2

Welcome to the GitHub repository for my second homework assignment in the course **Optimization Methods with Applications** at Ben-Gurion University.

This repository includes Python implementations and visualizations for a variety of topics in numerical optimization, linear algebra, and matrix decompositions.

## 📁 Structure

Each file corresponds to a different question in the assignment:

### ✅ Q1 – Matrix Norms and Spectral Radius
- Computation of matrix norms: \( \|A\|_1 \), \( \|A\|_\infty \), and the spectral norm \( \|A\|_2 \).
- Uses SVD to find the largest singular value.
- Compares the norms to the spectral radius (maximum modulus of eigenvalues).

### ✅ Q2 – Least Squares via Decompositions
- Implements forward and backward substitution.
- Solves a least squares problem using:
  - Cholesky factorization of \( A^TA \),
  - QR decomposition,
  - SVD decomposition.
- Verifies that all methods produce similar solutions.

### ✅ Q3 – QR Factorization: Stability Comparison
- Implements:
  - Classical Gram-Schmidt algorithm.
  - Modified Gram-Schmidt algorithm.
- Tests orthogonality error for ill-conditioned matrices (parameter \( \varepsilon \)).
- Highlights the improved numerical stability of the modified version.

### ✅ Q5 – Low-Rank Approximation
- Computes rank-1 and rank-2 approximations of a matrix using SVD.
- Measures and compares approximation error using the Frobenius norm.

### ✅ Q6 – Quadratic Forms and Geometry
- Visualizes contours of quadratic functions (paraboloids).
- Explores rotation via orthogonal matrices.
- Demonstrates the equivalence of coordinate transformation and matrix formulation \( f(x) = x^\top U D U^\top x \).
- Shows how the nature of \( A \) (SPD vs indefinite) affects the shape (minimum vs saddle).

## 📊 Visualizations
Several questions include `matplotlib`-based plots to visualize:
- Contour lines of level sets.
- Differences in decomposition-based approximations.
- Stability of orthogonality.

## 💡 Concepts Covered
- Matrix and vector norms
- Cholesky, QR, and SVD decompositions
- Regularized least squares (Tikhonov)
- Orthogonality and numerical stability
- Spectral radius vs norm comparison
- Quadratic forms and their geometric interpretation

## 🔧 Requirements
- Python 3.8+
- NumPy
- Matplotlib

Install dependencies via:

```bash
pip install numpy matplotlib

