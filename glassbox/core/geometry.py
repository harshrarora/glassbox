"""Geometric projection and PCA from scratch with custom Jacobi eigensolver.

Formulas:
  Xc = X - mu
  Sigma = Xc^T Xc / (n - 1)
  scores = Xc V2
  projected points in original coordinates = scores V2^T + mu
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class JacobiConvergenceError(Exception):
    """Raised when the Jacobi eigenvalue solver fails to converge within max iterations."""
    pass


def compute_sample_covariance(Xc: np.ndarray) -> np.ndarray:
    """Compute sample covariance Sigma = (Xc^T @ Xc) / (n - 1).
    Expects centered data Xc of shape (n, p) where n > 1.
    """
    n, p = Xc.shape
    if n <= 1:
        raise ValueError(f"Sample covariance requires at least 2 samples, got {n}.")
    return (Xc.T @ Xc) / (n - 1)


def jacobi_eigenvalue_solver(
    A: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Symmetric Jacobi eigensolver for real symmetric matrices (specialized for 3x3 classroom use).

    Rotates away the largest off-diagonal element iteratively using Givens/Jacobi rotations
    until max(|A_pq|) / max(|A_initial|) < tol (scale-relative).

    Returns:
      eigenvalues: sorted in descending order (shape: (p,))
      eigenvectors: corresponding orthonormal columns (shape: (p, p))
      iterations: number of sweep iterations performed
    """
    A = np.array(A, dtype=np.float64, copy=True)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not A.size or not np.isfinite(A).all():
        raise ValueError("Matrix must be nonempty, finite and square.")
    if not np.isfinite(tol) or tol <= 0 or max_iter < 1:
        raise ValueError("Require positive tolerance and iteration budget.")
    p = A.shape[0]
    scale = float(np.max(np.abs(A)))
    if not np.allclose(A, A.T, atol=1e-12 * scale, rtol=0):
        raise ValueError("Matrix must be symmetric.")
    # Relative stopping criterion is invariant to the units of covariance.
    if scale > 0:
        A /= scale

    V = np.eye(p, dtype=np.float64)

    iteration = 0
    while iteration < max_iter:
        # Find maximum off-diagonal entry in upper triangle
        max_val = 0.0
        p_idx, q_idx = 0, 1
        for i in range(p):
            for j in range(i + 1, p):
                val = abs(A[i, j])
                if val > max_val:
                    max_val = val
                    p_idx, q_idx = i, j

        if max_val < tol:
            break

        # Compute Jacobi rotation angle
        App = A[p_idx, p_idx]
        Aqq = A[q_idx, q_idx]
        Apq = A[p_idx, q_idx]

        # tau = (Aqq - App) / (2 * Apq)
        tau = (Aqq - App) / (2.0 * Apq)
        if tau >= 0:
            t = 1.0 / (tau + np.hypot(1.0, tau))
        else:
            t = -1.0 / (-tau + np.hypot(1.0, tau))

        c = 1.0 / np.sqrt(1.0 + t * t)
        s = t * c

        # Construct Jacobi rotation matrix J
        J = np.eye(p, dtype=np.float64)
        J[p_idx, p_idx] = c
        J[q_idx, q_idx] = c
        J[p_idx, q_idx] = s
        J[q_idx, p_idx] = -s

        # Apply similarity transformation: A_next = J^T @ A @ J
        A = J.T @ A @ J
        # Accumulate eigenvectors: V_next = V @ J
        V = V @ J

        iteration += 1

    # The final allowed rotation may have converged. Inspect the updated matrix.
    max_val = float(np.max(np.abs(A - np.diag(np.diag(A)))))
    if max_val >= tol:
        raise JacobiConvergenceError(
            f"Jacobi eigensolver did not converge within {max_iter} iterations (max off-diagonal = {max_val:.2e} >= tol {tol:.2e})."
        )

    # Diagonal entries are the eigenvalues
    eigenvalues = np.diag(A).copy() * scale

    # Sort eigenpairs by descending eigenvalue
    sort_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sort_indices]
    sorted_eigenvectors = V[:, sort_indices]

    # Enforce deterministic sign convention:
    # The component with the maximum absolute value in each eigenvector is made positive.
    for col in range(p):
        max_component_idx = int(np.argmax(np.abs(sorted_eigenvectors[:, col])))
        if sorted_eigenvectors[max_component_idx, col] < 0:
            sorted_eigenvectors[:, col] *= -1.0

    return sorted_eigenvalues, sorted_eigenvectors, iteration


@dataclass
class PCA3DResult:
    """Encapsulates all intermediate and final mathematical entities of the 3D PCA projection."""
    terms: List[str]
    X_original: np.ndarray       # (n, 3)
    mean_vector: np.ndarray      # (3,)
    X_centered: np.ndarray      # (n, 3)
    covariance_matrix: np.ndarray  # (3, 3)
    eigenvalues: np.ndarray      # (3,)
    eigenvectors: np.ndarray     # (3, 3) columns are v1, v2, v3
    explained_variance_ratio: np.ndarray  # (3,)
    scores_2d: np.ndarray        # (n, 2)
    projected_3d: np.ndarray     # (n, 3) points in original coordinates
    residuals_3d: np.ndarray     # (n, 3) orthogonal projection vectors
    jacobi_iterations: int
    is_degenerate: bool          # True if near-equal eigenvalues detected


def fit_pca_3d(X: np.ndarray, term_names: List[str]) -> PCA3DResult:
    """Fit 3D PCA to 2D projection on term coordinates X (n, 3).
    Follows exact course syllabus steps:
      1. Mean calculation mu
      2. Centering Xc = X - mu
      3. Covariance Sigma = (Xc^T Xc) / (n - 1)
      4. Jacobi Eigensolver -> eigenvalues, eigenvectors (loadings)
      5. 2D scores = Xc @ V2
      6. Reconstruction in original coordinates = scores @ V2^T + mu
    """
    n, p = X.shape
    if p != 3:
        raise ValueError(f"Expected 3 features for classroom 3D PCA, got {p}.")
    if n < 2:
        raise ValueError(f"Expected at least 2 documents, got {n}.")

    # Step 1: Mean vector
    mean_vec = np.mean(X, axis=0)

    # Step 2: Centered coordinates
    Xc = X - mean_vec

    # Step 3: Sample covariance
    cov_matrix = compute_sample_covariance(Xc)

    # Step 4: Jacobi eigensolver
    eigenvalues, eigenvectors, n_iter = jacobi_eigenvalue_solver(cov_matrix)

    # Check for subspace degeneracy (near-equal eigenvalues)
    is_degenerate = False
    if len(eigenvalues) >= 2:
        diff_1_2 = abs(eigenvalues[0] - eigenvalues[1])
        diff_2_3 = abs(eigenvalues[1] - eigenvalues[2])
        if diff_1_2 < 1e-6 or diff_2_3 < 1e-6:
            is_degenerate = True

    # Explained variance ratios
    total_var = np.sum(eigenvalues)
    if total_var > 0:
        evr = eigenvalues / total_var
    else:
        evr = np.zeros_like(eigenvalues)

    # Step 5: Top 2 principal components: V2 = [v1, v2]
    V2 = eigenvectors[:, :2]  # shape (3, 2)
    scores_2d = Xc @ V2       # shape (n, 2)

    # Step 6: Projected points in original 3D coordinates
    projected_3d = (scores_2d @ V2.T) + mean_vec  # shape (n, 3)

    # Residual orthogonal vectors: r = X - projected_3d = Xc - scores_2d @ V2.T
    residuals_3d = X - projected_3d

    return PCA3DResult(
        terms=term_names,
        X_original=X,
        mean_vector=mean_vec,
        X_centered=Xc,
        covariance_matrix=cov_matrix,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        explained_variance_ratio=evr,
        scores_2d=scores_2d,
        projected_3d=projected_3d,
        residuals_3d=residuals_3d,
        jacobi_iterations=n_iter,
        is_degenerate=is_degenerate,
    )


@dataclass
class SVDRankKResult:
    """Encapsulates truncated rank-k SVD reconstruction entities."""
    k: int
    singular_values: np.ndarray      # (p,)
    U: np.ndarray                    # (n, p)
    Vt: np.ndarray                   # (p, p)
    X_reconstructed: np.ndarray      # (n, p)
    residual_matrix: np.ndarray      # (n, p)
    frobenius_error: float
    relative_error: float
    energy_captured: float           # sum(sigma_1..k^2) / sum(sigma_all^2)


def compute_svd_rank_k(X: np.ndarray, k: int) -> SVDRankKResult:
    """Compute exact full SVD and rank-k truncated reconstruction:
    X_hat_k = sum_{j=1}^k sigma_j * u_j * v_j^T
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not X.size or not np.isfinite(X).all():
        raise ValueError("SVD requires a nonempty finite matrix.")
    n, p = X.shape
    if not isinstance(k, (int, np.integer)) or k < 1 or k > min(n, p):
        raise ValueError(f"Rank k must be between 1 and {min(n, p)}, got {k}.")

    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Reconstruct with rank k
    # U_k is (n, k), S_k is (k,), Vt_k is (k, p)
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    X_hat_k = (U_k * S_k) @ Vt_k
    residual = X - X_hat_k

    frob_orig = float(np.linalg.norm(X, ord="fro"))
    frob_err = float(np.linalg.norm(residual, ord="fro"))
    rel_err = (frob_err / frob_orig) if frob_orig > 0 else 0.0

    tot_energy = float(np.sum(S ** 2))
    k_energy = float(np.sum(S_k ** 2))
    energy_ratio = (k_energy / tot_energy) if tot_energy > 0 else 1.0

    return SVDRankKResult(
        k=k,
        singular_values=S,
        U=U,
        Vt=Vt,
        X_reconstructed=X_hat_k,
        residual_matrix=residual,
        frobenius_error=frob_err,
        relative_error=rel_err,
        energy_captured=energy_ratio,
    )

