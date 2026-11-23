"""Linear Discriminant Analysis (LDA) and Support Vector Machine (SVM) engines.

Mathematical implementations conforming strictly to course syllabus:
  - Within-Class Scatter: S_W = sum_c sum_{x in C_c} (x - mu_c)(x - mu_c)^T
  - Between-Class Scatter: S_B = sum_c N_c (mu_c - mu)(mu_c - mu)^T
  - Generalized Eigenproblem: S_W^{-1} S_B w = lambda w
  - Linear SVM with Margin Width: gamma = 2 / ||w||_2
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.svm import SVC


@dataclass
class LDAResult:
    classes: List[str]
    global_mean: np.ndarray             # (p,)
    class_means: Dict[str, np.ndarray]  # class -> (p,)
    class_counts: Dict[str, int]
    S_W: np.ndarray                     # (p, p) Within-class scatter matrix
    S_B: np.ndarray                     # (p, p) Between-class scatter matrix
    eigenvalues: np.ndarray             # sorted real eigenvalues
    eigenvectors: np.ndarray            # columns are discriminant axes W
    projected_2d: np.ndarray            # (n, 2) projection onto top 2 discriminant directions
    explained_ratio: np.ndarray         # percentage of discriminant power


class LDASimulator:
    """Computes exact within-class and between-class scatter matrices,
    solves the generalized eigenvalue problem, and projects data.
    """

    @staticmethod
    def fit(X: np.ndarray, y: List[str], reg_eps: float = 1e-4) -> LDAResult:
        n, p = X.shape
        classes = sorted(list(set(y)))
        if len(classes) < 2:
            raise ValueError(f"LDA requires at least 2 distinct classes, found {len(classes)}.")

        global_mean = np.mean(X, axis=0)

        # 1. Class means and counts
        class_means = {}
        class_counts = {}
        for c in classes:
            X_c = X[np.array([lbl == c for lbl in y])]
            class_means[c] = np.mean(X_c, axis=0)
            class_counts[c] = len(X_c)

        # 2. Within-class scatter matrix S_W = sum_c sum_{i in c} (x_i - mu_c)(x_i - mu_c)^T
        S_W = np.zeros((p, p), dtype=np.float64)
        for c in classes:
            X_c = X[np.array([lbl == c for lbl in y])]
            diff_c = X_c - class_means[c]
            S_W += diff_c.T @ diff_c

        # 3. Between-class scatter matrix S_B = sum_c N_c (mu_c - mu)(mu_c - mu)^T
        S_B = np.zeros((p, p), dtype=np.float64)
        for c in classes:
            N_c = class_counts[c]
            diff_mean = (class_means[c] - global_mean).reshape(-1, 1)
            S_B += N_c * (diff_mean @ diff_mean.T)

        # 4. Regularize S_W slightly for numerical stability if near singular
        S_W_reg = S_W + reg_eps * np.eye(p)

        # 5. Solve generalized eigenproblem: (S_W^{-1} @ S_B) w = lambda w
        inv_SW_SB = np.linalg.pinv(S_W_reg) @ S_B
        eigenvalues, eigenvectors = np.linalg.eig(inv_SW_SB)

        # Keep real parts
        real_eigs = np.real(eigenvalues)
        real_vecs = np.real(eigenvectors)

        # Sort descending
        sort_idx = np.argsort(real_eigs)[::-1]
        sorted_eigs = real_eigs[sort_idx]
        sorted_vecs = real_vecs[:, sort_idx]

        # Clean small negative values
        sorted_eigs = np.maximum(0.0, sorted_eigs)

        # Normalize eigenvectors to unit length
        for col in range(sorted_vecs.shape[1]):
            norm_v = np.linalg.norm(sorted_vecs[:, col])
            if norm_v > 0:
                sorted_vecs[:, col] /= norm_v

        # Proportions of discriminant power
        tot_eig = np.sum(sorted_eigs)
        explained_ratio = (sorted_eigs / tot_eig) if tot_eig > 0 else np.zeros_like(sorted_eigs)

        # Project data onto top 2 discriminant axes (or top 1 if binary)
        k_proj = min(2, p, len(classes) - 1)
        W_proj = sorted_vecs[:, :k_proj]
        projected = (X - global_mean) @ W_proj

        if k_proj == 1:
            # If 1D, pad with zeros for 2D visualization
            projected_2d = np.column_stack([projected, np.zeros(n)])
        else:
            projected_2d = projected

        return LDAResult(
            classes=classes,
            global_mean=global_mean,
            class_means=class_means,
            class_counts=class_counts,
            S_W=S_W,
            S_B=S_B,
            eigenvalues=sorted_eigs,
            eigenvectors=sorted_vecs,
            projected_2d=projected_2d,
            explained_ratio=explained_ratio,
        )


@dataclass
class SVMResult:
    w: np.ndarray                 # normal vector (2,)
    b: float                      # bias intercept
    margin_width: float           # 2 / ||w||
    support_vector_indices: List[int]
    dual_coef: np.ndarray         # alpha * y
    C: float
    accuracy: float
    x1_range: Tuple[float, float]
    # Decision boundary line coordinates (x_vals, y_boundary, y_gutter_pos, y_gutter_neg)
    line_x: np.ndarray
    line_boundary: np.ndarray
    line_gutter_pos: np.ndarray
    line_gutter_neg: np.ndarray
    sample_margins: np.ndarray    # functional margin y_i * (w^T x_i + b)
    sample_slacks: np.ndarray     # slack xi_i = max(0, 1 - y_i*(w^T x + b))


class SVMSimulator:
    """Fits linear SVM on 2D coordinates, extracts hyperplanes, margin gutters,
    and identifies support vectors.
    """

    @staticmethod
    def fit(X_2d: np.ndarray, y_binary: np.ndarray, C: float = 1.0) -> SVMResult:
        """y_binary must be +1 and -1."""
        clf = SVC(kernel="linear", C=C)
        clf.fit(X_2d, y_binary)

        w = clf.coef_[0]
        b = float(clf.intercept_[0])
        w_norm = float(np.linalg.norm(w))

        margin_width = (2.0 / w_norm) if w_norm > 0 else 0.0
        sv_indices = clf.support_.tolist()

        # Compute functional margins and slacks
        decision_vals = clf.decision_function(X_2d)
        func_margins = y_binary * decision_vals
        slacks = np.maximum(0.0, 1.0 - func_margins)

        accuracy = float(np.mean(clf.predict(X_2d) == y_binary))

        # Generate boundary and gutter lines across x1 range
        x1_min, x1_max = float(np.min(X_2d[:, 0])), float(np.max(X_2d[:, 0]))
        span = x1_max - x1_min
        x1_min -= 0.15 * span
        x1_max += 0.15 * span
        line_x = np.linspace(x1_min, x1_max, 100)

        # w1 * x1 + w2 * x2 + b = 0 => x2 = -(w1 * x1 + b) / w2
        # gutter pos: w^T x + b = +1 => x2 = -(w1 * x1 + b - 1) / w2
        # gutter neg: w^T x + b = -1 => x2 = -(w1 * x1 + b + 1) / w2
        w1, w2 = w[0], w[1]
        if abs(w2) > 1e-8:
            line_boundary = -(w1 * line_x + b) / w2
            line_gutter_pos = -(w1 * line_x + b - 1.0) / w2
            line_gutter_neg = -(w1 * line_x + b + 1.0) / w2
        else:
            # Vertical line fallback
            line_boundary = np.zeros_like(line_x)
            line_gutter_pos = np.zeros_like(line_x)
            line_gutter_neg = np.zeros_like(line_x)

        return SVMResult(
            w=w,
            b=b,
            margin_width=margin_width,
            support_vector_indices=sv_indices,
            dual_coef=clf.dual_coef_[0],
            C=C,
            accuracy=accuracy,
            x1_range=(x1_min, x1_max),
            line_x=line_x,
            line_boundary=line_boundary,
            line_gutter_pos=line_gutter_pos,
            line_gutter_neg=line_gutter_neg,
            sample_margins=func_margins,
            sample_slacks=slacks,
        )
