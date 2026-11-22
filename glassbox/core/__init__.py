"""GlassBox Core Module - Mathematical foundations for text vectorization,
geometric projection, and evidence retrieval.
"""

from .vectorizer import TextVectorizer
from .geometry import (
    compute_sample_covariance,
    jacobi_eigenvalue_solver,
    fit_pca_3d,
    PCA3DResult,
    JacobiConvergenceError,
)
from .retrieval import (
    EvidenceRetrieval,
    SearchResult,
    QueryStatus,
    ShiftComparison,
)

__all__ = [
    "TextVectorizer",
    "compute_sample_covariance",
    "jacobi_eigenvalue_solver",
    "fit_pca_3d",
    "PCA3DResult",
    "JacobiConvergenceError",
    "EvidenceRetrieval",
    "SearchResult",
    "QueryStatus",
    "ShiftComparison",
]
