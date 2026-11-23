"""GlassBox Core Module - Mathematical foundations for text vectorization,
geometric projection, dimensionality reduction, and linear classification.
"""

from .vectorizer import TextVectorizer
from .geometry import (
    compute_sample_covariance,
    jacobi_eigenvalue_solver,
    fit_pca_3d,
    PCA3DResult,
    compute_svd_rank_k,
    SVDRankKResult,
    JacobiConvergenceError,
)
from .classifiers import (
    LDASimulator,
    LDAResult,
    SVMSimulator,
    SVMResult,
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
    "compute_svd_rank_k",
    "SVDRankKResult",
    "JacobiConvergenceError",
    "LDASimulator",
    "LDAResult",
    "SVMSimulator",
    "SVMResult",
    "EvidenceRetrieval",
    "SearchResult",
    "QueryStatus",
    "ShiftComparison",
]
