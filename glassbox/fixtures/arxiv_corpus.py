"""Curated 64-paper arXiv research corpus across 4 fundamental categories:
- cs.CV (Computer Vision)
- cs.CL (Computation and Language / NLP)
- cs.LG (Machine Learning / Deep Learning Theory)
- stat.ML (Optimization & Statistical Learning)
"""

from glassbox.data.arxiv_corpus import (
    PaperRecord,
    ARXIV_PAPERS,
    get_arxiv_corpus,
    get_papers_by_category,
)

__all__ = [
    "PaperRecord",
    "ARXIV_PAPERS",
    "get_arxiv_corpus",
    "get_papers_by_category",
]
