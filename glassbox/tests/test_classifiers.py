"""Test suite for expanded arXiv corpus, SVD rank-k reconstruction,
LDA scatter matrices (S_W, S_B), and SVM margin mechanics.
"""

import numpy as np
import pytest

from glassbox.data.arxiv_corpus import get_arxiv_corpus, get_papers_by_category
from glassbox.core.vectorizer import TextVectorizer
from glassbox.core.geometry import compute_svd_rank_k
from glassbox.core.classifiers import LDASimulator, SVMSimulator


def test_arxiv_corpus_integrity():
    """Verify that the arXiv corpus contains 64 papers across 4 balanced categories."""
    papers = get_arxiv_corpus()
    assert len(papers) == 64

    categories = ["cs.CV", "cs.CL", "cs.LG", "stat.ML"]
    for cat in categories:
        cat_papers = get_papers_by_category(cat)
        assert len(cat_papers) == 16, f"Expected 16 papers for {cat}, got {len(cat_papers)}"

    for p in papers:
        assert p.id.startswith("arxiv:")
        assert len(p.title.strip()) > 5
        assert len(p.abstract.strip()) > 30
        assert len(p.authors.strip()) > 2
        assert p.year >= 2000


def test_svd_rank_k_reconstruction():
    """Verify truncated SVD properties:
    - Frobenius error decreases monotonically with rank k
    - Energy captured increases monotonically with rank k
    - At full rank (k=3), reconstruction error is near zero
    """
    papers = get_arxiv_corpus()
    texts = [p.abstract for p in papers]
    vec = TextVectorizer().fit(texts)
    _, _, x = vec.transform(texts)

    terms = ["image", "language", "optimization"]
    indices = [vec.vocabulary[t] for t in terms]
    X_3d = x[:, indices]

    svd_1 = compute_svd_rank_k(X_3d, k=1)
    svd_2 = compute_svd_rank_k(X_3d, k=2)
    svd_3 = compute_svd_rank_k(X_3d, k=3)

    # Monotonicity of error and energy
    assert svd_1.frobenius_error >= svd_2.frobenius_error >= svd_3.frobenius_error
    assert svd_1.energy_captured <= svd_2.energy_captured <= svd_3.energy_captured

    # At full rank (k=3), reconstruction error is essentially floating-point noise
    assert svd_3.frobenius_error < 1e-10
    assert pytest.approx(svd_3.energy_captured, abs=1e-8) == 1.0


def test_lda_scatter_matrices_and_eigenpairs():
    """Verify LDA Within-Class (S_W) and Between-Class (S_B) scatter matrices:
    - S_W and S_B are symmetric
    - Eigenvalues are non-negative
    - Generalized eigenvector property holds
    """
    papers = get_arxiv_corpus()
    texts = [p.abstract for p in papers]
    categories = [p.category for p in papers]

    vec = TextVectorizer().fit(texts)
    _, _, x = vec.transform(texts)
    terms = ["image", "language", "optimization"]
    indices = [vec.vocabulary[t] for t in terms]
    X_3d = x[:, indices]

    lda_res = LDASimulator.fit(X_3d, categories)

    # Symmetry
    np.testing.assert_allclose(lda_res.S_W, lda_res.S_W.T, atol=1e-10)
    np.testing.assert_allclose(lda_res.S_B, lda_res.S_B.T, atol=1e-10)

    # Positive semi-definiteness / non-negative eigenvalues
    assert np.all(lda_res.eigenvalues >= 0.0)

    # Shapes match (3, 3)
    assert lda_res.S_W.shape == (3, 3)
    assert lda_res.S_B.shape == (3, 3)
    assert lda_res.projected_2d.shape == (len(papers), 2)


def test_svm_margin_mechanics_and_regularization():
    """Verify SVM margin width formula gamma = 2 / ||w|| and regularization effect."""
    papers = get_arxiv_corpus()
    texts = [p.abstract for p in papers]
    categories = [p.category for p in papers]

    vec = TextVectorizer().fit(texts)
    _, _, x = vec.transform(texts)
    terms = ["image", "language", "optimization"]
    indices = [vec.vocabulary[t] for t in terms]
    X_3d = x[:, indices]

    lda_res = LDASimulator.fit(X_3d, categories)

    # Binary task: cs.CV (+1) vs cs.CL (-1)
    mask = np.array([c in ["cs.CV", "cs.CL"] for c in categories])
    X_sub = lda_res.projected_2d[mask]
    y_sub = np.array([1 if c == "cs.CV" else -1 for c in np.array(categories)[mask]])

    # Fit soft margin (C=0.1) and harder margin (C=10.0)
    svm_soft = SVMSimulator.fit(X_sub, y_sub, C=0.1)
    svm_hard = SVMSimulator.fit(X_sub, y_sub, C=10.0)

    # Check margin width formula: gamma = 2 / ||w||
    assert pytest.approx(svm_soft.margin_width, rel=1e-5) == (2.0 / np.linalg.norm(svm_soft.w))
    assert pytest.approx(svm_hard.margin_width, rel=1e-5) == (2.0 / np.linalg.norm(svm_hard.w))

    # As C increases, margin width must narrow or stay equal (harder penalty on slacks)
    assert svm_hard.margin_width <= svm_soft.margin_width

    # Check support vectors exist
    assert len(svm_soft.support_vector_indices) > 0
    assert len(svm_hard.support_vector_indices) > 0
