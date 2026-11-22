"""Test mathematical foundations:
- Hand-computable vectorization, DF, IDF, L2 normalization
- Scikit-learn numerical agreement within 1e-5
- Sample covariance, Jacobi eigensolver residuals, and orthonormality
- Comparison with Scikit-learn PCA
- Geometric 3D-to-2D projection and residual orthogonality
- Cosine score reconciliation and retrieval edge cases
"""

import math
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from glassbox.core.vectorizer import TextVectorizer, default_tokenizer
from glassbox.core.geometry import (
    compute_sample_covariance,
    jacobi_eigenvalue_solver,
    fit_pca_3d,
    JacobiConvergenceError,
)
from glassbox.core.retrieval import EvidenceRetrieval, QueryStatus
from glassbox.data.fixture import (
    get_original_corpus,
    DEFAULT_TEACHING_CLAIM,
    DEFAULT_PCA_TERMS,
)


def test_hand_computable_counts_df_idf_norm():
    """Verify hand-computable step-by-step numbers on a minimal 3-document corpus."""
    docs = [
        "synaptic plasticity",        # Doc 0
        "synaptic consolidation",     # Doc 1
        "memory consolidation",       # Doc 2
    ]
    # Tokens sorted: ['consolidation', 'memory', 'plasticity', 'synaptic']
    # N = 3
    # DF:
    #   'consolidation': in Doc 1, Doc 2 -> DF = 2
    #   'memory': in Doc 2 -> DF = 1
    #   'plasticity': in Doc 0 -> DF = 1
    #   'synaptic': in Doc 0, Doc 1 -> DF = 2
    vec = TextVectorizer().fit(docs)
    assert vec.feature_names == ["consolidation", "memory", "plasticity", "synaptic"]
    assert np.array_equal(vec.doc_frequencies, np.array([2, 1, 1, 2]))

    # Hand-computed idf(t) = ln((1 + N)/(1 + DF)) + 1
    # For DF = 2: ln((1 + 3)/(1 + 2)) + 1 = ln(4/3) + 1 = ln(1.33333333) + 1 = 0.287682 + 1 = 1.287682
    # For DF = 1: ln((1 + 3)/(1 + 1)) + 1 = ln(4/2) + 1 = ln(2) + 1 = 0.693147 + 1 = 1.693147
    expected_idf_df2 = math.log(4.0 / 3.0) + 1.0
    expected_idf_df1 = math.log(4.0 / 2.0) + 1.0

    assert pytest.approx(vec.idf_weights[0], rel=1e-6) == expected_idf_df2
    assert pytest.approx(vec.idf_weights[1], rel=1e-6) == expected_idf_df1
    assert pytest.approx(vec.idf_weights[2], rel=1e-6) == expected_idf_df1
    assert pytest.approx(vec.idf_weights[3], rel=1e-6) == expected_idf_df2

    # Transform docs
    c, z, x = vec.transform(docs)

    # Doc 0: plasticity=1, synaptic=1
    # z(plasticity) = 1 * expected_idf_df1
    # z(synaptic) = 1 * expected_idf_df2
    # norm = sqrt(z_plast^2 + z_syn^2)
    expected_z0_plast = expected_idf_df1
    expected_z0_syn = expected_idf_df2
    expected_norm0 = math.sqrt(expected_z0_plast**2 + expected_z0_syn**2)
    expected_x0_plast = expected_z0_plast / expected_norm0
    expected_x0_syn = expected_z0_syn / expected_norm0

    assert pytest.approx(z[0, 2], rel=1e-6) == expected_z0_plast
    assert pytest.approx(z[0, 3], rel=1e-6) == expected_z0_syn
    assert pytest.approx(x[0, 2], rel=1e-6) == expected_x0_plast
    assert pytest.approx(x[0, 3], rel=1e-6) == expected_x0_syn

    # Check inspection dictionary
    info = vec.inspect_cell(docs[0], "plasticity", c, z, x, 0)
    assert info["raw_count"] == 1
    assert info["doc_token_count"] == 2
    assert pytest.approx(info["relative_freq"], rel=1e-6) == 0.5
    assert info["DF"] == 1
    assert info["N"] == 3
    assert pytest.approx(info["idf"], rel=1e-6) == expected_idf_df1
    assert pytest.approx(info["normalized_tfidf"], rel=1e-6) == expected_x0_plast


def test_tfidf_agreement_with_sklearn():
    """Verify exact numerical agreement with scikit-learn's TfidfVectorizer within 1e-5."""
    corpus = [d.abstract for d in get_original_corpus()]

    # GlassBox scratch vectorizer
    scratch_vec = TextVectorizer().fit(corpus)
    c_scratch, z_scratch, x_scratch = scratch_vec.transform(corpus)

    # Scikit-learn reference implementation with identical settings:
    # token_pattern matches (?u)\b\w+\b, smooth_idf=True, norm='l2', use_idf=True
    sk_vec = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        smooth_idf=True,
        norm="l2",
        use_idf=True,
    )
    x_sklearn = sk_vec.fit_transform(corpus).toarray()

    # Compare vocabulary length and terms
    assert scratch_vec.feature_names == sk_vec.get_feature_names_out().tolist()

    # Compare IDF weights
    np.testing.assert_allclose(scratch_vec.idf_weights, sk_vec.idf_, atol=1e-5)

    # Compare L2-normalized TF-IDF matrix
    np.testing.assert_allclose(x_scratch, x_sklearn, atol=1e-5)


def test_covariance_calculation():
    """Verify sample covariance calculation matches NumPy's np.cov."""
    np.random.seed(42)
    X = np.random.randn(10, 3)
    mean_vec = np.mean(X, axis=0)
    Xc = X - mean_vec

    cov_scratch = compute_sample_covariance(Xc)
    cov_numpy = np.cov(X, rowvar=False, bias=False)

    np.testing.assert_allclose(cov_scratch, cov_numpy, atol=1e-12)


def test_jacobi_eigenvalues_and_orthonormality():
    """Verify symmetric Jacobi eigensolver:
    - Eigenpair residual ||Sigma * v - lambda * v|| < 1e-10
    - Orthonormality V^T @ V = I
    - Decreasing order of eigenvalues
    """
    # Create arbitrary symmetric 3x3 matrix
    np.random.seed(123)
    M = np.random.randn(3, 3)
    Sigma = M.T @ M + 0.1 * np.eye(3)

    eigenvalues, eigenvectors, n_iter = jacobi_eigenvalue_solver(Sigma, tol=1e-12)

    # Check orthonormality
    I_approx = eigenvectors.T @ eigenvectors
    np.testing.assert_allclose(I_approx, np.eye(3), atol=1e-10)

    # Check decreasing order
    assert eigenvalues[0] >= eigenvalues[1] >= eigenvalues[2]

    # Check eigenpair residuals: Sigma @ v_i = lambda_i * v_i
    for i in range(3):
        v_i = eigenvectors[:, i]
        lam_i = eigenvalues[i]
        residual = Sigma @ v_i - lam_i * v_i
        np.testing.assert_allclose(residual, np.zeros(3), atol=1e-10)


def test_pca_comparison_with_sklearn():
    """Compare 3D PCA results with scikit-learn PCA."""
    corpus = [d.abstract for d in get_original_corpus()]
    scratch_vec = TextVectorizer().fit(corpus)
    _, _, x_norm = scratch_vec.transform(corpus)

    # Extract 3 selected terms
    pca_indices = [scratch_vec.vocabulary[t] for t in DEFAULT_PCA_TERMS]
    X_3d = x_norm[:, pca_indices]

    # Fit GlassBox PCA
    result = fit_pca_3d(X_3d, DEFAULT_PCA_TERMS)

    # Fit Scikit-learn PCA
    sk_pca = PCA(n_components=3)
    sk_pca.fit(X_3d)

    # Compare eigenvalues (sample variance along components)
    # Sklearn's explained_variance_ holds eigenvalues of sample covariance
    np.testing.assert_allclose(result.eigenvalues, sk_pca.explained_variance_, atol=1e-5)

    # Compare explained variance ratios
    np.testing.assert_allclose(
        result.explained_variance_ratio, sk_pca.explained_variance_ratio_, atol=1e-5
    )

    # Compare eigenvectors (loadings) up to sign:
    # Sklearn components_ has eigenvectors as rows; result.eigenvectors has them as columns
    for i in range(3):
        v_scratch = result.eigenvectors[:, i]
        v_sklearn = sk_pca.components_[i, :]
        # Dot product of unit vectors should be +1 or -1
        dot = abs(np.dot(v_scratch, v_sklearn))
        assert pytest.approx(dot, abs=1e-5) == 1.0


def test_geometric_projection_and_residuals():
    """Verify orthogonal projection properties:
    - scores = Xc @ V2
    - projected_3d = scores @ V2^T + mu
    - Residual vectors are strictly orthogonal to PC1 and PC2
    """
    corpus = [d.abstract for d in get_original_corpus()]
    scratch_vec = TextVectorizer().fit(corpus)
    _, _, x_norm = scratch_vec.transform(corpus)
    pca_indices = [scratch_vec.vocabulary[t] for t in DEFAULT_PCA_TERMS]
    X_3d = x_norm[:, pca_indices]

    result = fit_pca_3d(X_3d, DEFAULT_PCA_TERMS)

    v1 = result.eigenvectors[:, 0]
    v2 = result.eigenvectors[:, 1]

    for i in range(len(corpus)):
        res_i = result.residuals_3d[i]
        # Check orthogonality: res_i . v1 = 0 and res_i . v2 = 0
        assert abs(np.dot(res_i, v1)) < 1e-10
        assert abs(np.dot(res_i, v2)) < 1e-10

        # Check reconstruction
        recon = result.projected_3d[i]
        orig = X_3d[i]
        np.testing.assert_allclose(orig, recon + res_i, atol=1e-10)


def test_cosine_scoring_and_contribution_reconciliation():
    """Verify that per-term contributions sum to the total cosine similarity score."""
    corpus_docs = get_original_corpus()
    corpus = [d.abstract for d in corpus_docs]
    ids = [d.doc_id for d in corpus_docs]
    titles = [d.title for d in corpus_docs]
    annotations = [d.annotation for d in corpus_docs]

    vec = TextVectorizer().fit(corpus)
    c, z, x = vec.transform(corpus)
    retrieval = EvidenceRetrieval(vec)

    status, msg, results = retrieval.search(
        DEFAULT_TEACHING_CLAIM, ids, titles, corpus, annotations, x, c
    )
    assert status == QueryStatus.VALID
    assert len(results) == len(corpus)

    for r in results:
        # Cosine score must equal sum of matching term contributions
        term_sum = sum(tc.contribution for tc in r.matching_terms)
        assert pytest.approx(r.score, abs=1e-12) == term_sum


def test_retrieval_edge_cases():
    """Verify handling of blank query, zero vocabulary overlap, and stable tie-breaking."""
    corpus_docs = get_original_corpus()
    corpus = [d.abstract for d in corpus_docs]
    ids = [d.doc_id for d in corpus_docs]
    titles = [d.title for d in corpus_docs]
    annotations = [d.annotation for d in corpus_docs]

    vec = TextVectorizer().fit(corpus)
    c, z, x = vec.transform(corpus)
    retrieval = EvidenceRetrieval(vec)

    # 1. Blank claim
    status_blank, msg_blank, res_blank = retrieval.search(
        "   ", ids, titles, corpus, annotations, x, c
    )
    assert status_blank == QueryStatus.BLANK
    assert len(res_blank) == 0

    # 2. Zero vocabulary overlap
    status_none, msg_none, res_none = retrieval.search(
        "quantum astrophysics telescope observatory galaxy",
        ids,
        titles,
        corpus,
        annotations,
        x,
        c,
    )
    assert status_none == QueryStatus.NO_VOCAB_OVERLAP
    assert msg_none == "This claim shares no terms with the current vocabulary."
    assert len(res_none) == 0

    # 3. Deterministic tie-breaking
    # Query matching DOC-07 has score 0.0 with some papers; ties must sort by doc_id ascending
    status_valid, msg_valid, res_valid = retrieval.search(
        "paired-associate quiescent", ids, titles, corpus, annotations, x, c
    )
    zero_score_docs = [r.doc_id for r in res_valid if r.score == 0.0]
    # Check that zero-score doc IDs are in strictly ascending alphabetical order
    assert zero_score_docs == sorted(zero_score_docs)
