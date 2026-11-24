"""Regression cases found during the final scientific audit."""
import numpy as np
import pytest
from sklearn.metrics import silhouette_score
from scipy.linalg import eigh

from glassbox.core.geometry import jacobi_eigenvalue_solver, fit_pca_3d, compute_svd_rank_k
from glassbox.core.classifiers import LDASimulator, SVMSimulator
from glassbox.modules.comparisons import compare_kmeans
from glassbox.utils.math_engine import (
    simulate_kmeans, compute_elbow_and_silhouette, compute_confusion_matrix_and_metrics,
    clean_regex_text, SimpleMLP,
)
from glassbox.utils.visual_components import render_living_diff_text


@pytest.mark.parametrize("scale", [1e-18, 1., 1e18])
def test_jacobi_preserves_eigenpairs_across_units(scale):
    A = np.array([[3., .8, .2], [.8, 2., .3], [.2, .3, 1.]]) * scale
    values, vectors, _ = jacobi_eigenvalue_solver(A)
    np.testing.assert_allclose(values / scale, np.linalg.eigvalsh(A / scale)[::-1], atol=1e-10)
    np.testing.assert_allclose((A @ vectors - vectors * values) / scale, 0., atol=1e-10)


def test_jacobi_accepts_convergence_on_last_allowed_rotation():
    values, vectors, steps = jacobi_eigenvalue_solver(np.array([[2., 1.], [1., 2.]]), max_iter=1)
    np.testing.assert_allclose(values, [3., 1.])
    assert steps == 1


@pytest.mark.parametrize("matrix", [np.ones(3), np.ones((2, 3)), np.array([[np.nan]]), np.zeros((0, 0))])
def test_jacobi_rejects_invalid_inputs(matrix):
    with pytest.raises(ValueError):
        jacobi_eigenvalue_solver(matrix)


def test_centered_svd_snap_is_the_pca_affine_projection():
    X = np.random.default_rng(5).normal(size=(20, 3)) + [12., -7., 9.]
    pca = fit_pca_3d(X, ["a", "b", "c"])
    svd = compute_svd_rank_k(pca.X_centered, 2)
    np.testing.assert_allclose(svd.X_reconstructed + pca.mean_vector, pca.projected_3d, atol=1e-10)
    assert svd.frobenius_error**2 == pytest.approx((len(X)-1)*pca.eigenvalues[-1])


def test_svd_rejects_rank_above_smaller_dimension():
    with pytest.raises(ValueError):
        compute_svd_rank_k(np.ones((2, 3)), 3)


def test_kmeans_integer_input_retains_fractional_centers():
    X = np.array([[0, 0], [1, 0], [10, 0], [11, 0]])
    result = simulate_kmeans(X, k=2, max_iter=30)
    np.testing.assert_allclose(np.sort(result["final_centroids"][:, 0]), [.5, 10.5])
    assert result["converged"]


def test_kmeans_truncated_run_reports_coherent_labels_and_inertia():
    X = np.random.default_rng(17).normal(size=(40, 2))
    result = simulate_kmeans(X, k=4, max_iter=1)
    distances = ((X[:, None] - result["final_centroids"][None])**2).sum(axis=2)
    np.testing.assert_array_equal(result["final_labels"], distances.argmin(axis=1))
    assert result["wcss"] == pytest.approx(distances.min(axis=1).sum())
    assert not result["converged"]
    assert np.all(np.diff([h["wcss"] for h in result["history"]]) <= 1e-12)


def test_silhouette_singleton_matches_reference():
    X = np.array([[0.], [.1], [.2], [10.]])
    labels = simulate_kmeans(X, k=2, max_iter=20)["final_labels"]
    assert 1 in np.unique(labels, return_counts=True)[1]
    score = compute_elbow_and_silhouette(X, [2])["silhouettes"][0]
    assert score == pytest.approx(silhouette_score(X, labels), abs=1e-12)


def test_silhouette_marks_undefined_partitions():
    result = compute_elbow_and_silhouette(np.ones((5, 2)), [2])
    assert np.isnan(result["silhouettes"][0])


def test_kmeans_reference_matches_with_shared_initialization():
    rng = np.random.default_rng(19)
    X = np.vstack([rng.normal([x, y], .1, size=(12, 2)) for x, y in [(0, 0), (4, 0), (0, 4)]])
    scratch, reference, residual, agreement = compare_kmeans(X, 3)
    assert residual < 1e-10
    assert agreement == 1.
    assert scratch["wcss"] == pytest.approx(reference.inertia_)


def test_vertical_svm_boundary_and_both_margins_are_level_sets():
    X = np.array([[-2., -1], [-2, 1], [2, -1], [2, 1]])
    result = SVMSimulator.fit(X, np.array([-1, -1, 1, 1]))
    for x, y, level in [(result.line_x, result.line_boundary, 0),
                         (result.line_gutter_pos_x, result.line_gutter_pos, 1),
                         (result.line_gutter_neg_x, result.line_gutter_neg, -1)]:
        np.testing.assert_allclose(result.w[0]*x + result.w[1]*y + result.b, level, atol=1e-10)


def test_lda_regularized_generalized_eigenproblem_matches_scipy():
    X = np.random.default_rng(8).normal(size=(20, 3))
    X[:, 2] = X[:, 0]  # singular within-class scatter before regularization
    result = LDASimulator.fit(X, ["a"]*10 + ["b"]*10)
    regularized = result.S_W + 1e-4*np.eye(3)
    expected = eigh(result.S_B, regularized, eigvals_only=True)[::-1]
    np.testing.assert_allclose(result.eigenvalues, expected, atol=1e-8)
    np.testing.assert_allclose(result.S_B @ result.eigenvectors,
                               (regularized @ result.eigenvectors)*result.eigenvalues, atol=1e-8)


def test_metrics_reject_unaccounted_predictions():
    with pytest.raises(ValueError):
        compute_confusion_matrix_and_metrics(["a", "b"], ["a"], ["a", "b"])
    with pytest.raises(ValueError):
        compute_confusion_matrix_and_metrics(["a"], ["missing"], ["a", "b"])


def test_preprocessing_diff_marks_entire_formulas_urls_and_escapes_html():
    raw = "See $x+y$ at https://example.org/a and <tag>"
    cleaned, removed = clean_regex_text(raw)
    result = render_living_diff_text(raw, cleaned, removed, ["and"])
    assert 'title="Stripped by Regex">$x+y$</del>' in result
    assert 'title="Stripped by Regex">https://example.org/a</del>' in result
    assert "<tag>" not in result
    assert "&lt;" in result


def test_mlp_backprop_matches_finite_difference_of_regularized_loss():
    X = np.array([[.2, .7], [-.5, .3], [1., -.4]])
    y = np.array([0, 1, 0])
    model = SimpleMLP(hidden_dim=3, num_classes=2)
    model.b1[:] = .8  # keep ReLU away from its nondifferentiable point
    initial = model.W1.copy()
    def loss():
        probs = np.array([model.inspect_sample(x)["probs"] for x in X])
        return -np.log(probs[np.arange(len(y)), y]).mean() + .05*(np.sum(model.W1**2)+np.sum(model.W2**2))
    numerical = np.zeros_like(initial)
    for idx in np.ndindex(initial.shape):
        model.W1[idx] += 1e-6
        plus = loss()
        model.W1[idx] -= 2e-6
        minus = loss()
        model.W1[idx] += 1e-6
        numerical[idx] = (plus-minus)/2e-6
    model.fit(X, y, epochs=1, lr=.01, l2=.1)
    np.testing.assert_allclose((initial-model.W1)/.01, numerical, atol=1e-7)
