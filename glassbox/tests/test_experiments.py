"""Scientific contracts for the four investigations, beyond UI appearance."""
import numpy as np
import pytest

from glassbox.core.experiments import (
    ranking_experiment, text_projection_split, leakage_experiment,
    rotating_experiment, mri_dataset,
)
from glassbox.modules.investigations import boundary_trace
from glassbox.utils.math_engine import (
    fit_logistic_regression_2d, compute_influence_functions_2d,
    compute_retrained_counterfactual,
)


@pytest.mark.parametrize("refresh,include", [(False, False), (False, True), (True, False), (True, True)])
def test_ranking_interventions_preserve_text_and_reconcile_scores(refresh, include):
    base_vec, before = ranking_experiment(False, False)
    vec, after = ranking_experiment(refresh, include)
    assert vec.vocabulary == base_vec.vocabulary
    assert len(after) == (10 if include else 7)
    old = {r.doc_id: r for r in before}
    for r in after:
        assert r.score == pytest.approx(r.sum_contributions, abs=1e-12)
        if r.doc_id in old:
            assert r.abstract == old[r.doc_id].abstract
    if refresh and not include:
        assert next(r.rank for r in after if r.doc_id == "DOC-03") == 6


def test_honest_representation_cannot_see_held_out_text_or_labels():
    texts = ["image bright", "image visual", "language word", "language grammar", "privateonly image"]
    labels = ["a", "a", "b", "b", "a"]
    train, test = np.arange(4), np.array([4])
    a = text_projection_split(texts, labels, train, test)
    labels[-1] = "b"
    texts[-1] = "unseenonly language"
    b = text_projection_split(texts, labels, train, test)
    assert "privateonly" not in a["vectorizer"].vocabulary
    assert "unseenonly" not in b["vectorizer"].vocabulary
    np.testing.assert_allclose(a["train"], b["train"])
    np.testing.assert_allclose(a["lda"].S_B, b["lda"].S_B)
    np.testing.assert_allclose(a["vectorizer"].idf_weights, b["vectorizer"].idf_weights)


@pytest.mark.parametrize("source", ["noise", "papers"])
def test_paired_leakage_experiment_is_reproducible(source):
    a, b = leakage_experiment(42, source), leakage_experiment(42, source)
    assert not set(a["train_indices"]) & set(a["test_indices"])
    for protocol in ("honest", "leaked"):
        assert a[protocol] == b[protocol]
        assert all(0 <= v <= 1 for v in a[protocol]["test"])
    # Do not require leakage to improve the score: that is not a mathematical law.


def test_covariance_fixture_realizes_both_matrices_and_checks_eigenpairs():
    r = rotating_experiment()
    np.testing.assert_allclose(np.cov(r["points_before"], rowvar=False), r["baseline"], atol=1e-12)
    np.testing.assert_allclose(np.cov(r["points_after"], rowvar=False), r["changed"], atol=1e-12)
    np.testing.assert_allclose(r["changed"] @ r["axes_after"], r["axes_after"] * r["values_after"], atol=1e-10)
    assert r["axis_angle"] > 35
    assert r["projector_distance"] < .002
    assert r["axis_angle"] > rotating_experiment(gap=1.)["axis_angle"]


def test_zero_perturbation_does_not_invent_instability():
    r = rotating_experiment(perturbation=0)
    assert r["axis_angle"] == 0
    assert r["projector_distance"] == 0


def test_influence_matches_finite_difference_of_actual_upweighting():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 2))
    y = np.where(X[:, 0] + .3 * X[:, 1] > 0, 1, -1)
    x_test, y_test = np.array([.6, -.4]), 1
    result = compute_influence_functions_2d(X, y, x_test, y_test)
    eps = 1e-5
    for index in [0, 7, 19]:
        losses = []
        for direction in [-1, 1]:
            weights = np.ones(len(X))
            weights[index] += direction * eps * len(X)
            theta, _, _ = fit_logistic_regression_2d(X, y, sample_weights=weights)
            z = float(np.append(x_test, 1.) @ theta)
            losses.append(np.logaddexp(0., z) - z)
        measured = (losses[1] - losses[0]) / (2 * eps)
        assert measured == pytest.approx(result["influences"][index], abs=1e-7)


def test_counterfactual_retrains_same_objective_and_is_stationary():
    data = mri_dataset(0)
    cf = compute_retrained_counterfactual(data["X"], data["y"], data["x_test"], data["y_test"], 3)
    mask = np.arange(len(data["X"])) != 3
    augmented = np.column_stack([data["X"][mask], np.ones(mask.sum())])
    z = augmented @ cf["theta_cf"]
    probability = np.exp(-np.logaddexp(0., -z))
    gradient = augmented.T @ (probability - (data["y"][mask] == 1)) / mask.sum() + .1 * cf["theta_cf"]
    assert np.linalg.norm(gradient) < 1e-8
    assert cf["actual_delta_loss"] == pytest.approx(cf["test_loss_cf"] - cf["test_loss_orig"])
    assert np.isfinite(cf["pred_delta_loss_normalized"])


def test_vertical_boundary_is_a_vertical_level_set():
    theta = np.array([2., 0., -1.])
    trace = boundary_trace(theta, np.array([[0., 0.], [1., 1.]]), "boundary", "black")
    np.testing.assert_allclose(trace.x, .5)
    np.testing.assert_allclose(theta[0] * np.asarray(trace.x) + theta[1] * np.asarray(trace.y) + theta[2], 0.)


def test_invalid_logistic_input_is_not_silently_accepted():
    with pytest.raises(ValueError):
        fit_logistic_regression_2d(np.ones((3, 2)), np.array([-1, 0, 1]))
    with pytest.raises(ValueError):
        fit_logistic_regression_2d(np.ones((3, 2)), np.array([-1, 1, 1]), l2_reg=0.)


@pytest.mark.parametrize("morphology,bigrams", [("None",False),("Lemma",True),("Stem",True)])
def test_live_tfidf_comparison_aligns_analyzer_and_vocabulary(morphology,bigrams):
    from glassbox.modules.comparisons import compare_tfidf
    config={"regex":True,"stopwords":True,"morphology":morphology,"bigrams":bigrams}
    vec,counts,scratch,reference=compare_tfidf(["Studies of running models 123", "Models running quickly", "New studies of matrices"],config)
    np.testing.assert_allclose(scratch,reference,atol=1e-12)
    assert counts.shape==scratch.shape
    assert any(" " in term for term in vec.feature_names) == bigrams


def test_selected_coordinates_follow_the_analyzer_in_honest_projection():
    analyzer=lambda text: ["processed_"+t for t in text.split()]
    result=text_projection_split(["image image", "image language", "language language", "language optimization", "image"],
                                 ["a","a","b","b","a"],np.arange(4),np.array([4]),analyzer=analyzer)
    assert np.linalg.norm(result["train"]) > 0
    assert "processed_image" in result["vectorizer"].vocabulary
