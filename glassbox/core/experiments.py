"""Reproducible experiments, independent of Streamlit and presentation state."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from glassbox.core.classifiers import LDASimulator
from glassbox.core.geometry import jacobi_eigenvalue_solver
from glassbox.core.retrieval import EvidenceRetrieval
from glassbox.core.vectorizer import TextVectorizer
from glassbox.data.fixture import (
    DEFAULT_TEACHING_CLAIM, get_original_corpus, get_teaching_fixture,
)


def ranking_experiment(refresh_idf=True, include_incoming=False):
    """Change statistics and candidate eligibility independently; freeze vocabulary."""
    original = get_original_corpus()
    expanded = get_teaching_fixture()
    vec = TextVectorizer().fit([d.abstract for d in original])
    if refresh_idf:
        counts = vec.transform_raw_counts([d.abstract for d in expanded])
        vec.n_documents = len(expanded)
        vec.doc_frequencies = np.count_nonzero(counts, axis=0)
        vec.idf_weights = np.log((1 + len(expanded)) / (1 + vec.doc_frequencies)) + 1
    candidates = expanded if include_incoming else original
    texts = [d.abstract for d in candidates]
    counts, _, normalized = vec.transform(texts)
    _, _, results = EvidenceRetrieval(vec).search(
        DEFAULT_TEACHING_CLAIM, [d.doc_id for d in candidates],
        [d.title for d in candidates], texts, [d.annotation for d in candidates],
        normalized, counts,
    )
    return vec, results


def text_projection_split(texts, labels, train_idx, test_idx,
                          terms=("image", "language", "optimization"), leak=False, analyzer=None):
    """Fit vocabulary, IDF and LDA on training rows unless explicitly leaking.

    Missing teaching terms become zero columns, never arbitrary vocabulary columns.
    Returns provenance so a test can verify which rows were used to fit.
    """
    fit_idx = np.arange(len(texts)) if leak else np.asarray(train_idx)
    vec = (TextVectorizer(tokenizer=analyzer) if analyzer else TextVectorizer()).fit([texts[i] for i in fit_idx])
    X = vec.transform(texts)[2]
    terms = [t if t in vec.vocabulary else next(iter(vec.tokenizer(t)), "") for t in terms]
    selected = np.column_stack([
        X[:, vec.vocabulary[t]] if t in vec.vocabulary else np.zeros(len(texts))
        for t in terms
    ])
    lda = LDASimulator.fit(selected[fit_idx], [labels[i] for i in fit_idx])
    return {
        "train": LDASimulator.transform(selected[train_idx], lda),
        "test": LDASimulator.transform(selected[test_idx], lda),
        "fit_indices": fit_idx, "vectorizer": vec, "lda": lda,
    }


def leakage_experiment(seed=42, source="noise"):
    """Paired protocols on exactly the same split. No forced improvement from leakage."""
    if source not in {"noise", "papers"}:
        raise ValueError("source must be noise or papers")
    if source == "noise":
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(120, 40))
        y = np.tile(np.arange(4), 30)
        rng.shuffle(y)  # Labels are independent of all features.
    else:
        from glassbox.data.arxiv_corpus import get_arxiv_corpus
        papers = get_arxiv_corpus()
        texts = [p.abstract for p in papers]
        y = np.array([p.category for p in papers])
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y,
    )
    output = {"train_indices": train, "test_indices": test, "seed": seed,
              "source": source, "depths": list(range(1, 13))}
    for name, leak in [("honest", False), ("leaked", True)]:
        if source == "papers":
            projection = text_projection_split(texts, y, train, test, leak=leak)
            X_train, X_test = projection["train"], projection["test"]
        else:
            fit = np.arange(len(y)) if leak else train
            lda = LDASimulator.fit(X[fit], y[fit].tolist())
            X_train = LDASimulator.transform(X[train], lda)
            X_test = LDASimulator.transform(X[test], lda)
        train_scores, test_scores = [], []
        for depth in output["depths"]:
            model = DecisionTreeClassifier(max_depth=depth, random_state=seed)
            model.fit(X_train, y[train])
            train_scores.append(float(model.score(X_train, y[train])))
            test_scores.append(float(model.score(X_test, y[test])))
        output[name] = {"train": train_scores, "test": test_scores}
    return output


def rotating_experiment(gap=0.002, perturbation=0.01):
    """An exact covariance experiment with a nearly tied leading eigenvalue pair.

    A small x-y coupling rotates axes within the leading plane. A smaller x-z
    coupling also perturbs the plane, avoiding a tautologically unchanged plane.
    Six symmetric observations realize each covariance exactly (n-1 convention).
    """
    if not 0 < gap <= 1 or not 0 <= perturbation <= 0.1:
        raise ValueError("gap must be in (0, 1]; perturbation in [0, 0.1]")
    baseline = np.diag([2.0 + gap, 2.0, 0.15])
    delta = perturbation * np.array([[0., 1., .2], [1., 0., 0.], [.2, 0., 0.]])
    changed = baseline + delta
    vals0, V0, _ = jacobi_eigenvalue_solver(baseline)
    vals1, V1, _ = jacobi_eigenvalue_solver(changed)
    cosine = np.clip(abs(V0[:, 0] @ V1[:, 0]), 0., 1.)
    P0, P1 = V0[:, :2] @ V0[:, :2].T, V1[:, :2] @ V1[:, :2].T
    def observations(C):
        root = np.linalg.cholesky(C)
        return np.sqrt(5 / 2) * np.vstack([root.T, -root.T])
    return {
        "baseline": baseline, "changed": changed, "delta": delta,
        "values_before": vals0, "values_after": vals1,
        "axes_before": V0, "axes_after": V1,
        "axis_angle": float(np.degrees(np.arccos(cosine))),
        "projector_distance": float(np.linalg.norm(P1 - P0, ord=2)),
        "relative_perturbation": float(np.linalg.norm(delta, ord=2) / np.linalg.norm(baseline, ord=2)),
        "points_before": observations(baseline), "points_after": observations(changed),
        "outer_gap": float(vals1[1] - vals1[2]),
    }


def mri_dataset(test_idx=0):
    """Train-only TF-IDF on two fixed, interpretable term axes; no supervised projection."""
    from glassbox.data.arxiv_corpus import get_arxiv_corpus
    papers = [p for p in get_arxiv_corpus() if p.category in ("cs.CV", "cs.CL")]
    train_papers = [p for i, p in enumerate(papers) if i != test_idx]
    vec = TextVectorizer().fit([p.abstract for p in train_papers])
    X = vec.transform([p.abstract for p in papers])[2]
    terms = ("image", "language")
    features = np.column_stack([
        X[:, vec.vocabulary[t]] if t in vec.vocabulary else np.zeros(len(papers))
        for t in terms
    ])
    # Fixed scaling is specified, not estimated using the held-out observation.
    features *= 10.0
    y = np.array([1 if p.category == "cs.CV" else -1 for p in papers])
    mask = np.arange(len(papers)) != test_idx
    return {"X": features[mask], "y": y[mask], "x_test": features[test_idx],
            "y_test": int(y[test_idx]), "papers": train_papers,
            "test_paper": papers[test_idx], "vectorizer": vec}
