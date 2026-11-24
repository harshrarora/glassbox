"""Tests for Stage 2 Linguistics Lab & Advanced Math Engine:
- Regex sanitization & tokenization
- Stopword elimination & living diffs
- Porter Stemmer vs. WordNet Lemmatizer divergence
- Vocabulary contraction funnel
- Decision Tree Gini & Information Gain
- KNN neighborhood ball
- Model MRI empirical influence functions
"""

import numpy as np
import pytest
from glassbox.utils.math_engine import (
    clean_regex_text,
    tokenize_text,
    filter_stopwords,
    compare_stemmer_lemmatizer,
    compute_vocabulary_contraction,
    compute_decision_tree_split,
    compute_knn_neighbors,
    compute_influence_functions_2d,
    simulate_kmeans,
    compute_elbow_and_silhouette,
    compute_hac_linkage,
    SimpleMLP,
    compute_confusion_matrix_and_metrics,
    compute_bias_variance_curve,
    verify_first_principles_vs_sklearn,
)


def test_regex_cleaning_and_tokenization():
    raw_sample = "In this paper, we optimize $\\min_w f(w)$ using Adam https://arxiv.org/abs/1412.6980 with 100 iterations!"
    cleaned, removed = clean_regex_text(raw_sample)

    assert "https://" not in cleaned
    assert "\\min_w" not in cleaned
    assert "100" not in cleaned
    assert len(removed) > 0

    tokens = tokenize_text(cleaned)
    assert "paper" in tokens
    assert "optimize" in tokens
    assert "adam" in tokens


def test_stopwords_and_stemmer_lemmatizer():
    tokens = ["studies", "running", "better", "neural", "networks", "the", "which", "is"]
    retained, discarded = filter_stopwords(tokens)

    assert "the" in discarded
    assert "is" in discarded
    assert "which" in discarded
    assert "neural" in retained
    assert "studies" in retained

    comp = compare_stemmer_lemmatizer(retained)
    comp_map = {item["original"]: (item["stem"], item["lemma"]) for item in comp}

    # "studies" -> stem is "studi", lemma is "study"
    assert comp_map["studies"][0] != comp_map["studies"][1]
    assert comp_map["studies"][1] == "study"


def test_vocabulary_contraction_funnel():
    sample_corpus = [
        "Deep convolutional networks achieve state-of-the-art $O(1/k^2)$ accuracy across 100 ImageNet benchmarks.",
        "Self-attention mechanisms in Transformer architectures optimize translation BLEU scores efficiently.",
    ]
    stats = compute_vocabulary_contraction(sample_corpus)

    # Monotonic shrinking
    assert stats["v_raw"] >= stats["v_regex"] >= stats["v_stop"] >= stats["v_lemma"]
    assert stats["total_compression_pct"] > 0


def test_decision_tree_split_math():
    X = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    y = np.array(["A", "A", "A", "B", "B", "B"])

    split = compute_decision_tree_split(X, y, threshold=0.5)
    # Perfect split: left has all A, right has all B
    assert split["info_gain"] > 0.9  # 1 bit of information gain
    assert split["split_gini"] == 0.0  # zero impurity after split


def test_knn_neighborhood_ball():
    X_train = np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1]])
    y_train = np.array(["A", "A", "B", "B"])
    q_pt = np.array([0.05, 0.05])

    knn = compute_knn_neighbors(X_train, y_train, q_pt, k=2)
    assert knn["prediction"] == "A"
    assert knn["indices"] == [0, 1]
    assert knn["radius"] > 0


def test_influence_functions_2d():
    np.random.seed(42)
    X = np.random.randn(20, 2)
    y = np.where(X[:, 0] > 0, 1, -1)

    x_test = np.array([1.5, 0.5])
    y_test = 1

    influences = compute_influence_functions_2d(X, y, x_test, y_test)
    assert len(influences) == 20
    assert not np.isnan(influences).any()


def test_kmeans_simulation_and_elbow():
    # 4 distinct cluster centers
    rng = np.random.RandomState(42)
    c1 = rng.randn(10, 2) + np.array([5, 5])
    c2 = rng.randn(10, 2) + np.array([-5, 5])
    c3 = rng.randn(10, 2) + np.array([5, -5])
    c4 = rng.randn(10, 2) + np.array([-5, -5])
    X = np.vstack([c1, c2, c3, c4])

    km = simulate_kmeans(X, k=4, max_iter=15, random_state=42)
    assert len(km["history"]) > 0
    assert km["final_centroids"].shape == (4, 2)
    assert km["wcss"] < km["history"][0]["wcss"]  # WCSS decreases with iterations

    elbow = compute_elbow_and_silhouette(X, k_range=[2, 3, 4, 5])
    assert len(elbow["wcss"]) == 4
    assert len(elbow["silhouettes"]) == 4
    # K=4 should have highest or near-highest silhouette
    assert max(elbow["silhouettes"]) > 0.4


def test_hac_linkage_matrix():
    X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
    Z = compute_hac_linkage(X, method="ward")
    assert Z.shape == (3, 4)  # n-1 merges
    assert Z[0, 2] < Z[-1, 2]  # merge distances increase monotonically


def test_mlp_forward_pass_and_activations():
    X = np.array([[1.0, 0.5], [-1.0, -0.5], [0.5, -1.0], [-0.5, 1.0]])
    y = np.array([0, 1, 2, 3])

    mlp = SimpleMLP(hidden_dim=4, num_classes=4).fit(X, y, epochs=100)
    act = mlp.inspect_sample(X[0])

    assert act["z1"].shape == (4,)
    assert act["a1"].shape == (4,)
    assert (act["a1"] >= 0).all()  # ReLU non-negativity
    assert act["probs"].shape == (4,)
    assert pytest.approx(float(np.sum(act["probs"])), rel=1e-5) == 1.0  # Softmax sum to 1
    assert 0 <= act["pred_idx"] < 4


def test_confusion_matrix_and_macro_metrics():
    y_true = ["cs.CV", "cs.CV", "cs.CL", "cs.CL", "cs.LG", "stat.ML"]
    y_pred = ["cs.CV", "cs.CL", "cs.CL", "cs.CL", "cs.LG", "stat.ML"]
    labels = ["cs.CV", "cs.CL", "cs.LG", "stat.ML"]

    res = compute_confusion_matrix_and_metrics(y_true, y_pred, labels)
    assert res["confusion_matrix"].shape == (4, 4)
    assert np.sum(res["confusion_matrix"]) == 6
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert res["per_class"]["cs.LG"]["f1"] == 1.0  # Perfect class


def test_bias_variance_curve_overfitting():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 2)
    y = np.array(["A" if x[0] > 0 else "B" for x in X])

    bv = compute_bias_variance_curve(X, y, max_depth_range=[1, 2, 3, 4, 5])
    assert len(bv["train_acc"]) == 5
    assert len(bv["test_acc"]) == 5
    assert bv["train_acc"][-1] >= bv["train_acc"][0]  # Deep tree fits training data better


def test_first_principles_vs_sklearn_audit():
    sample_texts = [
        "Deep convolutional neural networks for image classification and feature maps.",
        "Natural language processing using self attention transformer representations.",
        "Stochastic gradient descent with momentum for deep learning loss minimization.",
        "Convex optimization and proximal gradient methods in statistical learning.",
    ]
    sample_cats = ["cs.CV", "cs.CL", "cs.LG", "stat.ML"]

    audit = verify_first_principles_vs_sklearn(sample_texts, sample_cats)
    assert len(audit) == 4
    for record in audit:
        assert record["passed"], f"Failed verification test: {record['test']} (diff: {record['diff']})"
