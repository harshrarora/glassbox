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
