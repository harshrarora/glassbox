"""Pure mathematical and linguistic processing engine from first principles.

Implements:
1. Regex sanitization, tokenization, stopword filtering, and Stemming vs Lemmatization
2. Count, TF-IDF, N-gram expansion, and sparsity calculations
3. Sample covariance, Jacobi eigensolver, PCA, SVD rank-k reconstruction
4. Multiclass LDA scatter matrices (S_W and S_B)
5. Supervised classifier geometries (SVM margins, Decision Tree Gini splits, KNN, Naive Bayes)
6. Model MRI Influence Functions (Koh & Liang, ICML 2017)
"""

import math
import re
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np

# NLTK imports with robust pure-Python fallbacks
try:
    from nltk.corpus import stopwords as nltk_stopwords, wordnet
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_STOPWORDS = set(nltk_stopwords.words("english"))
    STEMMER = PorterStemmer()
    LEMMATIZER = WordNetLemmatizer()
    wordnet.ensure_loaded()
    MORPHOLOGY_BACKEND = "NLTK Porter stemmer / WordNet noun lemmatizer"
except Exception:
    NLTK_STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
        "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for",
        "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
        "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
        "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
        "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
        "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
        "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
        "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves"
    }
    class FallbackStemmer:
        def stem(self, w: str) -> str:
            w = w.lower()
            if w.endswith("ing") and len(w) > 4: return w[:-3]
            if w.endswith("ies") and len(w) > 4: return w[:-3] + "i"
            if w.endswith("ed") and len(w) > 3: return w[:-2]
            if w.endswith("s") and not w.endswith("ss") and len(w) > 2: return w[:-1]
            return w
    class FallbackLemmatizer:
        def lemmatize(self, w: str, pos: str = "n") -> str:
            w = w.lower()
            irreg = {"better": "good", "running": "run", "studies": "study", "matrices": "matrix", "mice": "mouse"}
            if w in irreg: return irreg[w]
            if w.endswith("ies") and len(w) > 4: return w[:-3] + "y"
            if w.endswith("s") and not w.endswith("ss") and len(w) > 2: return w[:-1]
            return w
    STEMMER = FallbackStemmer()
    LEMMATIZER = FallbackLemmatizer()
    MORPHOLOGY_BACKEND = "Simplified offline suffix rules / small lemma dictionary"


# -----------------------------------------------------------------------------
# 1. Linguistic & Preprocessing Engine
# -----------------------------------------------------------------------------
def clean_regex_text(text: str) -> Tuple[str, List[str]]:
    """Strips LaTeX math, formulas, URLs, punctuation, and digits.
    Returns: (cleaned_text, list_of_removed_segments)
    """
    removed_segments = []

    # 1. Find and strip LaTeX math formulas like $...$ or \( ... \)
    latex_matches = re.findall(r"\$.*?\$|\\\(.*?\\\)", text)
    removed_segments.extend(latex_matches)
    text_no_latex = re.sub(r"\$.*?\$|\\\(.*?\\\)", " ", text)

    # 2. Find and strip URLs
    url_matches = re.findall(r"https?://\S+|www\.\S+", text_no_latex)
    removed_segments.extend(url_matches)
    text_no_url = re.sub(r"https?://\S+|www\.\S+", " ", text_no_latex)

    # 3. Find and strip punctuation, symbols, and digits
    punct_matches = re.findall(r"[^a-zA-Z\s]", text_no_url)
    removed_segments.extend([p for p in punct_matches if p.strip()])
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text_no_url)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned, removed_segments


def tokenize_text(text: str) -> List[str]:
    """Splits cleaned text into lowercase word tokens."""
    if not text:
        return []
    return [t.lower() for t in re.findall(r"\b[a-zA-Z]{2,}\b", text)]


def filter_stopwords(tokens: List[str]) -> Tuple[List[str], List[str]]:
    """Filters English stopwords. Returns (retained_tokens, discarded_stopwords)."""
    retained = []
    discarded = []
    for t in tokens:
        if t in NLTK_STOPWORDS:
            discarded.append(t)
        else:
            retained.append(t)
    return retained, discarded


def compare_stemmer_lemmatizer(tokens: List[str]) -> List[Dict[str, Any]]:
    """Runs Porter Stemmer vs WordNet Lemmatizer on a unique token list."""
    results = []
    unique_tokens = sorted(list(set(tokens)))
    for token in unique_tokens:
        stem = STEMMER.stem(token)
        lemma = LEMMATIZER.lemmatize(token)
        results.append({
            "original": token,
            "stem": stem,
            "lemma": lemma,
            "differs": (stem != lemma),
        })
    return results


def compute_vocabulary_contraction(corpus_texts: List[str]) -> Dict[str, Any]:
    """Computes exact unique vocabulary contraction steps across the whole corpus:
    Raw Words -> Regex Cleaned -> Stopword Filtered -> Lemmatized
    """
    raw_words = set()
    for text in corpus_texts:
        raw_words.update(re.findall(r"\b\w+\b", text))

    regex_tokens = set()
    for text in corpus_texts:
        cleaned, _ = clean_regex_text(text)
        regex_tokens.update(tokenize_text(cleaned))
    v_regex = len(regex_tokens)
    v_raw = max(len(raw_words), v_regex)

    stop_tokens = set()
    for t in regex_tokens:
        if t not in NLTK_STOPWORDS:
            stop_tokens.add(t)
    v_stop = len(stop_tokens)

    lemmas = set()
    stems = set()
    for t in stop_tokens:
        lemmas.add(LEMMATIZER.lemmatize(t))
        stems.add(STEMMER.stem(t))
    v_lemma = len(lemmas)
    v_stem = len(stems)

    return {
        "v_raw": v_raw,
        "v_regex": v_regex,
        "v_stop": v_stop,
        "v_stem": v_stem,
        "v_lemma": v_lemma,
        "regex_reduction_pct": ((v_raw - v_regex) / v_raw * 100) if v_raw else 0,
        "stop_reduction_pct": ((v_regex - v_stop) / v_regex * 100) if v_regex else 0,
        "lemma_reduction_pct": ((v_stop - v_lemma) / v_stop * 100) if v_stop else 0,
        "total_compression_pct": ((v_raw - v_lemma) / v_raw * 100) if v_raw else 0,
    }


# -----------------------------------------------------------------------------
# 2. Vectorization, TF-IDF, N-Grams & Sparsity
# -----------------------------------------------------------------------------
def build_count_matrix(documents: List[str], ngram_range: Tuple[int, int] = (1, 1)) -> Tuple[np.ndarray, List[str]]:
    """Builds raw count matrix C with optional n-grams (1=unigram, 2=bigram)."""
    tokenized_docs = []
    vocab_set = set()

    for doc in documents:
        cleaned, _ = clean_regex_text(doc)
        tokens, _ = filter_stopwords(tokenize_text(cleaned))
        lemmatized = [LEMMATIZER.lemmatize(t) for t in tokens]

        doc_ngrams = []
        if ngram_range[0] <= 1 and ngram_range[1] >= 1:
            doc_ngrams.extend(lemmatized)
        if ngram_range[1] >= 2:
            bigrams = [f"{lemmatized[i]} {lemmatized[i+1]}" for i in range(len(lemmatized) - 1)]
            doc_ngrams.extend(bigrams)

        tokenized_docs.append(doc_ngrams)
        vocab_set.update(doc_ngrams)

    vocabulary = sorted(list(vocab_set))
    term_to_idx = {t: idx for idx, t in enumerate(vocabulary)}

    C = np.zeros((len(documents), len(vocabulary)), dtype=np.float64)
    for i, doc_terms in enumerate(tokenized_docs):
        for t in doc_terms:
            if t in term_to_idx:
                C[i, term_to_idx[t]] += 1.0

    return C, vocabulary


def compute_tfidf_from_counts(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Computes:
      DF(t): document frequencies
      IDF(t) = ln((1 + N)/(1 + DF(t))) + 1
      Z = C * IDF (unnormalized)
      X = Z / ||Z||_2 (L2 normalized)
      sparsity = fraction of zero entries in matrix
    """
    n_docs, v_size = C.shape
    df = np.sum(C > 0, axis=0)

    # Smoothed IDF formula matching textbook and sklearn
    idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

    # Unnormalized TF-IDF
    Z = C * idf[np.newaxis, :]

    # Row-wise L2 norm
    norms = np.linalg.norm(Z, ord=2, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        X = np.where(norms > 0, Z / norms, 0.0)

    sparsity = 1.0 - (np.count_nonzero(C) / float(n_docs * v_size)) if (n_docs * v_size) > 0 else 1.0
    return df, idf, X, sparsity


# -----------------------------------------------------------------------------
# 3. Supervised Classifiers & Decision Geometries
# -----------------------------------------------------------------------------
def compute_decision_tree_split(X_1d: np.ndarray, y: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Calculates Gini Impurity and Entropy Information Gain for an axis-aligned split."""
    def entropy(labels):
        if len(labels) == 0: return 0.0
        classes, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def gini(labels):
        if len(labels) == 0: return 0.0
        classes, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return 1.0 - np.sum(probs ** 2)

    parent_h = entropy(y)
    parent_g = gini(y)

    left_mask = (X_1d <= threshold)
    right_mask = ~left_mask

    y_l, y_r = y[left_mask], y[right_mask]
    n_total = len(y)
    n_l, n_r = len(y_l), len(y_r)

    if n_total > 0 and n_l > 0 and n_r > 0:
        weighted_h = (n_l / n_total) * entropy(y_l) + (n_r / n_total) * entropy(y_r)
        info_gain = parent_h - weighted_h
        weighted_g = (n_l / n_total) * gini(y_l) + (n_r / n_total) * gini(y_r)
    else:
        info_gain = 0.0
        weighted_g = parent_g

    return {
        "threshold": threshold,
        "parent_entropy": parent_h,
        "parent_gini": parent_g,
        "left_count": n_l,
        "right_count": n_r,
        "info_gain": info_gain,
        "split_gini": weighted_g,
    }


def compute_knn_neighbors(X_train: np.ndarray, y_train: np.ndarray, query_point: np.ndarray, k: int) -> Dict[str, Any]:
    """Computes K-Nearest Neighbors, distances, and neighborhood majority vote."""
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(X_train):
        raise ValueError("k must be between one and the training sample count")
    distances = np.linalg.norm(X_train - query_point, axis=1)
    k_indices = np.argsort(distances, kind="stable")[:k]

    k_distances = distances[k_indices]
    k_labels = y_train[k_indices]

    unique_labels, counts = np.unique(k_labels, return_counts=True)
    winner_idx = np.argmax(counts)
    prediction = unique_labels[winner_idx]

    radius = float(k_distances[-1]) if len(k_distances) > 0 else 1.0

    return {
        "k": k,
        "indices": k_indices.tolist(),
        "distances": k_distances.tolist(),
        "labels": k_labels.tolist(),
        "prediction": prediction,
        "radius": radius,
        "vote_distribution": dict(zip(unique_labels.tolist(), counts.tolist())),
    }


def compute_naive_bayes_posteriors(
    term_counts: np.ndarray, y: np.ndarray, query_tokens: List[int], classes: List[str], alpha: float = 1.0
) -> Dict[str, Any]:
    """Calculates Naive Bayes class priors and smoothed feature likelihoods."""
    n_docs, v_size = term_counts.shape
    priors = {}
    posteriors = {}

    for c in classes:
        c_mask = (y == c)
        n_c = np.sum(c_mask)
        priors[c] = n_c / float(n_docs)

        # Class word counts with Laplace smoothing
        c_words = np.sum(term_counts[c_mask], axis=0)
        tot_c_words = np.sum(c_words)
        denom = tot_c_words + alpha * v_size

        log_likelihood = 0.0
        for t_idx in query_tokens:
            prob_t = (c_words[t_idx] + alpha) / denom
            log_likelihood += np.log(prob_t)

        posteriors[c] = np.log(priors[c]) + log_likelihood

    # Softmax normalization for probabilities
    max_log = max(posteriors.values())
    exp_probs = {c: np.exp(val - max_log) for c, val in posteriors.items()}
    sum_exp = sum(exp_probs.values())
    norm_probs = {c: exp_probs[c] / sum_exp for c in classes}

    return {
        "priors": priors,
        "posterior_probs": norm_probs,
        "predicted_class": max(norm_probs, key=norm_probs.get),
    }


# -----------------------------------------------------------------------------
# 6. Model MRI Influence Functions (Koh & Liang, ICML 2017)
# -----------------------------------------------------------------------------
def fit_logistic_regression_2d(
    X: np.ndarray,
    y: np.ndarray,
    l2_reg: float = 0.1,
    max_iter: int = 100,
    sample_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Damped Newton fit of mean logistic loss + lambda/2 * ||theta||^2.
    y in {-1, +1} or {0, 1}. The intercept is regularized.
    Optional weights multiply losses with the original 1/n denominator retained.
    Returns: (theta, probs, Hessian) where theta has shape (dim,) with bias at end.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) == 0 or y.shape != (len(X),):
        raise ValueError("Expected a nonempty feature matrix and one label per row")
    if not np.isfinite(X).all() or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError("Features must be finite; binary labels must be -1/1 or 0/1")
    if -1 in y and 0 in y:
        raise ValueError("Do not mix -1/1 and 0/1 label encodings")
    if not np.isfinite(l2_reg) or l2_reg <= 0:
        raise ValueError("Positive finite regularization is required")
    n, p = X.shape
    X_aug = np.column_stack([X, np.ones(n)])
    dim = p + 1
    y_binary = (y == 1).astype(float)

    weights = np.ones(n) if sample_weights is None else np.asarray(sample_weights, dtype=float)
    if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Weights must be finite, nonnegative, and match the training rows")
    def sigmoid(z):
        return np.exp(-np.logaddexp(0., -z))
    def objective(th):
        z = X_aug @ th
        return float(np.mean(weights * (np.logaddexp(0., z) - y_binary * z)) + .5 * l2_reg * (th @ th))
    theta = np.zeros(dim)
    for _ in range(max_iter):
        logits = X_aug @ theta
        probs = sigmoid(logits)
        grad = (1.0 / n) * (X_aug.T @ (weights * (probs - y_binary))) + l2_reg * theta
        W = weights * probs * (1.0 - probs)
        H = (1.0 / n) * (X_aug.T @ (W[:, np.newaxis] * X_aug)) + l2_reg * np.eye(dim)
        if np.linalg.norm(grad, ord=np.inf) < 1e-12:
            break
        delta = np.linalg.solve(H, grad)
        step = 1.0
        current = objective(theta)
        while objective(theta - step * delta) > current - 1e-4 * step * float(grad @ delta) + 2e-15 * (1 + abs(current)):
            step *= .5
            if step < 1e-12:
                raise RuntimeError("Logistic line search failed to find a descent step")
        theta -= step * delta
    else:
        raise RuntimeError("Logistic fit did not converge")

    logits = X_aug @ theta
    probs = sigmoid(logits)
    W = weights * probs * (1.0 - probs)
    H_opt = (1.0 / n) * (X_aug.T @ (W[:, np.newaxis] * X_aug)) + l2_reg * np.eye(dim)

    return theta, probs, H_opt


def compute_influence_functions_2d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: int,
    l2_reg: float = 0.1,
) -> Dict[str, Any]:
    """Computes empirical influence functions under a 2D regularized Logistic Regression model.
    Koh & Liang (ICML 2017):
    I_up,loss(z_train, z_test) = - grad_theta(L_test)^T * H^{-1} * grad_theta(L_train)

    Sign convention:
    - Positive influence: infinitesimal upweighting increases test loss.
    - Negative influence: infinitesimal upweighting reduces test loss.
    A finite removal is an approximation, not a guaranteed sign of change.
    """
    n, p = X_train.shape
    X_aug = np.column_stack([X_train, np.ones(n)])
    x_test_aug = np.append(x_test, 1.0)
    theta, probs, H_opt = fit_logistic_regression_2d(X_train, y_train, l2_reg=l2_reg)

    test_logit = float(theta @ x_test_aug)
    test_prob = float(np.exp(-np.logaddexp(0., -test_logit)))
    test_target = 1.0 if y_test == 1 else 0.0
    grad_test = (test_prob - test_target) * x_test_aug

    test_loss = float(np.logaddexp(0., test_logit) - test_target * test_logit)
    v = np.linalg.solve(H_opt, grad_test)

    influences = np.zeros(n)
    for i in range(n):
        train_prob = probs[i]
        train_target = 1.0 if y_train[i] == 1 else 0.0
        grad_train_i = (train_prob - train_target) * X_aug[i]
        influences[i] = -float(np.dot(v, grad_train_i))

    return {
        "theta": theta,
        "test_prob": test_prob,
        "test_loss": test_loss,
        "influences": influences,
        "inverse_hessian_test_gradient": v,
    }


def compute_retrained_counterfactual(
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: int,
    drop_idx: int,
    l2_reg: float = 0.1,
) -> Dict[str, Any]:
    """Evaluates the influence approximation by comparing the Taylor predicted loss change
    against the actual retrained model loss change after dropping sample drop_idx.
    """
    if len(X_train) < 2 or not 0 <= drop_idx < len(X_train):
        raise ValueError("Removal requires at least two rows and a valid training index")
    orig_res = compute_influence_functions_2d(X_train, y_train, x_test, y_test, l2_reg=l2_reg)
    inf_drop = orig_res["influences"][drop_idx]
    n = len(X_train)

    pred_delta_loss = - (1.0 / n) * inf_drop
    # Actual LOO uses 1/(n-1), so the first-order perturbation includes the
    # regularizer gradient via the original optimum's stationarity condition.
    pred_delta_loss_normalized = float(
        (-inf_drop + l2_reg * np.dot(orig_res["inverse_hessian_test_gradient"], orig_res["theta"])) / (n - 1)
    )

    cf_mask = np.ones(n, dtype=bool)
    cf_mask[drop_idx] = False
    X_cf = X_train[cf_mask]
    y_cf = y_train[cf_mask]

    theta_cf, _, _ = fit_logistic_regression_2d(X_cf, y_cf, l2_reg=l2_reg)
    x_test_aug = np.append(x_test, 1.0)
    test_logit_cf = float(theta_cf @ x_test_aug)
    test_prob_cf = float(np.exp(-np.logaddexp(0., -test_logit_cf)))
    test_target = 1.0 if y_test == 1 else 0.0

    test_loss_cf = float(np.logaddexp(0., test_logit_cf) - test_target * test_logit_cf)
    actual_delta_loss = test_loss_cf - orig_res["test_loss"]

    return {
        "theta_orig": orig_res["theta"],
        "theta_cf": theta_cf,
        "test_loss_orig": orig_res["test_loss"],
        "test_loss_cf": test_loss_cf,
        "pred_delta_loss": pred_delta_loss,
        "pred_delta_loss_normalized": pred_delta_loss_normalized,
        "actual_delta_loss": actual_delta_loss,
        "test_prob_orig": orig_res["test_prob"],
        "test_prob_cf": test_prob_cf,
        "inf_drop": inf_drop,
    }


# -----------------------------------------------------------------------------
# 7. Unsupervised Clustering Engine (K-Means & HAC)
# -----------------------------------------------------------------------------
def simulate_kmeans(
    X: np.ndarray,
    k: int = 4,
    max_iter: int = 15,
    random_state: int = 42,
    tol: float = 1e-4,
) -> Dict[str, Any]:
    """Simulates K-Means clustering step-by-step from first principles.
    Records the exact migration history of centroids and assignments at each iteration t.
    """
    rng = np.random.RandomState(random_state)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not len(X) or not np.isfinite(X).all():
        raise ValueError("K-means requires a nonempty finite feature matrix")
    n_samples, n_features = X.shape
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= n_samples or max_iter < 1:
        raise ValueError("Require 1 <= k <= number of samples and max_iter >= 1")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("Tolerance must be finite and nonnegative")

    init_indices = rng.choice(n_samples, size=k, replace=False)
    centroids = X[init_indices].copy()

    history = []
    converged = False
    converged_iter = 0

    for t in range(max_iter):
        dists = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        wcss = float(np.sum(np.min(dists, axis=1)))

        history.append({
            "iteration": t,
            "centroids": centroids.copy(),
            "labels": labels.copy(),
            "wcss": wcss,
        })

        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                new_centroids[j] = np.mean(members, axis=0)
            else:
                new_centroids[j] = centroids[j]

        shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
        if shift <= tol and not converged:
            converged = True
            converged_iter = t

        centroids = new_centroids
        if converged and t >= converged_iter + 1:
            break

    # Report one coherent final state even when the iteration budget is exhausted.
    dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    history.append({"iteration": len(history), "centroids": centroids.copy(),
                    "labels": labels.copy(), "wcss": float(np.min(dists, axis=1).sum())})
    converged_iter = len(history) - 1

    return {
        "history": history,
        "converged": converged,
        "converged_iter": converged_iter,
        "final_centroids": centroids,
        "final_labels": history[-1]["labels"],
        "wcss": history[-1]["wcss"],
    }


def compute_elbow_and_silhouette(
    X: np.ndarray,
    k_range: List[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Computes Inertia / WCSS and Silhouette coefficients across a range of K for the Elbow method."""
    if k_range is None:
        k_range = list(range(2, 9))

    n_samples = len(X)
    wcss_list = []
    silhouette_list = []

    for k in k_range:
        res = simulate_kmeans(X, k=k, max_iter=20, random_state=random_state)
        wcss_list.append(res["wcss"])
        labels = res["final_labels"]

        # Silhouette is undefined for one occupied cluster or all singletons.
        occupied = np.unique(labels)
        if not 2 <= len(occupied) < n_samples:
            silhouette_list.append(float("nan"))
            continue
        s_scores = np.zeros(n_samples)
        for i in range(n_samples):
            own_cluster = labels[i]
            own_members = X[labels == own_cluster]
            if len(own_members) > 1:
                a_i = float(np.mean(np.linalg.norm(own_members - X[i], axis=1)) * (len(own_members) / (len(own_members) - 1)))
            else:
                # The singleton convention is s(i)=0, not (b-0)/b=1.
                continue

            b_i = float("inf")
            for other_c in range(k):
                if other_c == own_cluster:
                    continue
                other_members = X[labels == other_c]
                if len(other_members) > 0:
                    d_mean = float(np.mean(np.linalg.norm(other_members - X[i], axis=1)))
                    if d_mean < b_i:
                        b_i = d_mean
            if b_i == float("inf"):
                b_i = 0.0

            denom = max(a_i, b_i)
            s_scores[i] = (b_i - a_i) / denom if denom > 1e-12 else 0.0

        silhouette_list.append(float(np.mean(s_scores)))

    return {
        "k_values": k_range,
        "wcss": wcss_list,
        "silhouettes": silhouette_list,
    }


def compute_hac_linkage(X: np.ndarray, method: str = "ward") -> np.ndarray:
    """Computes Hierarchical Agglomerative Clustering (HAC) linkage matrix using scipy."""
    from scipy.cluster.hierarchy import linkage
    return linkage(X, method=method)


# -----------------------------------------------------------------------------
# 8. Neural Network (MLP) Forward & Activation Inspector
# -----------------------------------------------------------------------------
class SimpleMLP:
    """2-Layer Multi-Layer Perceptron (Input 2D -> Hidden 4D -> Output 4 Classes).
    Includes step-by-step numerical activation tracking:
    z^[1] = W^[1] x + b^[1] -> a^[1] = ReLU(z^[1]) -> z^[2] = W^[2] a^[1] + b^[2] -> Softmax
    """
    def __init__(self, hidden_dim: int = 4, num_classes: int = 4):
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(hidden_dim, 2) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(num_classes, hidden_dim) * 0.5
        self.b2 = np.zeros(num_classes)

    def fit(self, X: np.ndarray, y_indices: np.ndarray, epochs: int = 150, lr: float = 0.08, l2: float = 1e-4):
        """Fits network via full-batch gradient descent with momentum."""
        n = len(X)
        Y_onehot = np.zeros((n, self.num_classes))
        for i, idx in enumerate(y_indices):
            Y_onehot[i, idx] = 1.0

        vW1, vb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        vW2, vb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        gamma = 0.85

        for _ in range(epochs):
            Z1 = X @ self.W1.T + self.b1
            A1 = np.maximum(0, Z1)
            Z2 = A1 @ self.W2.T + self.b2
            exp_Z2 = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
            Probs = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True)

            dZ2 = (Probs - Y_onehot) / n
            dW2 = dZ2.T @ A1 + l2 * self.W2
            db2 = np.sum(dZ2, axis=0)

            dA1 = dZ2 @ self.W2
            dZ1 = dA1 * (Z1 > 0).astype(float)
            dW1 = dZ1.T @ X + l2 * self.W1
            db1 = np.sum(dZ1, axis=0)

            vW1 = gamma * vW1 + lr * dW1
            vb1 = gamma * vb1 + lr * db1
            vW2 = gamma * vW2 + lr * dW2
            vb2 = gamma * vb2 + lr * db2

            self.W1 -= vW1
            self.b1 -= vb1
            self.W2 -= vW2
            self.b2 -= vb2

        return self

    def inspect_sample(self, x: np.ndarray) -> Dict[str, Any]:
        """Calculates exact forward values for a single input vector x (2D)."""
        x_vec = np.asarray(x, dtype=float).ravel()
        z1 = self.W1 @ x_vec + self.b1
        a1 = np.maximum(0, z1)
        z2 = self.W2 @ a1 + self.b2
        exp_z2 = np.exp(z2 - np.max(z2))
        probs = exp_z2 / np.sum(exp_z2)
        pred_idx = int(np.argmax(probs))

        return {
            "x": x_vec,
            "W1": self.W1,
            "b1": self.b1,
            "z1": z1,
            "a1": a1,
            "W2": self.W2,
            "b2": self.b2,
            "z2": z2,
            "probs": probs,
            "pred_idx": pred_idx,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z1 = X @ self.W1.T + self.b1
        A1 = np.maximum(0, Z1)
        Z2 = A1 @ self.W2.T + self.b2
        return np.argmax(Z2, axis=1)


# -----------------------------------------------------------------------------
# 9. Rigorous Evaluation & Error Diagnostics Suite
# -----------------------------------------------------------------------------
def compute_confusion_matrix_and_metrics(
    y_true: List[str],
    y_pred: List[str],
    class_labels: List[str],
) -> Dict[str, Any]:
    """Computes 4x4 Confusion Matrix, Per-Class Precision/Recall/F1, and Macro-F1 from pure NumPy."""
    n_classes = len(class_labels)
    if len(y_true) != len(y_pred) or not class_labels or len(set(class_labels)) != n_classes:
        raise ValueError("Require aligned predictions and unique, nonempty class labels")
    if not set(y_true).union(y_pred).issubset(class_labels):
        raise ValueError("All observed labels must be included in class_labels")
    label_to_idx = {l: i for i, l in enumerate(class_labels)}

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt], label_to_idx[yp]] += 1

    per_class = {}
    precisions, recalls, f1s = [], [], []

    for i, c in enumerate(class_labels):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        per_class[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": int(support),
        }

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    accuracy = float(np.sum(np.diag(cm)) / np.sum(cm)) if np.sum(cm) > 0 else 0.0

    return {
        "confusion_matrix": cm,
        "class_labels": class_labels,
        "per_class": per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


def compute_bias_variance_curve(
    X: np.ndarray,
    y: np.ndarray,
    max_depth_range: List[int] = None,
) -> Dict[str, Any]:
    """Evaluates Decision Tree Train vs Test Accuracy across max_depth to show overfitting curve."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split

    if max_depth_range is None:
        max_depth_range = list(range(1, 16))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    train_accs = []
    test_accs = []

    for d in max_depth_range:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        train_accs.append(float(dt.score(X_train, y_train)))
        test_accs.append(float(dt.score(X_test, y_test)))

    return {
        "depths": max_depth_range,
        "train_acc": train_accs,
        "test_acc": test_accs,
    }


# -----------------------------------------------------------------------------
# 10. First-Principles vs. Scikit-Learn Verification HUD
# -----------------------------------------------------------------------------
def verify_first_principles_vs_sklearn(
    corpus_texts: List[str],
    categories: List[str],
) -> List[Dict[str, Any]]:
    """Runs automated side-by-side numerical verification between scratch NumPy algorithms
    and standard scikit-learn implementations.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    from glassbox.core.vectorizer import TextVectorizer
    from glassbox.core.geometry import fit_pca_3d
    from glassbox.core.classifiers import LDASimulator

    results = []

    # 1. TF-IDF Matrix Norm Difference
    vec_scratch = TextVectorizer().fit(corpus_texts)
    _, _, X_scratch = vec_scratch.transform(corpus_texts)
    vec_sk = TfidfVectorizer(smooth_idf=True, norm="l2", use_idf=True, token_pattern=r"(?u)\b\w+\b")
    X_sk = vec_sk.fit_transform(corpus_texts).toarray()

    diff_tfidf = float(np.max(np.abs(X_scratch - X_sk)))

    results.append({
        "test": "TF-IDF Matrix Norm L_inf",
        "scratch_desc": "Scratch TextVectorizer (Pure NumPy)",
        "sklearn_desc": "sklearn.feature_extraction.text.TfidfVectorizer",
        "diff": diff_tfidf,
        "tol": 1e-6,
        "passed": bool(diff_tfidf < 1e-6),
        "detail": f"Max matrix difference: {diff_tfidf:.2e} (Threshold: 1e-06)",
    })

    # 2. PCA Eigenvalues Difference
    terms = ["image", "language", "optimization"]
    X_3d = np.column_stack([X_scratch[:, vec_scratch.vocabulary[t]]
                           if t in vec_scratch.vocabulary else np.zeros(len(corpus_texts))
                           for t in terms])
    pca_scratch = fit_pca_3d(X_3d, terms)
    pca_sk = PCA(n_components=3).fit(X_3d)
    diff_eig = float(np.max(np.abs(pca_scratch.eigenvalues - pca_sk.explained_variance_)))

    results.append({
        "test": "PCA Eigenvalues Spectrum",
        "scratch_desc": "Scratch Symmetric Jacobi Rotations",
        "sklearn_desc": "sklearn.decomposition.PCA (SVD LAPACK)",
        "diff": diff_eig,
        "tol": 1e-5,
        "passed": bool(diff_eig < 1e-5),
        "detail": f"Max eigenvalue difference: {diff_eig:.2e} (Threshold: 1e-05)",
    })

    # 3. Sample Covariance Matrix Difference
    cov_scratch = pca_scratch.covariance_matrix
    cov_np = np.cov(X_3d, rowvar=False)
    diff_cov = float(np.max(np.abs(cov_scratch - cov_np)))

    results.append({
        "test": "Sample Covariance Matrix",
        "scratch_desc": "Scratch Sigma = (1/(n-1)) * X_c^T * X_c",
        "sklearn_desc": "numpy.cov(X, rowvar=False)",
        "diff": diff_cov,
        "tol": 1e-12,
        "passed": bool(diff_cov < 1e-12),
        "detail": f"Max matrix difference: {diff_cov:.2e} (Threshold: 1e-12)",
    })

    # 4. Fisher LDA Scatter Invariance (S_W + S_B = S_Total)
    lda_res = LDASimulator.fit(X_3d, categories)
    mean_total = np.mean(X_3d, axis=0)
    S_total = np.sum([(x - mean_total)[:, None] @ (x - mean_total)[None, :] for x in X_3d], axis=0)
    diff_lda = float(np.max(np.abs(lda_res.S_W + lda_res.S_B - S_total)))

    results.append({
        "test": "Fisher LDA Scatter Conservation",
        "scratch_desc": "Scratch S_W (Within) + S_B (Between)",
        "sklearn_desc": "Total Scatter Matrix S_Total",
        "diff": diff_lda,
        "tol": 1e-10,
        "passed": bool(diff_lda < 1e-10),
        "detail": f"Conservation norm: {diff_lda:.2e} (Threshold: 1e-10)",
    })

    return results
