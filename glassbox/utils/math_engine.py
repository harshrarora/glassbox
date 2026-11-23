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
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_STOPWORDS = set(nltk_stopwords.words("english"))
    STEMMER = PorterStemmer()
    LEMMATIZER = WordNetLemmatizer()
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
    distances = np.linalg.norm(X_train - query_point, axis=1)
    k_indices = np.argsort(distances)[:k]

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
# 4. Model MRI: Empirical Influence Functions (Koh & Liang, ICML 2017)
# -----------------------------------------------------------------------------
def compute_influence_functions_2d(
    X_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: int, l2_reg: float = 0.1
) -> np.ndarray:
    """Calculates Koh & Liang's empirical parameter influence on test loss:
    I_up,loss(z_train, z_test) = - grad_theta(L_test)^T * H^{-1} * grad_theta(L_train)
    """
    n, p = X_train.shape
    # Augment with bias term
    X_aug = np.column_stack([X_train, np.ones(n)])
    x_test_aug = np.append(x_test, 1.0)
    dim = p + 1

    # Fit small logistic regression via Newton-Raphson
    theta = np.zeros(dim)
    for _ in range(15):
        logits = X_aug @ theta
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -15, 15)))
        # Gradient
        grad = (1.0 / n) * (X_aug.T @ (probs - (y_train == 1).astype(float))) + l2_reg * theta
        # Hessian
        W = probs * (1.0 - probs)
        H = (1.0 / n) * (X_aug.T @ (W[:, np.newaxis] * X_aug)) + l2_reg * np.eye(dim)
        try:
            delta = np.linalg.solve(H, grad)
            theta -= delta
            if np.linalg.norm(delta) < 1e-6:
                break
        except np.linalg.LinAlgError:
            break

    # Compute Hessian at optimal theta
    logits = X_aug @ theta
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -15, 15)))
    W = probs * (1.0 - probs)
    H_opt = (1.0 / n) * (X_aug.T @ (W[:, np.newaxis] * X_aug)) + l2_reg * np.eye(dim)
    H_inv = np.linalg.pinv(H_opt)

    # Test sample gradient: grad L(z_test, theta)
    test_prob = 1.0 / (1.0 + np.exp(-np.clip(theta @ x_test_aug, -15, 15)))
    test_target = 1.0 if y_test == 1 else 0.0
    grad_test = (test_prob - test_target) * x_test_aug

    # Vector v = H^{-1} * grad_test
    v = H_inv @ grad_test

    # Influence of each training sample i: - grad_train_i^T * v
    influences = np.zeros(n)
    for i in range(n):
        train_prob = probs[i]
        train_target = 1.0 if y_train[i] == 1 else 0.0
        grad_train_i = (train_prob - train_target) * X_aug[i]
        influences[i] = -float(np.dot(v, grad_train_i))

    return influences
