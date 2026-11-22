"""Vectorization and Document-Term Matrix calculations from scratch.

Conforms strictly to standard educational conventions:
  c(t, d) = raw count of term t in document d
  idf(t)  = ln((1 + N) / (1 + DF(t))) + 1
  z(t, d) = c(t, d) * idf(t)
  x(t, d) = z(t, d) / ||z(:, d)||_2 for ||z(:, d)||_2 > 0 else 0
"""

import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def default_tokenizer(text: str) -> List[str]:
    """Deterministic tokenization rule:
    Converts text to lowercase and extracts all word tokens matching
    the regular expression `(?u)\\b\\w+\\b` (words containing alphanumeric
    characters and underscores, length >= 1).
    """
    if not text:
        return []
    return re.findall(r"(?u)\b\w+\b", text.lower())


class TextVectorizer:
    """Computes raw counts, smooth-IDF, unnormalized TF-IDF, and L2-normalized TF-IDF.
    Provides complete step-by-step mathematical inspectability.
    """

    def __init__(self, tokenizer=default_tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary: Dict[str, int] = {}
        self.feature_names: List[str] = []
        self.doc_frequencies: np.ndarray = np.array([])
        self.idf_weights: np.ndarray = np.array([])
        self.n_documents: int = 0
        self.is_fitted: bool = False

    def fit(self, documents: List[str]) -> "TextVectorizer":
        """Learn vocabulary and inverse document frequencies from training documents."""
        self.n_documents = len(documents)
        if self.n_documents == 0:
            raise ValueError("Cannot fit vectorizer on an empty document list.")

        # Tokenize each document
        tokenized_docs = [self.tokenizer(doc) for doc in documents]

        # Build sorted, deterministic vocabulary
        unique_tokens = set()
        for tokens in tokenized_docs:
            unique_tokens.update(tokens)
        self.feature_names = sorted(list(unique_tokens))
        self.vocabulary = {term: idx for idx, term in enumerate(self.feature_names)}

        v_size = len(self.feature_names)
        df = np.zeros(v_size, dtype=np.int64)

        for tokens in tokenized_docs:
            doc_unique = set(tokens)
            for term in doc_unique:
                if term in self.vocabulary:
                    df[self.vocabulary[term]] += 1

        self.doc_frequencies = df

        # idf(t) = ln((1 + N) / (1 + DF(t))) + 1
        self.idf_weights = np.log((1.0 + self.n_documents) / (1.0 + self.doc_frequencies)) + 1.0
        self.is_fitted = True
        return self

    def transform_raw_counts(self, documents: List[str]) -> np.ndarray:
        """Compute the raw document-term count matrix C: shape (len(documents), len(vocabulary))."""
        if not self.is_fitted:
            raise RuntimeError("TextVectorizer must be fitted before calling transform.")

        matrix = np.zeros((len(documents), len(self.feature_names)), dtype=np.float64)
        for i, doc in enumerate(documents):
            tokens = self.tokenizer(doc)
            for t in tokens:
                if t in self.vocabulary:
                    matrix[i, self.vocabulary[t]] += 1.0
        return matrix

    def transform(self, documents: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform documents into:
        1. C: raw count matrix
        2. Z: unnormalized TF-IDF matrix = C * idf
        3. X: L2-normalized TF-IDF matrix = Z / ||Z||_2
        """
        c_matrix = self.transform_raw_counts(documents)
        z_matrix = c_matrix * self.idf_weights[np.newaxis, :]

        # Compute Euclidean (L2) norm for each row
        row_norms = np.linalg.norm(z_matrix, ord=2, axis=1, keepdims=True)
        # Avoid division by zero for all-zero rows
        with np.errstate(divide="ignore", invalid="ignore"):
            x_matrix = np.where(row_norms > 0, z_matrix / row_norms, 0.0)

        return c_matrix, z_matrix, x_matrix

    def fit_transform(self, documents: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit to documents and transform in one call."""
        self.fit(documents)
        return self.transform(documents)

    def inspect_cell(
        self,
        document_text: str,
        term: str,
        c_matrix: np.ndarray,
        z_matrix: np.ndarray,
        x_matrix: np.ndarray,
        doc_idx: int,
    ) -> Dict[str, Any]:
        """Return exact mathematical breakdown for a specific (document, term) cell."""
        if not self.is_fitted:
            raise RuntimeError("TextVectorizer must be fitted before inspecting cells.")

        term_lower = term.lower().strip()
        term_idx = self.vocabulary.get(term_lower, None)

        tokens = self.tokenizer(document_text)
        total_tokens = len(tokens)

        if term_idx is None:
            return {
                "term": term_lower,
                "in_vocabulary": False,
                "raw_count": 0,
                "doc_token_count": total_tokens,
                "relative_freq": 0.0,
                "denominator_name": "Total word tokens in document",
                "N": self.n_documents,
                "DF": 0,
                "idf": 0.0,
                "idf_formula": "Term not in vocabulary",
                "unnormalized_tfidf": 0.0,
                "doc_l2_norm": 0.0,
                "normalized_tfidf": 0.0,
            }

        raw_count = int(c_matrix[doc_idx, term_idx])
        rel_freq = (raw_count / total_tokens) if total_tokens > 0 else 0.0
        df = int(self.doc_frequencies[term_idx])
        idf = float(self.idf_weights[term_idx])
        unnorm = float(z_matrix[doc_idx, term_idx])
        l2_norm = float(np.linalg.norm(z_matrix[doc_idx, :], ord=2))
        normed = float(x_matrix[doc_idx, term_idx])

        idf_formula = (
            f"ln((1 + N) / (1 + DF(t))) + 1 = "
            f"ln((1 + {self.n_documents}) / (1 + {df})) + 1 = "
            f"ln({1 + self.n_documents}/{1 + df}) + 1 = {idf:.4f}"
        )

        return {
            "term": term_lower,
            "in_vocabulary": True,
            "term_idx": term_idx,
            "raw_count": raw_count,
            "doc_token_count": total_tokens,
            "relative_freq": rel_freq,
            "denominator_name": "Total word tokens in document d",
            "N": self.n_documents,
            "DF": df,
            "idf": idf,
            "idf_formula": idf_formula,
            "unnormalized_tfidf": unnorm,
            "doc_l2_norm": l2_norm,
            "normalized_tfidf": normed,
        }
