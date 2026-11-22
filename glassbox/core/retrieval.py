"""Evidence retrieval, cosine similarity, term breakdown, and Evidence Under Shift engine.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .vectorizer import TextVectorizer


class QueryStatus(str, Enum):
    VALID = "valid"
    BLANK = "blank"
    NO_VOCAB_OVERLAP = "no_vocab_overlap"


@dataclass
class TermContribution:
    term: str
    query_raw_count: int
    query_normalized_weight: float
    doc_raw_count: int
    doc_normalized_weight: float
    contribution: float


@dataclass
class SearchResult:
    doc_id: str
    title: str
    abstract: str
    annotation: str
    score: float
    rank: int
    matching_terms: List[TermContribution] = field(default_factory=list)
    sum_contributions: float = 0.0


@dataclass
class ShiftItem:
    doc_id: str
    title: str
    annotation: str
    baseline_rank: Optional[int]
    baseline_score: Optional[float]
    updated_rank: Optional[int]
    updated_score: Optional[float]
    movement_label: str  # e.g., "fell 2 places", "rose 1 place", "unchanged", "new candidate"
    is_demoted_target: bool = False


@dataclass
class ShiftComparison:
    baseline_results: List[SearchResult]
    updated_results: List[SearchResult]
    comparison_items: List[ShiftItem]
    include_incoming: bool
    refresh_idf: bool
    target_doc_id: str
    target_baseline_rank: Optional[int]
    target_updated_rank: Optional[int]


class EvidenceRetrieval:
    """Manages document retrieval, cosine similarity ranking, and corpus shift experiments."""

    def __init__(self, vectorizer: TextVectorizer):
        self.vectorizer = vectorizer

    def process_query(self, query: str) -> Tuple[QueryStatus, str, Optional[np.ndarray], Dict[str, int]]:
        """Validate, tokenize, and compute L2-normalized TF-IDF vector for a query.
        Returns: (status, message, query_vector, query_token_counts)
        """
        trimmed = query.strip()
        if not trimmed:
            return QueryStatus.BLANK, "Please enter a claim to search the literature.", None, {}

        tokens = self.vectorizer.tokenizer(trimmed)
        if not tokens:
            return QueryStatus.BLANK, "Please enter a claim containing valid words.", None, {}

        # Count tokens present in the vocabulary
        matching_counts: Dict[str, int] = {}
        for t in tokens:
            if t in self.vectorizer.vocabulary:
                matching_counts[t] = matching_counts.get(t, 0) + 1

        if not matching_counts:
            return (
                QueryStatus.NO_VOCAB_OVERLAP,
                "This claim shares no terms with the current vocabulary.",
                None,
                {},
            )

        # Build raw count vector for query
        q_raw = np.zeros(len(self.vectorizer.feature_names), dtype=np.float64)
        for term, count in matching_counts.items():
            idx = self.vectorizer.vocabulary[term]
            q_raw[idx] = float(count)

        # Apply IDF weighting: z = c * idf
        q_z = q_raw * self.vectorizer.idf_weights

        # L2 normalize
        norm = np.linalg.norm(q_z, ord=2)
        if norm > 0:
            q_norm = q_z / norm
        else:
            q_norm = np.zeros_like(q_z)

        return QueryStatus.VALID, "OK", q_norm, matching_counts

    def search(
        self,
        query: str,
        candidate_ids: List[str],
        candidate_titles: List[str],
        candidate_abstracts: List[str],
        candidate_annotations: List[str],
        candidate_x_norm: np.ndarray,
        candidate_c_counts: np.ndarray,
    ) -> Tuple[QueryStatus, str, List[SearchResult]]:
        """Score candidate documents against query using cosine similarity and stable tie-breaking."""
        status, msg, q_vec, q_counts = self.process_query(query)
        if status != QueryStatus.VALID or q_vec is None:
            return status, msg, []

        results: List[SearchResult] = []
        n_candidates = len(candidate_ids)

        for i in range(n_candidates):
            d_vec = candidate_x_norm[i]
            # Cosine similarity on L2-normalized vectors is simply the dot product
            score = float(np.dot(q_vec, d_vec))

            # Breakdown matching terms
            matching_terms: List[TermContribution] = []
            running_sum = 0.0

            for term, q_cnt in q_counts.items():
                t_idx = self.vectorizer.vocabulary[term]
                q_w = float(q_vec[t_idx])
                d_w = float(d_vec[t_idx])
                contrib = q_w * d_w
                d_cnt = int(candidate_c_counts[i, t_idx])

                if contrib > 1e-12 or (q_w > 0 and d_w > 0):
                    matching_terms.append(
                        TermContribution(
                            term=term,
                            query_raw_count=q_cnt,
                            query_normalized_weight=q_w,
                            doc_raw_count=d_cnt,
                            doc_normalized_weight=d_w,
                            contribution=contrib,
                        )
                    )
                    running_sum += contrib

            # Sort matching terms by contribution descending
            matching_terms.sort(key=lambda item: item.contribution, reverse=True)

            results.append(
                SearchResult(
                    doc_id=candidate_ids[i],
                    title=candidate_titles[i],
                    abstract=candidate_abstracts[i],
                    annotation=candidate_annotations[i],
                    score=score,
                    rank=0,  # will assign below
                    matching_terms=matching_terms,
                    sum_contributions=running_sum,
                )
            )

        # Deterministic ranking: score descending, then doc_id ascending for ties
        results.sort(key=lambda r: (-r.score, r.doc_id))
        for r_idx, r in enumerate(results):
            r.rank = r_idx + 1

        return QueryStatus.VALID, "OK", results

    @staticmethod
    def compare_shift(
        baseline_results: List[SearchResult],
        updated_results: List[SearchResult],
        include_incoming: bool,
        refresh_idf: bool,
        target_doc_id: str,
    ) -> ShiftComparison:
        """Analyze rank movement between baseline results and updated results."""
        baseline_map = {r.doc_id: r for r in baseline_results}
        updated_map = {r.doc_id: r for r in updated_results}

        # Union of candidate IDs in display order
        # Show updated list first, then any baseline documents that fell out of updated list
        all_ids: List[str] = [r.doc_id for r in updated_results]
        for r in baseline_results:
            if r.doc_id not in all_ids:
                all_ids.append(r.doc_id)

        items: List[ShiftItem] = []
        for d_id in all_ids:
            b_res = baseline_map.get(d_id)
            u_res = updated_map.get(d_id)

            title = u_res.title if u_res else (b_res.title if b_res else "")
            annotation = u_res.annotation if u_res else (b_res.annotation if b_res else "")

            b_rank = b_res.rank if b_res else None
            b_score = b_res.score if b_res else None
            u_rank = u_res.rank if u_res else None
            u_score = u_res.score if u_res else None

            if b_rank is None and u_rank is not None:
                mov = "new candidate"
            elif u_rank is None and b_rank is not None:
                mov = "excluded from candidates"
            elif b_rank is not None and u_rank is not None:
                diff = b_rank - u_rank  # positive if moved up (e.g. from 5 to 3 is +2)
                if diff > 0:
                    mov = f"rose {diff} place{'s' if diff > 1 else ''}"
                elif diff < 0:
                    mov = f"fell {abs(diff)} place{'s' if abs(diff) > 1 else ''}"
                else:
                    mov = "unchanged"
            else:
                mov = "unranked"

            items.append(
                ShiftItem(
                    doc_id=d_id,
                    title=title,
                    annotation=annotation,
                    baseline_rank=b_rank,
                    baseline_score=b_score,
                    updated_rank=u_rank,
                    updated_score=u_score,
                    movement_label=mov,
                    is_demoted_target=(d_id == target_doc_id),
                )
            )

        target_b_rank = baseline_map[target_doc_id].rank if target_doc_id in baseline_map else None
        target_u_rank = updated_map[target_doc_id].rank if target_doc_id in updated_map else None

        return ShiftComparison(
            baseline_results=baseline_results,
            updated_results=updated_results,
            comparison_items=items,
            include_incoming=include_incoming,
            refresh_idf=refresh_idf,
            target_doc_id=target_doc_id,
            target_baseline_rank=target_b_rank,
            target_updated_rank=target_u_rank,
        )
