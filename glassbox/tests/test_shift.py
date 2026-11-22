"""Test Evidence Under Shift across all four experimental conditions:
1. Baseline: Hide new papers, frozen IDF (original 7 candidates)
2. Teaching Preset: Hide new papers, refreshed IDF (candidate list fixed, IDF diluted, target paper drops out of top 5)
3. Include new papers, frozen IDF (10 candidates, frozen statistics)
4. Include new papers, refreshed IDF (10 candidates, expanded statistics)
"""

import numpy as np
import pytest

from glassbox.core.vectorizer import TextVectorizer
from glassbox.core.retrieval import EvidenceRetrieval, QueryStatus
from glassbox.data.fixture import (
    get_teaching_fixture,
    get_original_corpus,
    get_incoming_batch,
    DEFAULT_TEACHING_CLAIM,
)


def run_shift_experiment(include_incoming: bool, refresh_idf: bool):
    """Executes search under the specified shift conditions."""
    orig_docs = get_original_corpus()
    all_docs = get_teaching_fixture()

    orig_texts = [d.abstract for d in orig_docs]
    all_texts = [d.abstract for d in all_docs]

    # Fit baseline vectorizer on original 7 documents
    vec_orig = TextVectorizer().fit(orig_texts)

    if not refresh_idf:
        active_vec = vec_orig
    else:
        # Refreshed IDF condition: Vocabulary frozen from original corpus!
        active_vec = TextVectorizer()
        active_vec.feature_names = vec_orig.feature_names
        active_vec.vocabulary = vec_orig.vocabulary
        active_vec.n_documents = len(all_texts)

        # Recompute DF and IDF on expanded 10-document corpus
        v_size = len(vec_orig.feature_names)
        df_expanded = np.zeros(v_size, dtype=np.int64)
        for t_doc in [active_vec.tokenizer(d) for d in all_texts]:
            for term in set(t_doc):
                if term in active_vec.vocabulary:
                    df_expanded[active_vec.vocabulary[term]] += 1
        active_vec.doc_frequencies = df_expanded
        active_vec.idf_weights = np.log((1.0 + active_vec.n_documents) / (1.0 + df_expanded)) + 1.0
        active_vec.is_fitted = True

    # Candidates eligibility
    candidate_docs = all_docs if include_incoming else orig_docs
    candidate_texts = [d.abstract for d in candidate_docs]
    candidate_ids = [d.doc_id for d in candidate_docs]
    candidate_titles = [d.title for d in candidate_docs]
    candidate_annotations = [d.annotation for d in candidate_docs]

    c_mat, z_mat, x_mat = active_vec.transform(candidate_texts)
    retrieval = EvidenceRetrieval(active_vec)

    status, msg, results = retrieval.search(
        DEFAULT_TEACHING_CLAIM,
        candidate_ids,
        candidate_titles,
        candidate_texts,
        candidate_annotations,
        x_mat,
        c_mat,
    )
    return active_vec, results


def test_baseline_top_five():
    """Verify that in the baseline condition, DOC-03 is within the top five."""
    vec, results = run_shift_experiment(include_incoming=False, refresh_idf=False)
    top_5_ids = [r.doc_id for r in results[:5]]

    assert "DOC-03" in top_5_ids
    doc3_result = next(r for r in results if r.doc_id == "DOC-03")
    assert doc3_result.rank == 5
    assert len(results) == 7


def test_preset_hide_new_papers_refresh_idf_demotes_target():
    """Verify the core teaching preset:
    - Fixed candidates (original 7 papers only)
    - Recomputed IDF on expanded 10-document corpus
    - DF of 'consolidation' increases
    - IDF of 'consolidation' decreases
    - DOC-03 drops from rank 5 to rank 6 (falls out of top five!)
    - DOC-03 remains inspectable in candidate list
    """
    # 1. Baseline
    vec_base, results_base = run_shift_experiment(include_incoming=False, refresh_idf=False)
    rank_base = {r.doc_id: r.rank for r in results_base}
    assert rank_base["DOC-03"] == 5

    # 2. Preset
    vec_preset, results_preset = run_shift_experiment(include_incoming=False, refresh_idf=True)
    rank_preset = {r.doc_id: r.rank for r in results_preset}

    # Verify candidate count is still 7
    assert len(results_preset) == 7

    # Verify IDF decrease for key term 'consolidation'
    idx_cons = vec_base.vocabulary["consolidation"]
    df_base_cons = vec_base.doc_frequencies[idx_cons]
    df_preset_cons = vec_preset.doc_frequencies[idx_cons]
    idf_base_cons = vec_base.idf_weights[idx_cons]
    idf_preset_cons = vec_preset.idf_weights[idx_cons]

    # DF must have increased due to incoming batch
    assert df_preset_cons > df_base_cons
    # IDF must have decreased
    assert idf_preset_cons < idf_base_cons

    # Verify DOC-03 fell out of top 5
    assert rank_preset["DOC-03"] > 5
    assert rank_preset["DOC-03"] == 6
    top_5_preset = [r.doc_id for r in results_preset[:5]]
    assert "DOC-03" not in top_5_preset

    # Verify DOC-06 took its place in the top 5
    assert "DOC-06" in top_5_preset

    # Verify comparison helper
    comp = EvidenceRetrieval.compare_shift(
        results_base, results_preset, include_incoming=False, refresh_idf=True, target_doc_id="DOC-03"
    )
    assert comp.target_baseline_rank == 5
    assert comp.target_updated_rank == 6
    target_item = next(it for it in comp.comparison_items if it.doc_id == "DOC-03")
    assert target_item.movement_label == "fell 1 place"


def test_all_four_shift_combinations_execute():
    """Verify all 4 combinations run cleanly and return valid search results."""
    combos = [
        (False, False),  # Baseline
        (False, True),   # Preset
        (True, False),   # Include new, frozen IDF
        (True, True),    # Include new, refreshed IDF
    ]

    for inc, ref in combos:
        vec, results = run_shift_experiment(include_incoming=inc, refresh_idf=ref)
        expected_len = 10 if inc else 7
        assert len(results) == expected_len
        # Ensure ranks are contiguous 1..N
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, expected_len + 1))
        # Ensure scores are non-increasing
        scores = [r.score for r in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
