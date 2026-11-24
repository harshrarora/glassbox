"""Stage 2: The Preprocessing & Linguistics Lab (HIGH PRIORITY).
Interactive step-by-step visualizer for regex sanitization, tokenization,
stopword elimination, and the Stemming vs. Lemmatization battle.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from glassbox.data.arxiv_corpus import get_arxiv_corpus
from glassbox.utils.visual_components import (
    render_hero_card,
    render_contraction_meter,
    render_living_diff_text,
)
from glassbox.utils.math_engine import (
    clean_regex_text,
    tokenize_text,
    filter_stopwords,
    compare_stemmer_lemmatizer,
    compute_vocabulary_contraction,
    STEMMER,
    LEMMATIZER,
    MORPHOLOGY_BACKEND,
)


def render_stage2():
    render_hero_card(
        stage_title="Stage 2: The Preprocessing & Linguistics Lab",
        what_text="Progressively sanitizing raw text through regular expressions, tokenizing into discrete word chips, filtering uninformative stopwords, and resolving inflectional endings.",
        why_text="Inflectional variations (e.g. 'run', 'running', 'runs') and syntactic glue words ('the', 'which') needlessly multiply matrix columns. Linguistic compression shrinks feature dimensionality while preserving core semantics.",
    )

    papers = get_arxiv_corpus()
    st.caption(f"Active morphology backend: {MORPHOLOGY_BACKEND}. Offline fallback rules are not the full Porter or WordNet algorithms.")
    all_abstracts = [p.abstract for p in papers]

    st.markdown("###  Interactive Linguistic Transformation Sandbox")
    st.caption("Select any paper from the corpus and toggle each preprocessing operation on or off to inspect the live visual diff.")

    col_paper, col_options = st.columns([2.0, 2.0])
    with col_paper:
        selected_id = st.selectbox(
            "Select Abstract to Preprocess:",
            options=[p.id for p in papers],
            format_func=lambda pid: f"[{next(p.category for p in papers if p.id == pid)}] {next(p.title for p in papers if p.id == pid)}",
        )
    chosen_paper = next(p for p in papers if p.id == selected_id)

    from glassbox.utils.pipeline import settings, update_setting, analyze
    cfg = settings()
    with col_options:
        st.markdown("#### Transformation Pipeline Toggles")
        t_regex = st.checkbox("Strip LaTeX, URLs and digits", value=cfg["regex"], key="prep_regex", on_change=update_setting, args=("regex", "prep_regex"))
        t_stop = st.checkbox("Remove stopwords", value=cfg["stopwords"], key="prep_stop", on_change=update_setting, args=("stopwords", "prep_stop"))
        st.selectbox("Morphology passed downstream", ["None", "Lemma", "Stem"], index=["None", "Lemma", "Stem"].index(cfg["morphology"]), key="prep_morph", on_change=update_setting, args=("morphology", "prep_morph"))
        t_morph = st.checkbox("Compare stemmer and lemmatizer", value=True)
        st.caption("These settings feed stages 3 to 5. Tokenization is always applied to create numerical features.")

    # Execute transformations on chosen text
    raw_text = chosen_paper.abstract
    cleaned_text, removed_regex = clean_regex_text(raw_text)
    tokens = tokenize_text(cleaned_text if t_regex else raw_text)
    retained_tokens, discarded_stops = filter_stopwords(tokens)

    # 2. Living Visual Diff View
    st.markdown("####  Living Visual Diff: What Gets Stripped")
    diff_html = render_living_diff_text(
        raw_text,
        cleaned_text,
        removed_regex,
        discarded_stops,
        show_regex_diff=t_regex,
        show_stopword_diff=t_stop,
    )
    st.markdown(
        f"""
        <div style="background:#ffffff; border:1px solid #cbd5e1; border-radius:8px;
                    padding:16px; line-height:1.9; font-size:0.95rem;">
          {diff_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Legend: <del style='color:#ef4444; background:#fee2e2; padding:1px 4px;'>Red Strikethrough</del> = Stripped by Regex &nbsp;|&nbsp; "
        "<span style='color:#b45309; background:#fef3c7; text-decoration:line-through; padding:1px 4px;'>Amber Strikethrough</span> = Filtered Stopword &nbsp;|&nbsp; "
        "Normal Text = Retained Informative Feature Tokens",
        unsafe_allow_html=True,
    )

    # 3. Discarded Stopwords Telemetry
    if t_stop and discarded_stops:
        with st.expander(f" Discarded Stopwords ({len(discarded_stops)} words removed from this abstract)"):
            stop_freq = pd.Series(discarded_stops).value_counts().reset_index()
            stop_freq.columns = ["Stopword", "Occurrences in Document"]
            st.dataframe(stop_freq.T, use_container_width=True)

    # 4. The Stemming vs Lemmatization Battle (Side-by-Side Comparison)
    if t_morph:
        st.markdown("---")
        st.markdown("###  The Stemming vs. Lemmatization Battle")
        st.markdown(
            """
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px 16px; margin-bottom:12px; font-size:0.9rem;">
              • <b>Porter Stemmer (1980):</b> Rule-based heuristic suffix chopping. Extremely fast, but frequently produces crude non-words (e.g. <code>studies</code> → <code>studi</code>, <code>density</code> → <code>densiti</code>).<br>
              • <b>WordNet Lemmatizer:</b> Dictionary-backed morphological analysis. Results depend on part of speech; this lab uses the noun default. Simplified offline rules may produce different results.
            </div>
            """,
            unsafe_allow_html=True,
        )

        comparison_data = compare_stemmer_lemmatizer(retained_tokens if t_stop else tokens)
        df_comp = pd.DataFrame(comparison_data)

        # Filter to show differing and notable transformations first
        diff_cases = df_comp[df_comp["differs"]]
        same_cases = df_comp[~df_comp["differs"]]

        col_tbl, col_diff_stats = st.columns([2.0, 1.2])

        with col_tbl:
            st.markdown(f"**Tokens Where Stemmer and Lemmatizer Diverge ({len(diff_cases)} words):**")
            if not diff_cases.empty:
                st.dataframe(
                    diff_cases[["original", "stem", "lemma"]].rename(
                        columns={"original": "Original word", "stem": "Active stemmer", "lemma": "Active lemmatizer"}
                    ),
                    use_container_width=True,
                    height=280,
                )
            else:
                st.info("In this abstract, all retained tokens produced identical stems and lemmas.")

        with col_diff_stats:
            st.markdown("####  Linguistic Comparison Metrics")
            st.metric("Unique Retained Tokens", len(df_comp))
            st.metric("Diverging Words (Stem ≠ Lemma)", len(diff_cases), f"{(len(diff_cases)/len(df_comp)*100):.1f}% of vocabulary")

            st.markdown(
                """
                <div style="background:#eff6ff; border-left:4px solid #3b82f6; padding:10px; border-radius:0 6px 6px 0; font-size:0.85rem; color:#1e40af;">
                  <b>Inspect the backend:</b> Rule-based stemming and dictionary lookup make different assumptions. Lemmatization requires an appropriate part-of-speech setting; it does not infer meaning here.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("Vocabulary contraction: the full cleaning recipe"):
        st.caption("Reference sequence: regex → stopwords → lemmas. The active pipeline may use a different choice above.")
        render_contraction_meter(compute_vocabulary_contraction(all_abstracts))
    st.markdown("#### Tokens passed to vectorization")
    st.write(" · ".join(analyze(raw_text, **{k: cfg[k] for k in ("regex", "stopwords", "morphology", "bigrams")})))
