"""Stage 1: The Raw Corpus Hub.
Explores raw, unstructured scientific text before machine learning touches it.
"""

import streamlit as st
import plotly.express as px
import pandas as pd
from typing import List
from glassbox.fixtures.arxiv_corpus import PaperRecord, get_arxiv_corpus
from glassbox.utils.visual_components import render_hero_card


CATEGORY_NAMES = {
    "cs.CV": "Computer Vision",
    "cs.CL": "Computation & Language (NLP)",
    "cs.LG": "Machine Learning (Deep Learning Theory)",
    "stat.ML": "Optimization & Statistical Learning",
}

CATEGORY_COLORS = {
    "cs.CV": "#10b981",    # Emerald
    "cs.CL": "#6366f1",    # Indigo
    "cs.LG": "#f59e0b",    # Amber
    "stat.ML": "#f43f5e",   # Rose
}


def render_stage1():
    render_hero_card(
        stage_title="Stage 1: The Raw Corpus Hub",
        what_text="Browsing 64 real, uncurated scientific research abstracts directly extracted from arXiv across four foundational machine learning and AI disciplines.",
        why_text="Machine learning models cannot read raw prose. We must first understand the natural language distribution, length variations, and vocabulary size before applying algorithmic transformations.",
    )

    papers = get_arxiv_corpus()
    all_abstracts = [p.abstract for p in papers]

    # Global Corpus Statistics
    word_counts = [len(p.abstract.split()) for p in papers]
    char_lengths = [len(p.abstract) for p in papers]
    all_raw_tokens = set(" ".join(all_abstracts).lower().split())
    v_raw = len(all_raw_tokens)

    st.markdown("### 📊 Raw Corpus Telemetry")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Research Papers", len(papers), "4 Balanced Domains")
    with col2:
        st.metric("Raw Vocabulary Size (V_raw)", f"{v_raw:,}", "Unfiltered Word Tokens")
    with col3:
        st.metric("Avg Words per Abstract", f"{sum(word_counts)/len(word_counts):.1f}", f"Min: {min(word_counts)}, Max: {max(word_counts)}")
    with col4:
        st.metric("Total Corpus Characters", f"{sum(char_lengths):,}", "Raw Text Volume")

    # Domain Distribution Chart
    st.markdown("---")
    col_chart, col_filter = st.columns([1.6, 1.4])

    with col_chart:
        cat_counts = pd.Series([p.category for p in papers]).value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        cat_counts["Name"] = cat_counts["Category"].map(CATEGORY_NAMES)

        fig_dist = px.bar(
            cat_counts,
            x="Name",
            y="Count",
            color="Category",
            color_discrete_map=CATEGORY_COLORS,
            title="Ground-Truth Domain Distribution (16 Papers per Category)",
            text="Count",
        )
        fig_dist.update_layout(height=280, margin=dict(l=20, r=20, b=20, t=35), showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_filter:
        st.markdown("#### 🔍 Filter & Explore Papers")
        selected_cat = st.selectbox(
            "Filter by Domain Category:",
            options=["All Domains", "cs.CV", "cs.CL", "cs.LG", "stat.ML"],
            format_func=lambda c: f"{c}: {CATEGORY_NAMES.get(c, 'All Domains')}",
        )
        search_query = st.text_input("Search titles or abstracts by keyword:", placeholder="e.g. attention, gradient, convolution...")

        filtered_papers = papers
        if selected_cat != "All Domains":
            filtered_papers = [p for p in filtered_papers if p.category == selected_cat]
        if search_query.strip():
            q = search_query.lower().strip()
            filtered_papers = [p for p in filtered_papers if q in p.title.lower() or q in p.abstract.lower()]

        st.caption(f"Showing **{len(filtered_papers)}** of {len(papers)} papers")

    # Paper Cards
    st.markdown("#### 📄 Scientific Paper Inspector")
    if not filtered_papers:
        st.warning("No papers match your current filter.")
    else:
        for p in filtered_papers[:6]:
            with st.container():
                st.markdown(
                    f"""
                    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:14px; margin-bottom:10px;">
                      <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; font-size:1.05rem; color:#0f172a;">{p.title}</span>
                        <span style="background:{CATEGORY_COLORS[p.category]}; color:#ffffff; padding:2px 8px; border-radius:12px; font-weight:700; font-size:0.75rem;">
                          {p.category}
                        </span>
                      </div>
                      <div style="font-size:0.85rem; color:#64748b; margin:4px 0;">
                        <b>Authors:</b> {p.authors} &nbsp;|&nbsp; <b>Year:</b> {p.year} &nbsp;|&nbsp; <code>{p.id}</code>
                      </div>
                      <div style="font-size:0.9rem; color:#334155; line-height:1.6; margin-top:8px;">
                        {p.abstract}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        if len(filtered_papers) > 6:
            st.info(f"Showing first 6 matches. Use domain filter or keyword search to narrow down.")
