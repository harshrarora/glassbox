"""Stage 3: Vectorization & Feature Engineering.
Interactive count/TF-IDF matrix toggle, plugged-in math breakdown cards,
N-gram expansion explorer, and matrix sparsity visualizer.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from glassbox.fixtures.arxiv_corpus import get_arxiv_corpus
from glassbox.utils.visual_components import render_hero_card
from glassbox.utils.math_engine import (
    build_count_matrix,
    compute_tfidf_from_counts,
)


def render_stage3():
    render_hero_card(
        stage_title="Stage 3: Vectorization & Feature Engineering",
        what_text="Translating discrete linguistic tokens into high-dimensional numerical coordinates across the document-term matrix.",
        why_text="Algorithms require numerical vector spaces to compute distances and angles. TF-IDF normalizes for document length and discounts ubiquitous terms, amplifying distinctive domain concepts.",
    )

    papers = get_arxiv_corpus()
    all_abstracts = [p.abstract for p in papers]

    # Controls: N-Gram Range & View Mode
    ctrl_col1, ctrl_col2 = st.columns([1.5, 1.5])
    with ctrl_col1:
        ngram_choice = st.radio(
            "Feature N-Gram Range (Vocabulary Explosion):",
            options=["Unigrams (1-word tokens)", "Unigrams + Bigrams (1- and 2-word phrases)"],
            horizontal=True,
        )
        ngram_range = (1, 2) if "Bigrams" in ngram_choice else (1, 1)

    with ctrl_col2:
        matrix_mode = st.radio(
            "Matrix Numerical Representation:",
            options=["L2-Normalized TF-IDF", "Raw Integer Counts", "Unnormalized TF-IDF (C × IDF)"],
            horizontal=True,
        )

    # Compute Matrices
    C, vocab = build_count_matrix(all_abstracts, ngram_range=ngram_range)
    df_counts, idf_weights, X_norm, sparsity = compute_tfidf_from_counts(C)

    # Sparsity & Dimensionality Telemetry
    col_dim, col_spar, col_docs, col_top_terms = st.columns(4)
    with col_dim:
        st.metric("Vocabulary Dimension (Columns)", f"{len(vocab):,}", f"{'+Bigrams' if ngram_range[1]>1 else 'Unigrams only'}")
    with col_spar:
        st.metric("Matrix Sparsity", f"{sparsity*100:.1f}%", "Fraction of Empty (Zero) Cells")
    with col_docs:
        st.metric("Documents (Rows)", len(papers), "64 arXiv Papers")
    with col_top_terms:
        non_zeros = np.count_nonzero(C)
        st.metric("Non-Zero Value Cells", f"{non_zeros:,}", f"{(1-sparsity)*100:.1f}% matrix fill")

    st.markdown("---")
    # Interactive Heatmap Matrix View
    st.markdown("### 🌡️ Document-Term Matrix Sparsity Visualizer")
    st.caption("Notice how real text matrices are overwhelmingly sparse (>95% zeros). Hover over any block to inspect coordinate values.")

    # Render a 15-doc by 20-term interactive heatmap snippet
    sub_docs_idx = list(range(0, len(papers), 4))[:15]
    top_vocab_indices = np.argsort(df_counts)[::-1][:20]
    sub_vocab = [vocab[i] for i in top_vocab_indices]

    if "Raw" in matrix_mode:
        disp_matrix = C[np.ix_(sub_docs_idx, top_vocab_indices)]
        color_scale = "Blues"
    elif "Unnormalized" in matrix_mode:
        disp_matrix = (C * idf_weights[np.newaxis, :])[np.ix_(sub_docs_idx, top_vocab_indices)]
        color_scale = "Viridis"
    else:
        disp_matrix = X_norm[np.ix_(sub_docs_idx, top_vocab_indices)]
        color_scale = "Purples"

    paper_labels = [f"[{papers[i].category}] {papers[i].id.split(':')[1]}" for i in sub_docs_idx]

    fig_heat = px.imshow(
        disp_matrix,
        x=sub_vocab,
        y=paper_labels,
        labels=dict(x="Vocabulary Feature", y="Document", color=matrix_mode),
        color_continuous_scale=color_scale,
        aspect="auto",
        title=f"Sample Heatmap ({matrix_mode}) : Non-zero informative coordinates",
    )
    fig_heat.update_layout(height=340, margin=dict(l=30, r=20, b=30, t=40))
    st.plotly_chart(fig_heat, use_container_width=True)

    # Plugged-in Math Breakdown Card
    st.markdown("---")
    st.markdown("### 📐 Step-by-Step Mathematical Substitution Inspector")
    st.caption("Select any document and vocabulary term to see the exact numbers substituted into the formula.")

    col_pick_doc, col_pick_term = st.columns([2.0, 2.0])
    with col_pick_doc:
        chosen_doc_idx = st.selectbox(
            "Select Document d:",
            options=list(range(len(papers))),
            format_func=lambda i: f"[{papers[i].category}] {papers[i].title[:45]}...",
        )
    with col_pick_term:
        # Show terms present in this document first
        present_terms = [vocab[idx] for idx in np.where(C[chosen_doc_idx] > 0)[0]]
        chosen_term = st.selectbox(
            "Select Term t:",
            options=present_terms if present_terms else vocab[:15],
        )

    t_idx = vocab.index(chosen_term) if chosen_term in vocab else 0
    raw_c = int(C[chosen_doc_idx, t_idx])
    doc_total_words = int(np.sum(C[chosen_doc_idx]))
    term_df = int(df_counts[t_idx])
    term_idf = float(idf_weights[t_idx])
    unnorm_z = raw_c * term_idf
    doc_norm = float(np.linalg.norm(C[chosen_doc_idx] * idf_weights))
    norm_x = (unnorm_z / doc_norm) if doc_norm > 0 else 0.0

    st.markdown(
        f"""
        <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:10px; padding:18px; margin-top:8px;">
          <div style="font-weight:700; font-size:1.1rem; color:#1e293b; margin-bottom:10px;">
            Target Cell Coordinate: [d = {papers[chosen_doc_idx].id}, t = "{chosen_term}"]
          </div>
          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
            <div>
              <b>1. Term Frequency (TF):</b>
              <div style="font-family:monospace; background:#fff; border:1px solid #e2e8f0; padding:8px; border-radius:6px; margin:4px 0;">
                TF(t, d) = c(t, d) / ∑_w c(w, d)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {raw_c} / {doc_total_words} = <b>{(raw_c/doc_total_words if doc_total_words else 0):.4f}</b>
              </div>
              <b>2. Smoothed Inverse Document Frequency (IDF):</b>
              <div style="font-family:monospace; background:#fff; border:1px solid #e2e8f0; padding:8px; border-radius:6px; margin:4px 0;">
                IDF(t) = ln((1 + N) / (1 + DF(t))) + 1<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= ln((1 + {len(papers)}) / (1 + {term_df})) + 1 = <b>{term_idf:.4f}</b>
              </div>
            </div>
            <div>
              <b>3. Unnormalized Product z(t, d):</b>
              <div style="font-family:monospace; background:#fff; border:1px solid #e2e8f0; padding:8px; border-radius:6px; margin:4px 0;">
                z(t, d) = c(t, d) × IDF(t)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {raw_c} × {term_idf:.4f} = <b>{unnorm_z:.4f}</b>
              </div>
              <b>4. L2-Normalized Coordinate x(t, d):</b>
              <div style="font-family:monospace; background:#fff; border:1px solid #6366f1; padding:8px; border-radius:6px; margin:4px 0; border-left:4px solid #6366f1;">
                x(t, d) = z(t, d) / ||z(:, d)||₂<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {unnorm_z:.4f} / {doc_norm:.4f} = <b>{norm_x:.4f}</b>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
