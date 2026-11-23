"""GlassBox: Interactive Mathematical Simulation Engine.

Pedagogical Visual Laboratory:
  Stage 1: Token Microscope, Matrix Heatmap, 3D PCA & SVD Rank-k Snapping
  Stage 2: LDA Scatter Matrices (S_W, S_B) & SVM Margin Gutters
  Stage 3: Evidence Under Shift & Lesson 1 Teaching Preset
"""

import sys
import os
import re

# Ensure local package importability
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from glassbox.core.vectorizer import TextVectorizer
from glassbox.core.geometry import (
    fit_pca_3d,
    PCA3DResult,
    compute_svd_rank_k,
    SVDRankKResult,
)
from glassbox.core.classifiers import (
    LDASimulator,
    LDAResult,
    SVMSimulator,
    SVMResult,
)
from glassbox.core.retrieval import (
    EvidenceRetrieval,
    SearchResult,
    QueryStatus,
)
from glassbox.data.arxiv_corpus import (
    PaperRecord,
    get_arxiv_corpus,
)
from glassbox.data.fixture import (
    DEFAULT_TEACHING_CLAIM,
    DEFAULT_PCA_TERMS,
    get_teaching_fixture,
    get_original_corpus,
    get_incoming_batch,
)

# -----------------------------------------------------------------------------
# Configuration and Global Styling (3Blue1Brown / Explorable Style)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GlassBox — Mathematical Simulation Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&family=Inter:wght@400;600;700&display=swap');

    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e293b 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.4rem;
        line-height: 1.5;
    }
    .badge-cv { background-color: #d1fae5; color: #065f46; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.8rem; }
    .badge-cl { background-color: #e0e7ff; color: #3730a3; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.8rem; }
    .badge-lg { background-color: #fef3c7; color: #92400e; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.8rem; }
    .badge-stat { background-color: #ffe4e6; color: #9f1239; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.8rem; }

    .math-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .math-step {
        font-family: 'Fira Code', monospace;
        font-size: 0.95rem;
        color: #1e293b;
        background: #ffffff;
        border: 1px solid #cbd5e1;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 6px 0;
    }
    .takeaway-box {
        background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 14px 18px;
        color: #1e3a8a;
        margin-top: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Cached Datasets and Vectorizers
# -----------------------------------------------------------------------------
@st.cache_data
def load_arxiv_data():
    return get_arxiv_corpus()

arxiv_papers = load_arxiv_data()
arxiv_texts = [p.abstract for p in arxiv_papers]
arxiv_categories = [p.category for p in arxiv_papers]
arxiv_id_to_paper = {p.id: p for p in arxiv_papers}

CATEGORY_COLORS = {
    "cs.CV": "#10b981",    # Emerald
    "cs.CL": "#6366f1",    # Indigo
    "cs.LG": "#f59e0b",    # Amber
    "stat.ML": "#f43f5e",   # Rose
}

@st.cache_resource
def get_arxiv_vectorizer():
    vec = TextVectorizer().fit(arxiv_texts)
    c, z, x = vec.transform(arxiv_texts)
    return vec, c, z, x

vec_arxiv, c_arxiv, z_arxiv, x_arxiv = get_arxiv_vectorizer()

# -----------------------------------------------------------------------------
# Global Navigation & Mode Selection
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔬 GlassBox Lab")
    st.markdown("**Pedagogical Visual Simulation Engine**")
    st.caption("UML501 (Machine Learning) & UCS551 (Accelerated Data Science)")
    st.markdown("---")

    mode = st.radio(
        "Simulation Stage:",
        options=[
            "1. Feature Vectors, 3D PCA & SVD Simulator",
            "2. Supervised LDA (S_W/S_B) & SVM Margin Explorer",
            "3. Evidence Retrieval & Shift Laboratory",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📚 Corpus Summary")
    st.markdown(f"• **{len(arxiv_papers)} Real arXiv Abstracts**")
    st.markdown("• **cs.CV:** 16 papers *(Vision & Diffusion)*")
    st.markdown("• **cs.CL:** 16 papers *(Language & Transformers)*")
    st.markdown("• **cs.LG:** 16 papers *(Generalization & Theory)*")
    st.markdown("• **stat.ML:** 16 papers *(Optimization & Bounds)*")
    st.markdown(f"• **Vocabulary Size:** {len(vec_arxiv.feature_names)} terms")


# =============================================================================
# STAGE 1: Feature Representation, 3D PCA & SVD Simulator
# =============================================================================
if mode.startswith("1."):
    st.markdown('<div class="main-title">Stage 1: Feature Representation, 3D PCA & SVD Reconstruction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Watch raw scientific text transform into weighted sparse vectors. '
        "Explore 3D geometric variance, covariance heatmaps, and truncated SVD rank-k reconstruction snapping.</div>",
        unsafe_allow_html=True,
    )

    tab_microscope, tab_pca_svd = st.tabs([
        "🔬 The Token Microscope (Text to Vector Strip)",
        "🌐 3D PCA & SVD Rank-k Simulator",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: The Token Microscope
    # -------------------------------------------------------------------------
    with tab_microscope:
        st.markdown("### Interactive Abstract Vectorizer")
        st.caption("Select a real arXiv research paper. Words in the abstract light up according to their exact L2-normalized TF-IDF weight.")

        col_sel_paper, col_sel_term = st.columns([2.5, 1.5])
        with col_sel_paper:
            selected_paper_id = st.selectbox(
                "Select Research Paper:",
                options=[p.id for p in arxiv_papers],
                format_func=lambda pid: f"[{arxiv_id_to_paper[pid].category}] {arxiv_id_to_paper[pid].title} ({arxiv_id_to_paper[pid].authors.split(',')[0]} et al., {arxiv_id_to_paper[pid].year})",
            )
        paper = arxiv_id_to_paper[selected_paper_id]
        paper_idx = [p.id for p in arxiv_papers].index(selected_paper_id)

        # Get top words in this document by TF-IDF
        doc_weights = x_arxiv[paper_idx]
        top_word_indices = np.argsort(doc_weights)[::-1][:15]
        top_terms = [vec_arxiv.feature_names[i] for i in top_word_indices if doc_weights[i] > 0]

        with col_sel_term:
            inspect_term = st.selectbox(
                "Inspect Term Derivation:",
                options=top_terms if top_terms else vec_arxiv.feature_names[:10],
            )

        col_text_view, col_math_view = st.columns([1.5, 1.5])

        with col_text_view:
            st.markdown(f"#### 📄 {paper.title}")
            cat_badge = f'<span class="badge-{paper.category.split(".")[1].lower()}">{paper.category}</span>'
            st.markdown(f"{cat_badge} &nbsp; <b>{paper.authors}</b> ({paper.year}) &nbsp; <code>{paper.id}</code>", unsafe_allow_html=True)

            # Highlight words in abstract according to TF-IDF weight
            words = re.findall(r"\b\w+\b|[^\w\s]|\s+", paper.abstract)
            annotated_html = []
            max_w = float(np.max(doc_weights)) if np.max(doc_weights) > 0 else 1.0

            for w in words:
                clean_w = w.lower().strip()
                if clean_w in vec_arxiv.vocabulary:
                    t_idx = vec_arxiv.vocabulary[clean_w]
                    weight = doc_weights[t_idx]
                    if weight > 0:
                        alpha = min(0.85, 0.15 + 0.7 * (weight / max_w))
                        is_sel = (clean_w == inspect_term.lower())
                        border = "2px solid #ca8a04" if is_sel else "none"
                        bg = "#fef08a" if is_sel else f"rgba(99, 102, 241, {alpha:.2f})"
                        color = "#000" if is_sel else ("#ffffff" if alpha > 0.5 else "#1e1b4b")
                        annotated_html.append(
                            f'<span style="background:{bg}; color:{color}; padding:2px 5px; border-radius:4px; font-weight:600; border:{border}; cursor:pointer;" '
                            f'title="{clean_w}: weight={weight:.4f}">{w}</span>'
                        )
                    else:
                        annotated_html.append(w)
                else:
                    annotated_html.append(w)

            st.markdown(
                f'<div style="background:#ffffff; border:1px solid #cbd5e1; border-radius:8px; padding:14px; line-height:1.8; margin-top:10px;">'
                f'{"".join(annotated_html)}</div>',
                unsafe_allow_html=True,
            )
            st.caption("Visual intensity corresponds to relative TF-IDF importance. Clicked term is highlighted in gold.")

        with col_math_view:
            st.markdown(f"#### 📐 Exact Mathematical Substitution: `{inspect_term}`")
            cell_info = vec_arxiv.inspect_cell(paper.abstract, inspect_term, c_arxiv, z_arxiv, x_arxiv, paper_idx)

            st.markdown(
                f"""
                <div class="math-card">
                  <div style="font-weight:700; margin-bottom:8px;">1. Term Frequency in Document:</div>
                  <div class="math-step">
                    c("{inspect_term}", d) = {cell_info['raw_count']} &nbsp;&nbsp;|&nbsp;&nbsp;
                    f("{inspect_term}", d) = {cell_info['raw_count']} / {cell_info['doc_token_count']} = {cell_info['relative_freq']:.4f}
                  </div>
                  <div style="font-weight:700; margin-top:12px; margin-bottom:8px;">2. Smoothed Inverse Document Frequency (IDF):</div>
                  <div class="math-step">
                    IDF("{inspect_term}") = ln((1 + N) / (1 + DF)) + 1<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= ln((1 + {cell_info['N']}) / (1 + {cell_info['DF']})) + 1<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= ln({cell_info['N']+1}/{cell_info['DF']+1}) + 1 = <b>{cell_info['idf']:.4f}</b>
                  </div>
                  <div style="font-weight:700; margin-top:12px; margin-bottom:8px;">3. Unnormalized Product z(t, d):</div>
                  <div class="math-step">
                    z("{inspect_term}", d) = c(t, d) × IDF(t) = {cell_info['raw_count']} × {cell_info['idf']:.4f} = <b>{cell_info['unnormalized_tfidf']:.4f}</b>
                  </div>
                  <div style="font-weight:700; margin-top:12px; margin-bottom:8px;">4. Euclidean (L2) Normalization:</div>
                  <div class="math-step" style="border-left:4px solid #6366f1;">
                    x("{inspect_term}", d) = z(t, d) / ||z(:, d)||₂<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {cell_info['unnormalized_tfidf']:.4f} / {cell_info['doc_l2_norm']:.4f} = <b>{cell_info['normalized_tfidf']:.4f}</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 2D Heatmap / Vector Strip
        st.markdown("#### 🌡️ Top Informative Features (Vector Strip Heatmap)")
        top_terms_display = top_terms[:12] if top_terms else vec_arxiv.feature_names[:12]
        term_sub_indices = [vec_arxiv.vocabulary[t] for t in top_terms_display]

        # Get papers in same category for clean heatmap
        cat_paper_indices = [i for i, p in enumerate(arxiv_papers) if p.category == paper.category][:8]
        cat_paper_ids = [arxiv_papers[i].id.split(":")[1] for i in cat_paper_indices]
        heatmap_matrix = x_arxiv[np.ix_(cat_paper_indices, term_sub_indices)]

        fig_heat = px.imshow(
            heatmap_matrix,
            labels=dict(x="Vocabulary Term", y="arXiv Paper ID", color="TF-IDF Weight"),
            x=top_terms_display,
            y=cat_paper_ids,
            color_continuous_scale="Purples",
            aspect="auto",
            title=f"L2-Normalized Feature Coordinates across {paper.category} Literature",
        )
        fig_heat.update_layout(height=280, margin=dict(l=40, r=20, b=30, t=40))
        st.plotly_chart(fig_heat, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: 3D PCA & SVD Rank-k Simulator
    # -------------------------------------------------------------------------
    with tab_pca_svd:
        st.markdown("### 3D Geometric PCA & Truncated SVD Reconstruction")
        st.caption("Inspect the 64 arXiv papers as 3D coordinate vectors across distinctive domain keywords. Observe covariance, principal axes, and rank-k SVD reconstruction.")

        col_terms, col_svd_k = st.columns([2.5, 1.5])
        with col_terms:
            triplet_presets = {
                "Vision vs NLP vs Optimization: ['image', 'language', 'optimization']": ["image", "language", "optimization"],
                "Deep Learning Architectures: ['convolutional', 'transformer', 'gradient']": ["convolutional", "transformer", "gradient"],
                "Foundations: ['visual', 'translation', 'convex']": ["visual", "translation", "convex"],
            }
            chosen_preset = st.selectbox("Feature Coordinate Triplet Preset:", options=list(triplet_presets.keys()))
            selected_pca_terms = triplet_presets[chosen_preset]

        with col_svd_k:
            rank_k = st.slider("Truncated SVD Rank k (Reconstruction Simulator):", min_value=1, max_value=3, value=2)

        pca_term_indices = [vec_arxiv.vocabulary.get(t, 0) for t in selected_pca_terms]
        X_3d_arxiv = x_arxiv[:, pca_term_indices]

        pca_3d_res = fit_pca_3d(X_3d_arxiv, selected_pca_terms)
        svd_res = compute_svd_rank_k(X_3d_arxiv, k=rank_k)

        col_3d_canvas, col_svd_metrics = st.columns([1.7, 1.3])

        with col_3d_canvas:
            fig_3d = go.Figure()

            # Render SVD reconstructed vs original coordinates
            # When rank_k < 3, show reconstructed points with snapping lines
            pts_display = svd_res.X_reconstructed if rank_k < 3 else X_3d_arxiv

            for cat in ["cs.CV", "cs.CL", "cs.LG", "stat.ML"]:
                mask = np.array([p.category == cat for p in arxiv_papers])
                pts_cat = pts_display[mask]
                ids_cat = [p.id for i, p in enumerate(arxiv_papers) if mask[i]]
                titles_cat = [p.title for i, p in enumerate(arxiv_papers) if mask[i]]

                fig_3d.add_trace(
                    go.Scatter3d(
                        x=pts_cat[:, 0],
                        y=pts_cat[:, 1],
                        z=pts_cat[:, 2],
                        mode="markers",
                        marker=dict(size=7, color=CATEGORY_COLORS[cat], symbol="circle"),
                        text=[f"<b>{pid}</b>: {t}" for pid, t in zip(ids_cat, titles_cat)],
                        name=f"{cat} ({np.sum(mask)} papers)",
                    )
                )

            # Draw Centroid Mean Vector μ
            mu = pca_3d_res.mean_vector
            fig_3d.add_trace(
                go.Scatter3d(
                    x=[mu[0]], y=[mu[1]], z=[mu[2]],
                    mode="markers+text",
                    marker=dict(size=11, color="#f59e0b", symbol="diamond"),
                    text=["Centroid μ"],
                    name="Global Mean μ",
                )
            )

            # Draw PC1 and PC2 axes from mean
            v1 = pca_3d_res.eigenvectors[:, 0] * 0.2
            v2 = pca_3d_res.eigenvectors[:, 1] * 0.2
            fig_3d.add_trace(
                go.Scatter3d(
                    x=[mu[0], mu[0] + v1[0]],
                    y=[mu[1], mu[1] + v1[1]],
                    z=[mu[2], mu[2] + v1[2]],
                    mode="lines+text",
                    line=dict(color="#10b981", width=6),
                    text=["", "PC1"],
                    name="PC1 Axis",
                )
            )
            fig_3d.add_trace(
                go.Scatter3d(
                    x=[mu[0], mu[0] + v2[0]],
                    y=[mu[1], mu[1] + v2[1]],
                    z=[mu[2], mu[2] + v2[2]],
                    mode="lines+text",
                    line=dict(color="#6366f1", width=6),
                    text=["", "PC2"],
                    name="PC2 Axis",
                )
            )

            fig_3d.update_layout(
                title=f"3D Space: {'Rank-' + str(rank_k) + ' SVD Truncated State' if rank_k < 3 else 'Original Coordinates'}",
                scene=dict(
                    xaxis=dict(title=f"'{selected_pca_terms[0]}'"),
                    yaxis=dict(title=f"'{selected_pca_terms[1]}'"),
                    zaxis=dict(title=f"'{selected_pca_terms[2]}'"),
                ),
                height=480,
                margin=dict(l=0, r=0, b=0, t=30),
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        with col_svd_metrics:
            st.markdown("#### 📐 SVD & Covariance Heatmap")

            # Covariance matrix heatmap
            fig_cov = px.imshow(
                pca_3d_res.covariance_matrix,
                labels=dict(x="Feature", y="Feature", color="Covariance"),
                x=selected_pca_terms,
                y=selected_pca_terms,
                color_continuous_scale="Blues",
                title="Sample Covariance Matrix Σ",
            )
            fig_cov.update_layout(height=220, margin=dict(l=30, r=20, b=20, t=35))
            st.plotly_chart(fig_cov, use_container_width=True)

            st.markdown(
                f"""
                <div class="math-card">
                  <div style="font-weight:700; margin-bottom:6px;">SVD Decomposition: <code>X = U Σ Vᵀ</code></div>
                  • Singular Values: <code>σ = [{svd_res.singular_values[0]:.4f}, {svd_res.singular_values[1]:.4f}, {svd_res.singular_values[2]:.4f}]</code><br>
                  • Selected Rank: <b>k = {rank_k}</b><br>
                  • Energy (Variance) Captured: <b>{svd_res.energy_captured*100:.2f}%</b><br>
                  • Frobenius Reconstruction Error: <code>||X - X̂_k||_F = {svd_res.frobenius_error:.4f}</code><br>
                  • Relative Residual Error: <b>{svd_res.relative_error*100:.2f}%</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="takeaway-box">
              <b>3Blue1Brown Geometric Insight:</b><br>
              When <b>k = 1</b>, every paper vector is collapsed onto the 1D line spanned by the first right singular vector <code>v₁</code>.
              When <b>k = 2</b>, papers snap onto the optimal 2D hyperplane, capturing <b>{svd_res.energy_captured*100:.2f}%</b> of the total matrix energy.
              At <b>k = 3</b>, the full rank is restored, reducing the Frobenius reconstruction error to <b>0.0000</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# STAGE 2: Supervised Linear Boundaries (LDA & SVM)
# =============================================================================
elif mode.startswith("2."):
    st.markdown('<div class="main-title">Stage 2: Supervised Linear Boundaries & Geometry</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Explore how known class labels transform dimensionality reduction from variance maximization (PCA) '
        "to class separability (LDA), and interact with Support Vector Machine (SVM) margin gutters and regularizer C.</div>",
        unsafe_allow_html=True,
    )

    tab_lda, tab_svm = st.tabs([
        "🔬 Linear Discriminant Analysis (S_W & S_B Scatter Simulator)",
        "⚔️ Support Vector Machine (SVM) Margin & Hyperplane Explorer",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Linear Discriminant Analysis
    # -------------------------------------------------------------------------
    with tab_lda:
        st.markdown("### Multiclass Scatter Matrices: $S_W$ vs $S_B$")
        st.caption("LDA maximizes between-class scatter while minimizing within-class variance: J(w) = (wᵀ S_B w) / (wᵀ S_W w).")

        # Compute LDA on full 3D features
        lda_terms = ["image", "language", "optimization"]
        idx_lda = [vec_arxiv.vocabulary[t] for t in lda_terms]
        X_lda = x_arxiv[:, idx_lda]
        lda_res: LDAResult = LDASimulator.fit(X_lda, arxiv_categories)

        col_sw_heat, col_sb_heat = st.columns(2)
        with col_sw_heat:
            fig_sw = px.imshow(
                lda_res.S_W,
                x=lda_terms,
                y=lda_terms,
                color_continuous_scale="Reds",
                title="Within-Class Scatter Matrix S_W = ∑_c ∑ (x - μ_c)(x - μ_c)ᵀ",
            )
            fig_sw.update_layout(height=260, margin=dict(l=30, r=20, b=20, t=35))
            st.plotly_chart(fig_sw, use_container_width=True)

        with col_sb_heat:
            fig_sb = px.imshow(
                lda_res.S_B,
                x=lda_terms,
                y=lda_terms,
                color_continuous_scale="Greens",
                title="Between-Class Scatter Matrix S_B = ∑_c N_c (μ_c - μ)(μ_c - μ)ᵀ",
            )
            fig_sb.update_layout(height=260, margin=dict(l=30, r=20, b=20, t=35))
            st.plotly_chart(fig_sb, use_container_width=True)

        # 2D Projection Comparison: PCA vs LDA
        col_proj_lda, col_proj_pca = st.columns(2)

        with col_proj_lda:
            fig_lda_2d = go.Figure()
            for cat in ["cs.CV", "cs.CL", "cs.LG", "stat.ML"]:
                mask = np.array([p.category == cat for p in arxiv_papers])
                pts = lda_res.projected_2d[mask]
                fig_lda_2d.add_trace(
                    go.Scatter(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        mode="markers",
                        marker=dict(size=9, color=CATEGORY_COLORS[cat]),
                        name=cat,
                    )
                )
            fig_lda_2d.update_layout(
                title=f"LDA Projection: Maximum Class Separability (λ₁={lda_res.eigenvalues[0]:.2f}, λ₂={lda_res.eigenvalues[1]:.2f})",
                xaxis_title="Discriminant Axis 1 (w₁)",
                yaxis_title="Discriminant Axis 2 (w₂)",
                height=380,
                margin=dict(l=30, r=20, b=30, t=40),
            )
            st.plotly_chart(fig_lda_2d, use_container_width=True)

        with col_proj_pca:
            pca_res_2d = fit_pca_3d(X_lda, lda_terms)
            fig_pca_comp = go.Figure()
            for cat in ["cs.CV", "cs.CL", "cs.LG", "stat.ML"]:
                mask = np.array([p.category == cat for p in arxiv_papers])
                pts = pca_res_2d.scores_2d[mask]
                fig_pca_comp.add_trace(
                    go.Scatter(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        mode="markers",
                        marker=dict(size=9, color=CATEGORY_COLORS[cat]),
                        name=cat,
                    )
                )
            fig_pca_comp.update_layout(
                title=f"PCA Projection: Total Variance Maximization (Unsupervised)",
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2",
                height=380,
                margin=dict(l=30, r=20, b=30, t=40),
            )
            st.plotly_chart(fig_pca_comp, use_container_width=True)

        st.markdown(
            """
            <div class="takeaway-box">
              <b>Core Mathematical Distinction (PCA vs LDA):</b><br>
              Notice how PCA projects along the direction of greatest total spread without regard for class labels, leading to overlapping clusters.
              In contrast, <b>LDA solves (S_W⁻¹ S_B) w = λ w</b>, explicitly contracting within-class spread (S_W) while expanding the distance
              between category centroids (S_B), producing tight, cleanly separated clusters!
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # TAB 2: Support Vector Machine (SVM) Margin Explorer
    # -------------------------------------------------------------------------
    with tab_svm:
        st.markdown("### The SVM Margin & Support Vector Simulator")
        st.caption("Classify two research domains in 2D space. Adjust regularization penalty C to watch the margin gutters expand or contract live.")

        col_svm_pair, col_svm_c = st.columns([2.0, 2.0])
        with col_svm_pair:
            pair_choice = st.selectbox(
                "Binary Classification Task:",
                options=[
                    ("cs.CV", "cs.CL", "Computer Vision vs Computation & Language"),
                    ("cs.CV", "stat.ML", "Computer Vision vs Statistical Optimization"),
                    ("cs.CL", "stat.ML", "Language Models vs Statistical Optimization"),
                ],
                format_func=lambda opt: f"{opt[0]} vs {opt[1]} ({opt[2]})",
            )
            cat_pos, cat_neg, pair_desc = pair_choice

        with col_svm_c:
            c_val = st.slider("Regularization Penalty C (Soft vs Hard Margin):", min_value=0.01, max_value=20.0, value=1.0, step=0.1)

        # Filter papers to the selected pair
        mask_binary = np.array([p.category in [cat_pos, cat_neg] for p in arxiv_papers])
        sub_papers = [p for i, p in enumerate(arxiv_papers) if mask_binary[i]]
        X_sub_2d = lda_res.projected_2d[mask_binary]
        y_binary = np.array([1 if p.category == cat_pos else -1 for p in sub_papers])

        # Fit SVM
        svm_res: SVMResult = SVMSimulator.fit(X_sub_2d, y_binary, C=c_val)

        col_svm_plot, col_svm_inspect = st.columns([1.8, 1.2])

        with col_svm_plot:
            fig_svm = go.Figure()

            # Plot Gutters and Hyperplane
            fig_svm.add_trace(
                go.Scatter(
                    x=svm_res.line_x,
                    y=svm_res.line_boundary,
                    mode="lines",
                    line=dict(color="#0f172a", width=3),
                    name="Hyperplane: wᵀ x + b = 0",
                )
            )
            fig_svm.add_trace(
                go.Scatter(
                    x=svm_res.line_x,
                    y=svm_res.line_gutter_pos,
                    mode="lines",
                    line=dict(color="#10b981", width=2, dash="dash"),
                    name="Gutter: wᵀ x + b = +1",
                )
            )
            fig_svm.add_trace(
                go.Scatter(
                    x=svm_res.line_x,
                    y=svm_res.line_gutter_neg,
                    mode="lines",
                    line=dict(color="#6366f1", width=2, dash="dash"),
                    name="Gutter: wᵀ x + b = -1",
                )
            )

            # Plot Data Points
            for cat, label_val, color in [(cat_pos, 1, CATEGORY_COLORS[cat_pos]), (cat_neg, -1, CATEGORY_COLORS[cat_neg])]:
                cat_mask = (y_binary == label_val)
                pts = X_sub_2d[cat_mask]
                paper_ids = [sub_papers[i].id for i in range(len(sub_papers)) if cat_mask[i]]
                paper_titles = [sub_papers[i].title for i in range(len(sub_papers)) if cat_mask[i]]

                fig_svm.add_trace(
                    go.Scatter(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        mode="markers",
                        marker=dict(size=9, color=color),
                        text=[f"<b>{pid}</b>: {t}" for pid, t in zip(paper_ids, paper_titles)],
                        name=f"{cat} ({np.sum(cat_mask)} papers)",
                    )
                )

            # Highlight Support Vectors with Glowing Rings
            sv_pts = X_sub_2d[svm_res.support_vector_indices]
            fig_svm.add_trace(
                go.Scatter(
                    x=sv_pts[:, 0],
                    y=sv_pts[:, 1],
                    mode="markers",
                    marker=dict(
                        size=18,
                        color="rgba(0,0,0,0)",
                        line=dict(color="#f59e0b", width=3),
                    ),
                    name=f"Support Vectors ({len(svm_res.support_vector_indices)})",
                )
            )

            # Limit plot range to avoid infinite lines
            y_pts_all = X_sub_2d[:, 1]
            y_span = np.max(y_pts_all) - np.min(y_pts_all)
            fig_svm.update_layout(
                title=f"SVM Margin (C={c_val:.2f}): Width γ = {svm_res.margin_width:.4f} | Accuracy = {svm_res.accuracy*100:.1f}%",
                xaxis_title="Discriminant Feature 1",
                yaxis_title="Discriminant Feature 2",
                yaxis=dict(range=[np.min(y_pts_all) - 0.4 * y_span, np.max(y_pts_all) + 0.4 * y_span]),
                height=460,
                margin=dict(l=30, r=20, b=30, t=40),
            )
            st.plotly_chart(fig_svm, use_container_width=True)

        with col_svm_inspect:
            st.markdown("#### 📐 Margin Equations & Metrics")
            st.markdown(
                f"""
                <div class="math-card">
                  <div style="font-weight:700; margin-bottom:6px;">1. Hyperplane Parameters:</div>
                  • Normal Vector <code>w = [{svm_res.w[0]:.4f}, {svm_res.w[1]:.4f}]</code><br>
                  • Euclidean Norm <code>||w||₂ = {np.linalg.norm(svm_res.w):.4f}</code><br>
                  • Bias Term <code>b = {svm_res.b:.4f}</code><br><br>
                  <div style="font-weight:700; margin-bottom:6px;">2. Margin Width:</div>
                  <div class="math-step">
                    γ = 2 / ||w||₂ = 2 / {np.linalg.norm(svm_res.w):.4f} = <b>{svm_res.margin_width:.4f}</b>
                  </div>
                  <div style="font-weight:700; margin-top:10px; margin-bottom:6px;">3. Regularization Tradeoff:</div>
                  • Penalty <b>C = {c_val}</b><br>
                  • Support Vectors Identified: <b>{len(svm_res.support_vector_indices)} of {len(sub_papers)}</b><br>
                  • Classification Accuracy: <b>{svm_res.accuracy*100:.1f}%</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Point Inspector
            st.markdown("#### 🔍 Individual Point Inspector")
            sel_pt_idx = st.selectbox(
                "Select Paper to Inspect Margin & Slack:",
                options=list(range(len(sub_papers))),
                format_func=lambda idx: f"{sub_papers[idx].id}: {sub_papers[idx].title[:35]}...",
            )
            f_margin = svm_res.sample_margins[sel_pt_idx]
            slack = svm_res.sample_slacks[sel_pt_idx]
            is_sv = sel_pt_idx in svm_res.support_vector_indices

            st.markdown(
                f"""
                <div class="math-card" style="border-left:4px solid {'#f59e0b' if is_sv else '#94a3b8'};">
                  <b>Paper: {sub_papers[sel_pt_idx].id}</b> ({sub_papers[sel_pt_idx].category})<br>
                  • Ground Truth Label: <code>y = {y_binary[sel_pt_idx]:+d}</code><br>
                  • Functional Margin: <code>y(wᵀx + b) = {f_margin:+.4f}</code><br>
                  • Slack Variable: <code>ξ = {slack:.4f}</code><br>
                  • Status: <b>{'★ SUPPORT VECTOR' if is_sv else 'Non-critical Point'}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="takeaway-box">
              <b>Bret Victor Explorable Insight:</b><br>
              Drag the <b>C slider</b>! As C increases (approaching a hard margin), the optimizer heavily penalizes margin violations,
              forcing the margin width <code>γ = 2/||w||</code> to narrow and reducing support vectors to the critical boundary boundary points.
              As C decreases (soft margin), the margin expands, allowing more points to enter the gutter to maximize generalization robustness.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# STAGE 3: Evidence Retrieval & Shift Laboratory (Preserving Lesson 1)
# =============================================================================
elif mode.startswith("3."):
    st.markdown('<div class="main-title">Stage 3: Evidence Retrieval & Statistical Shift Laboratory</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">The core Lesson 1 pedagogical paradox: Can the same paper fall out of the shortlist '
        "when its own text and candidate eligibility remain fixed?</div>",
        unsafe_allow_html=True,
    )

    fixture_all = get_teaching_fixture()
    fixture_orig = get_original_corpus()
    fixture_incoming = get_incoming_batch()

    orig_texts_f = [d.abstract for d in fixture_orig]
    all_texts_f = [d.abstract for d in fixture_all]
    orig_ids_f = [d.doc_id for d in fixture_orig]
    all_ids_f = [d.doc_id for d in fixture_all]

    vec_base_f = TextVectorizer().fit(orig_texts_f)
    c_b_f, z_b_f, x_b_f = vec_base_f.transform(orig_texts_f)

    # Refreshed IDF vectorizer
    vec_ref_f = TextVectorizer()
    vec_ref_f.feature_names = list(vec_base_f.feature_names)
    vec_ref_f.vocabulary = dict(vec_base_f.vocabulary)
    vec_ref_f.n_documents = len(all_texts_f)
    df_exp = np.zeros(len(vec_base_f.feature_names), dtype=np.int64)
    for t_doc in [vec_ref_f.tokenizer(d) for d in all_texts_f]:
        for term in set(t_doc):
            if term in vec_ref_f.vocabulary:
                df_exp[vec_ref_f.vocabulary[term]] += 1
    vec_ref_f.doc_frequencies = df_exp
    vec_ref_f.idf_weights = np.log((1.0 + vec_ref_f.n_documents) / (1.0 + df_exp)) + 1.0
    vec_ref_f.is_fitted = True

    # Controls
    col_preset, col_toggles = st.columns([1.5, 2.5])
    with col_preset:
        apply_preset = st.button("⭐ Apply Preset: Hide new papers, refresh IDF", use_container_width=True)
        if "shift_ref_toggle" not in st.session_state:
            st.session_state.shift_ref_toggle = True
        if "shift_inc_toggle" not in st.session_state:
            st.session_state.shift_inc_toggle = False

        if apply_preset:
            st.session_state.shift_ref_toggle = True
            st.session_state.shift_inc_toggle = False
            st.rerun()

    with col_toggles:
        tog1, tog2 = st.columns(2)
        with tog1:
            st.session_state.shift_inc_toggle = st.checkbox(
                "Include incoming papers in candidate results",
                value=st.session_state.shift_inc_toggle,
            )
        with tog2:
            st.session_state.shift_ref_toggle = st.checkbox(
                "Recompute IDF using expanded 10-document corpus statistics",
                value=st.session_state.shift_ref_toggle,
            )

    # Search executions
    ret_b = EvidenceRetrieval(vec_base_f)
    _, _, res_base = ret_b.search(
        DEFAULT_TEACHING_CLAIM,
        orig_ids_f,
        [d.title for d in fixture_orig],
        orig_texts_f,
        [d.annotation for d in fixture_orig],
        x_b_f,
        c_b_f,
    )

    active_shift_vec = vec_ref_f if st.session_state.shift_ref_toggle else vec_base_f
    cand_docs = fixture_all if st.session_state.shift_inc_toggle else fixture_orig
    cand_texts = [d.abstract for d in cand_docs]
    cand_ids = [d.doc_id for d in cand_docs]
    cand_titles = [d.title for d in cand_docs]
    cand_ann = [d.annotation for d in cand_docs]

    c_s, z_s, x_s = active_shift_vec.transform(cand_texts)
    ret_s = EvidenceRetrieval(active_shift_vec)
    _, _, res_shift = ret_s.search(
        DEFAULT_TEACHING_CLAIM,
        cand_ids,
        cand_titles,
        cand_texts,
        cand_ann,
        x_s,
        c_s,
    )

    col_t5_base, col_t5_exp = st.columns(2)
    with col_t5_base:
        st.markdown("#### 🏛️ Baseline Top 5 (N=7, Frozen IDF)")
        for r in res_base[:5]:
            is_target = (r.doc_id == "DOC-03")
            st.markdown(
                f"""
                <div style="border:1px solid {'#9333ea' if is_target else '#cbd5e1'};
                            background-color:{'#faf5ff' if is_target else '#ffffff'};
                            border-radius:6px; padding:8px 12px; margin-bottom:6px;">
                  <b>Rank #{r.rank} — {r.doc_id}</b> (Score: <code>{r.score:.4f}</code>)
                  {' <span style=\"color:#9333ea; font-weight:bold;\">★ FOCUS TARGET</span>' if is_target else ''}<br>
                  <span style="font-size:0.85rem; color:#475569;">{r.title}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_t5_exp:
        st.markdown(
            f"#### ⚡ Shifted Top 5 "
            f"({'Expanded Candidates' if st.session_state.shift_inc_toggle else 'Fixed 7 Candidates'}, "
            f"{'Refreshed IDF' if st.session_state.shift_ref_toggle else 'Frozen IDF'})"
        )
        base_map = {r.doc_id: r.rank for r in res_base}
        for r in res_shift[:5]:
            is_target = (r.doc_id == "DOC-03")
            b_r = base_map.get(r.doc_id)
            if b_r is None:
                mov = "<span style='color:#059669; font-weight:bold;'>[NEW CANDIDATE]</span>"
            elif b_r > r.rank:
                mov = f"<span style='color:#059669; font-weight:bold;'>[Rose {b_r - r.rank}]</span>"
            elif b_r < r.rank:
                mov = f"<span style='color:#dc2626; font-weight:bold;'>[Fell {r.rank - b_r}]</span>"
            else:
                mov = "<span style='color:#64748b;'>[Unchanged]</span>"

            st.markdown(
                f"""
                <div style="border:1px solid {'#9333ea' if is_target else '#cbd5e1'};
                            background-color:{'#faf5ff' if is_target else '#ffffff'};
                            border-radius:6px; padding:8px 12px; margin-bottom:6px;">
                  <b>Rank #{r.rank} — {r.doc_id}</b> (Score: <code>{r.score:.4f}</code>) {mov}<br>
                  <span style="font-size:0.85rem; color:#475569;">{r.title}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Demoted Target Paper Breakdown
    r_target_base = next((r for r in res_base if r.doc_id == "DOC-03"), None)
    r_target_shift = next((r for r in res_shift if r.doc_id == "DOC-03"), None)

    idx_cons = vec_base_f.vocabulary["consolidation"]
    df_b_c = int(vec_base_f.doc_frequencies[idx_cons])
    df_s_c = int(active_shift_vec.doc_frequencies[idx_cons])
    idf_b_c = float(vec_base_f.idf_weights[idx_cons])
    idf_s_c = float(active_shift_vec.idf_weights[idx_cons])

    st.markdown("---")
    st.markdown("#### 🔬 Detailed Mathematical Demotion Breakdown: `DOC-03`")

    col_tbl, col_expl = st.columns([1.5, 1.5])
    with col_tbl:
        df_demo = pd.DataFrame([
            {"Parameter": "Corpus Documents N", "Baseline": "7", "Shifted": f"{active_shift_vec.n_documents}", "Delta": f"+{active_shift_vec.n_documents-7}"},
            {"Parameter": "DF('consolidation')", "Baseline": f"{df_b_c}", "Shifted": f"{df_s_c}", "Delta": f"+{df_s_c-df_b_c} (diluted)"},
            {"Parameter": "IDF('consolidation')", "Baseline": f"{idf_b_c:.4f}", "Shifted": f"{idf_s_c:.4f}", "Delta": f"{idf_s_c-idf_b_c:.4f}"},
            {"Parameter": "DOC-03 Score", "Baseline": f"{r_target_base.score:.4f}", "Shifted": f"{r_target_shift.score:.4f}", "Delta": f"{r_target_shift.score-r_target_base.score:.4f}"},
            {"Parameter": "DOC-03 Standing", "Baseline": f"Rank #{r_target_base.rank} (in Top 5)", "Shifted": f"<b style='color:#dc2626;'>Rank #{r_target_shift.rank} (OUT OF TOP 5)</b>", "Delta": f"Fell {r_target_shift.rank-r_target_base.rank} place"},
        ])
        st.markdown(df_demo.to_html(classes="table", index=False, escape=False), unsafe_allow_html=True)

    with col_expl:
        st.markdown(
            f"""
            <div class="math-card">
              <div style="font-weight:700; margin-bottom:6px;">Why DOC-03 Dropped:</div>
              1. <b>IDF Dilution:</b> The incoming papers discuss unrelated consolidation (bone repair & fiscal budgets),
                 causing <code>DF('consolidation')</code> to jump from 4 to 7, reducing its IDF from <b>{idf_b_c:.4f}</b> to <b>{idf_s_c:.4f}</b>.<br>
              2. <b>Vector Normalization:</b> Because DOC-03 depends heavily on 'consolidation', its relative vector length drops.<br>
              3. <b>Overtaken by DOC-06:</b> DOC-06 contains 'stabilizes' and 'memory'—words whose IDF was not diluted—allowing DOC-06 to overtake DOC-03 and claim the #5 spot!
            </div>
            """,
            unsafe_allow_html=True,
        )
