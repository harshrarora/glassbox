"""Stage 4: Linear Algebra & Geometric Dimensionality Reduction.
3D PCA & SVD Rank-k snapping simulator and Multiclass Linear Discriminant Analysis (S_W / S_B).
"""

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from glassbox.fixtures.arxiv_corpus import get_arxiv_corpus
from glassbox.utils.visual_components import render_hero_card
from glassbox.core.geometry import fit_pca_3d, compute_svd_rank_k
from glassbox.core.classifiers import LDASimulator
from glassbox.core.vectorizer import TextVectorizer

CATEGORY_COLORS = {
    "cs.CV": "#10b981",    # Emerald
    "cs.CL": "#6366f1",    # Indigo
    "cs.LG": "#f59e0b",    # Amber
    "stat.ML": "#f43f5e",   # Rose
}


def render_stage4():
    render_hero_card(
        stage_title="Stage 4: Linear Algebra & Dimensionality Reduction",
        what_text="Compressing high-dimensional vocabulary spaces into printable 2D and 3D geometric coordinates using unsupervised PCA/SVD and supervised LDA.",
        why_text="1,000+ sparse dimensions cannot be plotted or easily modeled. Unsupervised PCA captures global variance, while supervised LDA contracts within-class scatter (S_W) and expands between-class distance (S_B) to maximize separability.",
    )

    papers = get_arxiv_corpus()
    all_abstracts = [p.abstract for p in papers]
    categories = [p.category for p in papers]

    vec = TextVectorizer().fit(all_abstracts)
    _, _, X_all = vec.transform(all_abstracts)

    tab_pca_svd, tab_lda = st.tabs([
        "🌐 PCA / SVD 3D Simulator & Rank-k Snapping",
        "⚖️ Linear Discriminant Analysis (S_W & S_B Scatter Matrices)",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: PCA / SVD 3D Simulator
    # -------------------------------------------------------------------------
    with tab_pca_svd:
        st.markdown("### 3D Geometric PCA & Truncated SVD Rank-k Snapping")
        st.caption("Inspect paper vectors across 3 distinctive domain features. Toggle rank k to watch vectors snap to lower-dimensional subspaces.")

        col_terms, col_svd = st.columns([2.0, 1.5])
        with col_terms:
            triplet_presets = {
                "Domain Triplet: ['image', 'language', 'optimization']": ["image", "language", "optimization"],
                "Architecture Triplet: ['convolutional', 'transformer', 'gradient']": ["convolutional", "transformer", "gradient"],
                "Theory Triplet: ['visual', 'translation', 'convex']": ["visual", "translation", "convex"],
            }
            chosen_preset = st.selectbox("Feature Coordinate Presets:", options=list(triplet_presets.keys()))
            pca_terms = triplet_presets[chosen_preset]

        with col_svd:
            rank_k = st.slider("Truncated SVD Rank k (Subspace Snapping):", min_value=1, max_value=3, value=2)

        idx_terms = [vec.vocabulary.get(t, 0) for t in pca_terms]
        X_3d = X_all[:, idx_terms]

        pca_res = fit_pca_3d(X_3d, pca_terms)
        svd_res = compute_svd_rank_k(X_3d, k=rank_k)

        col_canvas, col_diag = st.columns([1.7, 1.3])

        with col_canvas:
            fig_3d = go.Figure()
            pts_display = svd_res.X_reconstructed if rank_k < 3 else X_3d

            for cat in ["cs.CV", "cs.CL", "cs.LG", "stat.ML"]:
                mask = np.array([c == cat for c in categories])
                pts_cat = pts_display[mask]
                ids_cat = [p.id for i, p in enumerate(papers) if mask[i]]
                titles_cat = [p.title for i, p in enumerate(papers) if mask[i]]

                fig_3d.add_trace(
                    go.Scatter3d(
                        x=pts_cat[:, 0],
                        y=pts_cat[:, 1],
                        z=pts_cat[:, 2],
                        mode="markers",
                        marker=dict(size=7, color=CATEGORY_COLORS[cat]),
                        text=[f"<b>{pid}</b>: {t}" for pid, t in zip(ids_cat, titles_cat)],
                        name=f"{cat}",
                    )
                )

            # Draw Centroid Mean μ
            mu = pca_res.mean_vector
            fig_3d.add_trace(
                go.Scatter3d(
                    x=[mu[0]], y=[mu[1]], z=[mu[2]],
                    mode="markers+text",
                    marker=dict(size=10, color="#f59e0b", symbol="diamond"),
                    text=["Centroid μ"],
                    name="Global Mean μ",
                )
            )

            # Draw PC1 and PC2 axes
            v1 = pca_res.eigenvectors[:, 0] * 0.2
            v2 = pca_res.eigenvectors[:, 1] * 0.2
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
                    xaxis=dict(title=f"'{pca_terms[0]}'"),
                    yaxis=dict(title=f"'{pca_terms[1]}'"),
                    zaxis=dict(title=f"'{pca_terms[2]}'"),
                ),
                height=480,
                margin=dict(l=0, r=0, b=0, t=30),
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        with col_diag:
            st.markdown("#### 📐 Scree Plot & Covariance Heatmap")
            # Scree Plot
            cum_var = np.cumsum(pca_res.explained_variance_ratio) * 100
            fig_scree = go.Figure()
            fig_scree.add_trace(go.Bar(x=["PC1", "PC2", "PC3"], y=pca_res.explained_variance_ratio * 100, name="Individual %", marker_color="#3b82f6"))
            fig_scree.add_trace(go.Scatter(x=["PC1", "PC2", "PC3"], y=cum_var, name="Cumulative %", line=dict(color="#f59e0b", width=3)))
            fig_scree.update_layout(title="Scree Plot: Explained Variance Ratio", height=230, margin=dict(l=20, r=20, b=20, t=35))
            st.plotly_chart(fig_scree, use_container_width=True)

            # Covariance Matrix Heatmap
            fig_cov = px.imshow(
                pca_res.covariance_matrix,
                x=pca_terms,
                y=pca_terms,
                color_continuous_scale="Blues",
                title="Sample Covariance Matrix Σ",
            )
            fig_cov.update_layout(height=210, margin=dict(l=20, r=20, b=20, t=35))
            st.plotly_chart(fig_cov, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: Multiclass LDA Simulator
    # -------------------------------------------------------------------------
    with tab_lda:
        st.markdown("### Multiclass Scatter Matrices: $S_W$ vs. $S_B$")
        st.caption("LDA maximizes between-class scatter while minimizing within-class variance: J(w) = (wᵀ S_B w) / (wᵀ S_W w).")

        lda_res = LDASimulator.fit(X_3d, categories)

        col_sw, col_sb = st.columns(2)
        with col_sw:
            fig_sw = px.imshow(
                lda_res.S_W,
                x=pca_terms,
                y=pca_terms,
                color_continuous_scale="Reds",
                title="Within-Class Scatter Matrix S_W = ∑_c ∑ (x - μ_c)(x - μ_c)ᵀ",
            )
            fig_sw.update_layout(height=260, margin=dict(l=20, r=20, b=20, t=35))
            st.plotly_chart(fig_sw, use_container_width=True)

        with col_sb:
            fig_sb = px.imshow(
                lda_res.S_B,
                x=pca_terms,
                y=pca_terms,
                color_continuous_scale="Greens",
                title="Between-Class Scatter Matrix S_B = ∑_c N_c (μ_c - μ)(μ_c - μ)ᵀ",
            )
            fig_sb.update_layout(height=260, margin=dict(l=20, r=20, b=20, t=35))
            st.plotly_chart(fig_sb, use_container_width=True)

        # 2D Projection Comparison: PCA vs LDA
        col_lda_proj, col_pca_proj = st.columns(2)
        with col_lda_proj:
            fig_lda_2d = go.Figure()
            for cat in ["cs.CV", "cs.CL", "cs.LG", "stat.ML"]:
                mask = np.array([c == cat for c in categories])
                pts = lda_res.projected_2d[mask]
                fig_lda_2d.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", marker=dict(size=9, color=CATEGORY_COLORS[cat]), name=cat))
            fig_lda_2d.update_layout(
                title=f"LDA Projection: Maximum Class Separability (λ₁={lda_res.eigenvalues[0]:.2f}, λ₂={lda_res.eigenvalues[1]:.2f})",
                xaxis_title="Discriminant Axis 1",
                yaxis_title="Discriminant Axis 2",
                height=380,
                margin=dict(l=20, r=20, b=20, t=35),
            )
            st.plotly_chart(fig_lda_2d, use_container_width=True)

        with col_pca_proj:
            fig_pca_2d = go.Figure()
            for cat in ["cs.CV", "cs.CL", "cs.LG", "stat.ML"]:
                mask = np.array([c == cat for c in categories])
                pts = pca_res.scores_2d[mask]
                fig_pca_2d.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", marker=dict(size=9, color=CATEGORY_COLORS[cat]), name=cat))
            fig_pca_2d.update_layout(
                title="PCA Projection: Total Variance Maximization (Unsupervised)",
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2",
                height=380,
                margin=dict(l=20, r=20, b=20, t=35),
            )
            st.plotly_chart(fig_pca_2d, use_container_width=True)
