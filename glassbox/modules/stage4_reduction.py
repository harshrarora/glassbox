"""Stage 4: Linear Algebra & Geometric Dimensionality Reduction.
3D PCA & SVD Rank-k snapping simulator and Multiclass Linear Discriminant Analysis (S_W / S_B).
"""

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from glassbox.data.arxiv_corpus import get_arxiv_corpus
from glassbox.utils.visual_components import render_hero_card
from glassbox.core.geometry import fit_pca_3d, compute_svd_rank_k
from glassbox.core.classifiers import LDASimulator
from glassbox.core.vectorizer import TextVectorizer
from glassbox.utils.math_engine import (
    simulate_kmeans,
    compute_elbow_and_silhouette,
    compute_hac_linkage,
)

CATEGORY_COLORS = {
    "cs.CV": "#10b981",    # Emerald
    "cs.CL": "#6366f1",    # Indigo
    "cs.LG": "#f59e0b",    # Amber
    "stat.ML": "#f43f5e",   # Rose
}


def _render_stage4_playground():
    papers = get_arxiv_corpus()
    all_abstracts = [p.abstract for p in papers]
    categories = [p.category for p in papers]

    from glassbox.utils.pipeline import vectorizer, settings, coordinates, resolve_terms
    vec = vectorizer().fit(all_abstracts)
    _, _, X_all = vec.transform(all_abstracts)

    tab_pca_svd, tab_lda, tab_cluster = st.tabs([
        "PCA / SVD",
        "Fisher LDA",
        "Clustering",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: PCA / SVD 3D Simulator
    # -------------------------------------------------------------------------
    with tab_pca_svd:
        col_terms, col_svd = st.columns([2.0, 1.5])
        with col_terms:
            triplet_presets = {
                "Domain Triplet: ['image', 'language', 'optimization']": ["image", "language", "optimization"],
                "Architecture Triplet: ['convolutional', 'transformer', 'gradient']": ["convolutional", "transformer", "gradient"],
                "Theory Triplet: ['visual', 'translation', 'convex']": ["visual", "translation", "convex"],
            }
            cfg = settings()
            default_preset = list(triplet_presets.values()).index(cfg["terms"]) if cfg["terms"] in triplet_presets.values() else 0
            chosen_preset = st.selectbox("Feature Coordinate Presets:", options=list(triplet_presets.keys()), index=default_preset)
            pca_terms = triplet_presets[chosen_preset]
            cfg["terms"] = pca_terms

        with col_svd:
            rank_k = st.slider("Truncated SVD Rank k (Subspace Snapping):", min_value=1, max_value=3, value=settings()["rank"])
            settings()["rank"] = rank_k

        pca_terms = resolve_terms(vec, pca_terms)
        X_3d = coordinates(vec, X_all, pca_terms)
        missing = [t for t in pca_terms if t not in vec.vocabulary]
        if missing:
            st.warning("Absent after preprocessing (shown as zero columns): " + ", ".join(missing))

        pca_res = fit_pca_3d(X_3d, pca_terms)
        # Center before SVD so the snapping plane and displayed PCA axes agree.
        svd_res = compute_svd_rank_k(pca_res.X_centered, k=rank_k)
        st.caption("SVD acts on X − μ; reconstruction adds μ back. Rank 2 therefore lies on the same affine plane as the displayed PCA axes.")

        col_canvas, col_diag = st.columns([1.7, 1.3])

        with col_canvas:
            fig_3d = go.Figure()
            pts_display = svd_res.X_reconstructed + pca_res.mean_vector if rank_k < 3 else X_3d

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
            st.plotly_chart(fig_3d, width="stretch", theme=None)

        with col_diag:
            retained, residual = st.columns(2)
            retained.metric("Energy retained", f"{svd_res.energy_captured:.1%}")
            residual.metric("Residual ‖X − X̂‖F", f"{svd_res.frobenius_error:.6f}")
            # Scree Plot
            cum_var = np.cumsum(pca_res.explained_variance_ratio) * 100
            fig_scree = go.Figure()
            fig_scree.add_trace(go.Bar(x=["PC1", "PC2", "PC3"], y=pca_res.explained_variance_ratio * 100, name="Individual %", marker_color="#3b82f6"))
            fig_scree.add_trace(go.Scatter(x=["PC1", "PC2", "PC3"], y=cum_var, name="Cumulative %", line=dict(color="#f59e0b", width=3)))
            fig_scree.update_layout(title="Scree Plot: Explained Variance Ratio", height=230, margin=dict(l=20, r=20, b=20, t=35))
            st.plotly_chart(fig_scree, width="stretch", theme=None)

            # Covariance Matrix Heatmap
            fig_cov = px.imshow(
                pca_res.covariance_matrix,
                x=pca_terms,
                y=pca_terms,
                color_continuous_scale="Blues",
                title="Sample Covariance Matrix Σ",
            )
            fig_cov.update_layout(height=210, margin=dict(l=20, r=20, b=20, t=35))
            st.plotly_chart(fig_cov, width="stretch", theme=None)

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
            st.plotly_chart(fig_sw, width="stretch", theme=None)

        with col_sb:
            fig_sb = px.imshow(
                lda_res.S_B,
                x=pca_terms,
                y=pca_terms,
                color_continuous_scale="Greens",
                title="Between-Class Scatter Matrix S_B = ∑_c N_c (μ_c - μ)(μ_c - μ)ᵀ",
            )
            fig_sb.update_layout(height=260, margin=dict(l=20, r=20, b=20, t=35))
            st.plotly_chart(fig_sb, width="stretch", theme=None)

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
            st.plotly_chart(fig_lda_2d, width="stretch", theme=None)

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
            st.plotly_chart(fig_pca_2d, width="stretch", theme=None)

    # -------------------------------------------------------------------------
    # TAB 3: Unsupervised Clustering & Dendrogram Explorer
    # -------------------------------------------------------------------------
    with tab_cluster:
        st.markdown("###  Unsupervised Clustering & Hierarchical Dendrogram Explorer")
        st.caption("Watch centroid updates and hierarchical merges. The representation determines which distances the clustering algorithm sees.")

        cluster_subview = st.radio(
            "Clustering Analysis Sub-Module:",
            options=["1. K-Means Centroid Migration Simulation & Elbow Curve", "2. Hierarchical Agglomerative Clustering (HAC) & Dendrogram"],
            horizontal=True,
        )

        representation = st.radio("Coordinates to cluster", ["PCA (no labels)", "LDA (uses domain labels)"],
                                  horizontal=True, key="cluster_representation")
        pts_2d = pca_res.scores_2d if representation.startswith("PCA") else lda_res.projected_2d
        if representation.startswith("LDA"):
            st.warning("LDA has already used the domain labels. Clusters in this space are not independent evidence of unsupervised domain discovery.")
        else:
            st.caption("PCA and clustering do not use domain labels. Only three selected term coordinates enter this experiment.")

        if cluster_subview.startswith("1."):
            col_k_ctrl, col_iter_ctrl = st.columns([1.5, 2.0])
            with col_k_ctrl:
                k_val = st.slider("Number of Clusters K:", min_value=2, max_value=6, value=4)

            res_km = simulate_kmeans(pts_2d, k=k_val, max_iter=20, random_state=42)
            max_step = res_km["converged_iter"]

            with col_iter_ctrl:
                step_t = st.slider("Centroid Migration Step t:", min_value=0, max_value=max_step, value=max_step)

            cur_history = res_km["history"][step_t]
            cur_centroids = cur_history["centroids"]
            cur_labels = cur_history["labels"]
            cur_wcss = cur_history["wcss"]

            fig_km = go.Figure()
            cluster_palette = ["#10b981", "#6366f1", "#f59e0b", "#f43f5e", "#06b6d4", "#8b5cf6"]

            for c_idx in range(k_val):
                c_mask = (cur_labels == c_idx)
                pts_c = pts_2d[c_mask]
                ids_c = [papers[i].id for i in range(len(papers)) if c_mask[i]]
                titles_c = [papers[i].title for i in range(len(papers)) if c_mask[i]]
                domains_c = [papers[i].category for i in range(len(papers)) if c_mask[i]]

                fig_km.add_trace(go.Scatter(
                    x=pts_c[:, 0],
                    y=pts_c[:, 1],
                    mode="markers",
                    marker=dict(size=8, color=cluster_palette[c_idx % len(cluster_palette)]),
                    text=[f"<b>{pid}</b> [{cat}]: {t}" for pid, cat, t in zip(ids_c, domains_c, titles_c)],
                    name=f"Cluster {c_idx + 1} (n={len(pts_c)})",
                ))

                traj = np.array([res_km["history"][s]["centroids"][c_idx] for s in range(step_t + 1)])
                if len(traj) > 1:
                    fig_km.add_trace(go.Scatter(
                        x=traj[:, 0],
                        y=traj[:, 1],
                        mode="lines",
                        line=dict(color=cluster_palette[c_idx % len(cluster_palette)], width=2, dash="dot"),
                        showlegend=False,
                    ))

                fig_km.add_trace(go.Scatter(
                    x=[cur_centroids[c_idx, 0]],
                    y=[cur_centroids[c_idx, 1]],
                    mode="markers+text",
                    marker=dict(size=16, symbol="star-diamond", color=cluster_palette[c_idx % len(cluster_palette)], line=dict(color="#0f172a", width=2)),
                    text=[f"μ_{c_idx+1}"],
                    textposition="top center",
                    textfont=dict(size=12, color="#0f172a"),
                    name=f"Centroid μ_{c_idx + 1}",
                ))

            fig_km.update_layout(
                title=f"K-Means (K={k_val}) at Step t={step_t}/{max_step} | WCSS = {cur_wcss:.2f}",
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                height=430,
                margin=dict(l=20, r=20, b=20, t=35),
            )

            col_km_plot, col_km_stats = st.columns([2.0, 1.2])
            with col_km_plot:
                st.plotly_chart(fig_km, width="stretch", theme=None)
            with col_km_stats:
                st.markdown("####  Centroid Metrics & Convergence")
                status_txt = ("Converged" if res_km["converged"] else "Iteration limit reached") if step_t >= max_step else f"Migrating (Step {step_t}/{max_step})"
                st.markdown(
                    f"""
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px; margin-bottom:12px;">
                      <b>Algorithm Status:</b> {status_txt}<br>
                      <b>Inertia / WCSS:</b> <code>{cur_wcss:.4f}</code><br>
                      <b>Convergence Iteration:</b> t = {max_step}<br>
                      <b>Features Clustered:</b> 2D Projected Latent Vectors
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    """
                    <div style="font-size:0.85rem; color:#475569; line-height:1.4;">
                      <b>K-Means Update Formula:</b><br>
                      1. <i>Assignment:</i> c_i = argmin_k ||x_i - μ_k||²<br>
                      2. <i>Update:</i> μ_k = (1/|C_k|) ∑_{i ∈ C_k} x_i<br>
                      3. Repeat until ||μ_k^(t) - μ_k^(t-1)|| &lt; 10⁻⁴.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            show_elbow = st.checkbox(" Show Elbow Method & Silhouette Curve Analysis (K = 2..8)", value=True)
            if show_elbow:
                elbow_data = compute_elbow_and_silhouette(pts_2d, k_range=list(range(2, 9)))
                fig_elbow = go.Figure()

                fig_elbow.add_trace(go.Scatter(
                    x=elbow_data["k_values"],
                    y=elbow_data["wcss"],
                    mode="lines+markers",
                    name="WCSS (Inertia)",
                    line=dict(color="#6366f1", width=3),
                    marker=dict(size=8),
                    yaxis="y1",
                ))

                fig_elbow.add_trace(go.Scatter(
                    x=elbow_data["k_values"],
                    y=elbow_data["silhouettes"],
                    mode="lines+markers",
                    name="Silhouette Score",
                    line=dict(color="#10b981", width=3, dash="dash"),
                    marker=dict(size=8),
                    yaxis="y2",
                ))

                fig_elbow.update_layout(
                    title="The Elbow Method (WCSS) & Silhouette Coefficient Across K",
                    xaxis=dict(title="Number of Clusters K", dtick=1),
                    yaxis=dict(title=dict(text="Within-Cluster Sum of Squares (WCSS)", font=dict(color="#6366f1"))),
                    yaxis2=dict(title=dict(text="Silhouette Coefficient", font=dict(color="#10b981")), overlaying="y", side="right"),
                    legend=dict(x=0.65, y=0.95),
                    height=350,
                    margin=dict(l=20, r=20, b=20, t=35),
                )
                st.plotly_chart(fig_elbow, width="stretch", theme=None)
                valid = [(k, s) for k, s in zip(elbow_data["k_values"], elbow_data["silhouettes"]) if np.isfinite(s)]
                if valid:
                    best_k, best_score = max(valid, key=lambda item: item[1])
                    st.info(f"Highest measured silhouette in this sweep: K = {best_k} ({best_score:.3f}). This depends on the coordinates and initialization; it does not establish a true number of domains. Singleton samples contribute zero.")
                else:
                    st.info("Silhouette is undefined: a valid partition needs between 2 and n−1 occupied clusters.")

        else:
            st.markdown("####  Hierarchical Agglomerative Clustering (HAC) Tree")
            st.caption("Agglomerative bottom-up clustering merges closest pairs iteratively. Inspect how different linkage criteria alter the tree hierarchy.")

            col_link, col_cut = st.columns([1.5, 1.5])
            with col_link:
                link_crit = st.selectbox(
                    "Linkage Criterion:",
                    options=["ward", "complete", "average"],
                    format_func=lambda s: {
                        "ward": "Ward (Minimizes within-cluster variance)",
                        "complete": "Complete (Maximum pairwise distance)",
                        "average": "Average (Mean pairwise distance)",
                    }[s],
                )

            rep_indices = [
                0, 1, 2, 3,       # cs.CV: ResNet, ViT, YOLO, NeRF
                16, 17, 18, 19,   # cs.CL: Attention, BERT, GPT-3, RoBERTa
                32, 33, 34, 35,   # cs.LG: Adam, BatchNorm, Dropout, Double Descent
                48, 49, 50, 51,   # stat.ML: Bottou SGD, FISTA, PAC-Bayes, Stein Variational
            ]
            sub_pts = pts_2d[rep_indices]
            sub_labels = [f"[{papers[i].category}] {papers[i].title[:22]}..." for i in rep_indices]

            Z = compute_hac_linkage(sub_pts, method=link_crit)

            fig_dend, ax_dend = plt.subplots(figsize=(10, 4.5), dpi=150)
            dendrogram(
                Z,
                labels=sub_labels,
                orientation="top",
                leaf_rotation=45,
                leaf_font_size=9,
                ax=ax_dend,
                color_threshold=0.7 * np.max(Z[:, 2]),
            )
            ax_dend.set_title(f"HAC Dendrogram (Linkage: {link_crit.upper()}) on 16 Landmark arXiv Papers", fontsize=11, fontweight="bold")
            ax_dend.set_ylabel("Merge Distance / Dissimilarity", fontsize=9)
            ax_dend.grid(axis="y", linestyle="--", alpha=0.4)
            plt.tight_layout()

            st.pyplot(fig_dend)
            plt.close(fig_dend)

            st.markdown(
                """
                <div style="background:#f1f5f9; border-left:4px solid #6366f1; padding:10px; border-radius:4px; font-size:0.85rem; color:#334155;">
                  <b>Linkage Formulas:</b><br>
                  • <b>Ward's Linkage:</b> d(A, B) = √[ (2|A||B| / (|A|+|B|)) ||μ_A - μ_B||² ] (minimizes total intra-cluster variance)<br>
                  • <b>Complete Linkage:</b> d(A, B) = max_{x ∈ A, y ∈ B} ||x - y|| (produces compact, spherical clusters)<br>
                  • <b>Average Linkage:</b> d(A, B) = (1 / (|A||B|)) ∑_{x ∈ A} ∑_{y ∈ B} ||x - y|| (robust to noise)
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_stage4():
    from glassbox.utils.pipeline import lens_selector
    from glassbox.modules.comparisons import render_comparison
    from glassbox.modules.investigations import render_rotation
    lens = lens_selector(4, ["Playground", "Compare implementations", "Failure spotlight"])
    if lens == "Failure spotlight":
        render_rotation()
    elif lens == "Compare implementations":
        render_comparison(4)
    else:
        _render_stage4_playground()
