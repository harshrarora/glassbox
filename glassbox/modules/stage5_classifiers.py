"""Stage 5: The Supervised Classifier Arena.
Interactive models: SVM Margin Explorer, Decision Tree Gini Splits, KNN, and Naive Bayes.
"""

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from glassbox.fixtures.arxiv_corpus import get_arxiv_corpus
from glassbox.utils.visual_components import render_hero_card
from glassbox.core.classifiers import LDASimulator, SVMSimulator
from glassbox.core.vectorizer import TextVectorizer
from glassbox.utils.math_engine import (
    compute_decision_tree_split,
    compute_knn_neighbors,
    compute_naive_bayes_posteriors,
)

CATEGORY_COLORS = {
    "cs.CV": "#10b981",    # Emerald
    "cs.CL": "#6366f1",    # Indigo
    "cs.LG": "#f59e0b",    # Amber
    "stat.ML": "#f43f5e",   # Rose
}


def render_stage5():
    render_hero_card(
        stage_title="Stage 5: The Supervised Classifier Arena",
        what_text="Fitting and comparing four foundational syllabus classifiers—SVM, Decision Trees, KNN, and Naive Bayes—on our scientific paper representations.",
        why_text="Classifiers impose distinct inductive biases on data: SVM optimizes geometric margins, Decision Trees perform recursive orthogonal cuts, KNN estimates local neighborhood density, and Naive Bayes computes joint conditional probabilities.",
    )

    papers = get_arxiv_corpus()
    all_abstracts = [p.abstract for p in papers]
    categories = [p.category for p in papers]

    vec = TextVectorizer().fit(all_abstracts)
    _, _, X_all = vec.transform(all_abstracts)

    # Use 2D LDA projection for sharp geometric visualization
    terms = ["image", "language", "optimization"]
    idx_terms = [vec.vocabulary.get(t, 0) for t in terms]
    lda_res = LDASimulator.fit(X_all[:, idx_terms], categories)
    X_2d = lda_res.projected_2d

    model_choice = st.radio(
        "Select Supervised Classifier to Inspect:",
        options=[
            "1. Support Vector Machine (SVM) Margin & Gutters",
            "2. Decision Tree Axis-Aligned Split Explorer",
            "3. K-Nearest Neighbors (KNN) Metric Ball",
            "4. Naive Bayes Conditional Probability Inspector",
        ],
        horizontal=True,
    )

    # -------------------------------------------------------------------------
    # 1. Support Vector Machine
    # -------------------------------------------------------------------------
    if model_choice.startswith("1."):
        st.markdown("### ⚔️ Support Vector Machine (SVM) Margin Explorer")
        st.caption("Inspect the separating hyperplane and margin gutters. Adjust C to observe the soft vs. hard margin tradeoff.")

        col_pair, col_c = st.columns([1.5, 1.5])
        with col_pair:
            pair = st.selectbox(
                "Binary Classification Task:",
                options=[("cs.CV", "cs.CL"), ("cs.CV", "stat.ML"), ("cs.CL", "stat.ML")],
                format_func=lambda p: f"{p[0]} (+1) vs. {p[1]} (-1)",
            )
            cat_pos, cat_neg = pair
        with col_c:
            c_val = st.slider("Regularization Penalty C:", min_value=0.01, max_value=20.0, value=1.0, step=0.1)

        mask = np.array([c in [cat_pos, cat_neg] for c in categories])
        sub_papers = [p for i, p in enumerate(papers) if mask[i]]
        X_sub = X_2d[mask]
        y_sub = np.array([1 if p.category == cat_pos else -1 for p in sub_papers])

        svm_res = SVMSimulator.fit(X_sub, y_sub, C=c_val)

        col_plot, col_info = st.columns([1.8, 1.2])
        with col_plot:
            fig_svm = go.Figure()
            # Hyperplane and gutters
            fig_svm.add_trace(go.Scatter(x=svm_res.line_x, y=svm_res.line_boundary, mode="lines", line=dict(color="#0f172a", width=3), name="wᵀx + b = 0"))
            fig_svm.add_trace(go.Scatter(x=svm_res.line_x, y=svm_res.line_gutter_pos, mode="lines", line=dict(color="#10b981", width=2, dash="dash"), name="wᵀx + b = +1"))
            fig_svm.add_trace(go.Scatter(x=svm_res.line_x, y=svm_res.line_gutter_neg, mode="lines", line=dict(color="#6366f1", width=2, dash="dash"), name="wᵀx + b = -1"))

            # Data points
            for cat, label_val, color in [(cat_pos, 1, CATEGORY_COLORS[cat_pos]), (cat_neg, -1, CATEGORY_COLORS[cat_neg])]:
                cat_mask = (y_sub == label_val)
                fig_svm.add_trace(go.Scatter(x=X_sub[cat_mask, 0], y=X_sub[cat_mask, 1], mode="markers", marker=dict(size=9, color=color), name=cat))

            # Support Vectors
            sv_pts = X_sub[svm_res.support_vector_indices]
            fig_svm.add_trace(go.Scatter(x=sv_pts[:, 0], y=sv_pts[:, 1], mode="markers", marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color="#f59e0b", width=3)), name=f"Support Vectors ({len(svm_res.support_vector_indices)})"))

            y_span = np.max(X_sub[:, 1]) - np.min(X_sub[:, 1])
            fig_svm.update_layout(
                title=f"SVM Margin (C={c_val:.2f}): Width γ = {svm_res.margin_width:.4f} | Accuracy = {svm_res.accuracy*100:.1f}%",
                xaxis_title="Discriminant Feature 1",
                yaxis_title="Discriminant Feature 2",
                yaxis=dict(range=[np.min(X_sub[:, 1]) - 0.4*y_span, np.max(X_sub[:, 1]) + 0.4*y_span]),
                height=420,
                margin=dict(l=20, r=20, b=20, t=35),
            )
            st.plotly_chart(fig_svm, use_container_width=True)

        with col_info:
            st.markdown(
                f"""
                <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; padding:16px;">
                  <b>1. Hyperplane Equation:</b><br>
                  <code>{svm_res.w[0]:.4f} x₁ + {svm_res.w[1]:.4f} x₂ + {svm_res.b:.4f} = 0</code><br><br>
                  <b>2. Margin Width γ:</b><br>
                  <code>γ = 2 / ||w||₂ = 2 / {np.linalg.norm(svm_res.w):.4f} = {svm_res.margin_width:.4f}</code><br><br>
                  <b>3. Support Vectors:</b><br>
                  <b>{len(svm_res.support_vector_indices)}</b> critical samples define the boundary.<br><br>
                  <b>4. Classification Accuracy:</b> <b>{svm_res.accuracy*100:.1f}%</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # 2. Decision Tree Split Visualizer
    # -------------------------------------------------------------------------
    elif model_choice.startswith("2."):
        st.markdown("### 🌲 Decision Tree Axis-Aligned Split Explorer")
        st.caption("Decision trees recursively partition feature space with orthogonal cuts. Move the threshold slider to see exact Gini Impurity and Information Gain.")

        split_axis = st.radio("Partitioning Feature Axis:", options=["Feature 1 (Horizontal Split)", "Feature 2 (Vertical Split)"], horizontal=True)
        feat_idx = 0 if "Feature 1" in split_axis else 1

        val_min, val_max = float(np.min(X_2d[:, feat_idx])), float(np.max(X_2d[:, feat_idx]))
        thresh = st.slider("Split Threshold Value:", min_value=val_min, max_value=val_max, value=float(np.mean(X_2d[:, feat_idx])), step=0.05)

        split_stats = compute_decision_tree_split(X_2d[:, feat_idx], np.array(categories), threshold=thresh)

        col_tree_plot, col_tree_math = st.columns([1.8, 1.2])
        with col_tree_plot:
            fig_dt = go.Figure()
            for cat in ["cs.CV", "cs.CL", "cs.LG", "stat.ML"]:
                mask = np.array([c == cat for c in categories])
                fig_dt.add_trace(go.Scatter(x=X_2d[mask, 0], y=X_2d[mask, 1], mode="markers", marker=dict(size=9, color=CATEGORY_COLORS[cat]), name=cat))

            # Draw split line
            if feat_idx == 0:
                fig_dt.add_vline(x=thresh, line_width=3, line_dash="dash", line_color="#ef4444")
            else:
                fig_dt.add_hline(y=thresh, line_width=3, line_dash="dash", line_color="#ef4444")

            fig_dt.update_layout(title="Orthogonal Decision Cut across 4 Domains", height=420, margin=dict(l=20, r=20, b=20, t=35))
            st.plotly_chart(fig_dt, use_container_width=True)

        with col_tree_math:
            st.markdown(
                f"""
                <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; padding:16px;">
                  <b>1. Parent Node Impurity:</b><br>
                  • Parent Entropy: <code>H(S) = {split_stats['parent_entropy']:.4f} bits</code><br>
                  • Parent Gini: <code>Gini(S) = {split_stats['parent_gini']:.4f}</code><br><br>
                  <b>2. Children Node Partitions:</b><br>
                  • Left / Below: <b>{split_stats['left_count']} papers</b><br>
                  • Right / Above: <b>{split_stats['right_count']} papers</b><br><br>
                  <b>3. Information Gain (ΔH):</b><br>
                  <div style="font-family:monospace; background:#fff; padding:6px; border:1px solid #e2e8f0; border-radius:4px; margin-top:4px;">
                    Gain = H(Parent) - ∑ (|S_v|/|S|) H(S_v)<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <b>{split_stats['info_gain']:.4f} bits</b>
                  </div><br>
                  <b>4. Weighted Split Gini:</b> <code>{split_stats['split_gini']:.4f}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # 3. K-Nearest Neighbors
    # -------------------------------------------------------------------------
    elif model_choice.startswith("3."):
        st.markdown("### 🎯 K-Nearest Neighbors (KNN) Metric Ball")
        st.caption("Place a query coordinate and observe the expanding Euclidean metric ball selecting K neighbors.")

        col_q1, col_q2, col_k = st.columns(3)
        with col_q1:
            q_x = st.slider("Query X Coordinate:", float(np.min(X_2d[:, 0])), float(np.max(X_2d[:, 0])), 0.0, step=0.05)
        with col_q2:
            q_y = st.slider("Query Y Coordinate:", float(np.min(X_2d[:, 1])), float(np.max(X_2d[:, 1])), 0.0, step=0.05)
        with col_k:
            k_val = st.selectbox("Neighborhood Size K:", options=[1, 3, 5, 9, 15], index=2)

        knn_res = compute_knn_neighbors(X_2d, np.array(categories), np.array([q_x, q_y]), k=k_val)

        col_knn_plot, col_knn_info = st.columns([1.8, 1.2])
        with col_knn_plot:
            fig_knn = go.Figure()
            for cat in ["cs.CV", "cs.CL", "cs.LG", "stat.ML"]:
                mask = np.array([c == cat for c in categories])
                fig_knn.add_trace(go.Scatter(x=X_2d[mask, 0], y=X_2d[mask, 1], mode="markers", marker=dict(size=9, color=CATEGORY_COLORS[cat]), name=cat))

            # Query Point
            fig_knn.add_trace(go.Scatter(x=[q_x], y=[q_y], mode="markers", marker=dict(size=14, color="#000", symbol="star"), name="Query Point"))

            # Highlight Nearest Neighbors
            k_pts = X_2d[knn_res["indices"]]
            fig_knn.add_trace(go.Scatter(x=k_pts[:, 0], y=k_pts[:, 1], mode="markers", marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color="#f59e0b", width=3)), name=f"K={k_val} Neighbors"))

            # Circle for metric ball
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = q_x + knn_res["radius"] * np.cos(theta)
            circle_y = q_y + knn_res["radius"] * np.sin(theta)
            fig_knn.add_trace(go.Scatter(x=circle_x, y=circle_y, mode="lines", line=dict(color="#f59e0b", dash="dot"), name=f"Radius R={knn_res['radius']:.3f}"))

            fig_knn.update_layout(title=f"KNN Prediction: {knn_res['prediction']}", height=420, margin=dict(l=20, r=20, b=20, t=35))
            st.plotly_chart(fig_knn, use_container_width=True)

        with col_knn_info:
            st.markdown(
                f"""
                <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; padding:16px;">
                  <b>1. Majority Vote Winner:</b><br>
                  <span style="font-size:1.2rem; font-weight:800; color:{CATEGORY_COLORS[knn_res['prediction']]};">
                    {knn_res['prediction']}
                  </span><br><br>
                  <b>2. Neighbor Vote Breakdown:</b>
                  {knn_res['vote_distribution']}<br><br>
                  <b>3. Metric Ball Radius:</b> <code>R = {knn_res['radius']:.4f}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # 4. Naive Bayes
    # -------------------------------------------------------------------------
    elif model_choice.startswith("4."):
        st.markdown("### 🎲 Naive Bayes Probability Inspector")
        st.caption("Multinomial Naive Bayes multiplying term likelihoods with Laplace smoothing: P(C|D) ∝ P(C) ∏ P(w_i|C).")

        c_mat, vocab_nb = vec.transform_raw_counts(all_abstracts), vec.feature_names
        test_phrase = st.text_input("Enter a scientific query phrase:", value="convolutional neural network for image recognition")

        tokens_nb = [vec.vocabulary[t] for t in vec.tokenizer(test_phrase) if t in vec.vocabulary]
        nb_res = compute_naive_bayes_posteriors(c_mat, np.array(categories), tokens_nb, sorted(list(set(categories))))

        st.markdown(f"**Predicted Category:** `{nb_res['predicted_class']}`")
        df_nb = pd.DataFrame([{"Category": c, "Posterior Probability P(C|D)": f"{p*100:.2f}%"} for c, p in nb_res["posterior_probs"].items()])
        st.dataframe(df_nb, use_container_width=True)
