"""Stage 6: Model MRI & Failure Debugger (Influence Functions).
Traces test predictions and misclassifications back to training examples
using Koh & Liang's empirical influence functions (ICML 2017).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from glassbox.fixtures.arxiv_corpus import get_arxiv_corpus
from glassbox.utils.visual_components import render_hero_card
from glassbox.core.classifiers import LDASimulator, SVMSimulator
from glassbox.core.vectorizer import TextVectorizer
from glassbox.utils.math_engine import compute_influence_functions_2d

CATEGORY_COLORS = {
    "cs.CV": "#10b981",    # Emerald
    "cs.CL": "#6366f1",    # Indigo
}


def render_stage6():
    render_hero_card(
        stage_title="Stage 6: Model MRI & Failure Debugger",
        what_text="Tracing individual model predictions back to the specific training data points that shaped the decision boundary using Koh & Liang's (ICML 2017) influence functions.",
        why_text="When a model makes an error or accepts a shortcut, we must identify which training examples exerted the highest mathematical torque on the loss landscape, enabling targeted data debugging.",
    )

    papers = get_arxiv_corpus()
    all_abstracts = [p.abstract for p in papers]
    categories = [p.category for p in papers]

    # Binary task: Computer Vision (cs.CV, +1) vs NLP (cs.CL, -1)
    mask = np.array([c in ["cs.CV", "cs.CL"] for c in categories])
    cv_cl_papers = [p for i, p in enumerate(papers) if mask[i]]
    cv_cl_texts = [p.abstract for p in cv_cl_papers]
    cv_cl_cats = [p.category for p in cv_cl_papers]
    y_binary = np.array([1 if c == "cs.CV" else -1 for c in cv_cl_cats])

    vec = TextVectorizer().fit(cv_cl_texts)
    _, _, X_all = vec.transform(cv_cl_texts)

    terms = ["image", "language", "convolutional"]
    idx_terms = [vec.vocabulary.get(t, 0) for t in terms]
    lda_res = LDASimulator.fit(X_all[:, idx_terms], cv_cl_cats)
    X_2d = lda_res.projected_2d

    st.markdown("### 🔬 Interactive Counterfactual Attribution Explorer")
    st.caption("Select a test paper to audit. Compute empirical influence scores to reveal which training papers most strongly biased the classifier.")

    col_pick_test, col_action = st.columns([2.0, 1.2])
    with col_pick_test:
        test_idx = st.selectbox(
            "Select Test Paper to Inspect with Model MRI:",
            options=list(range(len(cv_cl_papers))),
            format_func=lambda i: f"[{cv_cl_papers[i].category}] {cv_cl_papers[i].title[:45]}...",
        )
    test_paper = cv_cl_papers[test_idx]
    x_test = X_2d[test_idx]
    y_test = y_binary[test_idx]

    # Leave-one-out training set
    train_mask = np.ones(len(cv_cl_papers), dtype=bool)
    train_mask[test_idx] = False
    X_train = X_2d[train_mask]
    y_train = y_binary[train_mask]
    train_papers = [p for i, p in enumerate(cv_cl_papers) if train_mask[i]]

    # Compute Influence Functions on training samples
    influences = compute_influence_functions_2d(X_train, y_train, x_test, y_test)

    # Sort culprits by influence (positive influence = helps test loss; negative influence = hurts / biases test loss)
    culprit_indices = np.argsort(influences)[:3]  # most adversarial / counter-productive
    helper_indices = np.argsort(influences)[::-1][:3] # most supportive

    with col_action:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        retrain_without_culprit = st.checkbox("Counterfactual: Remove Top Culprit Paper", value=False)

    col_mri_plot, col_culprits = st.columns([1.8, 1.2])

    with col_mri_plot:
        # Fit baseline model
        if retrain_without_culprit:
            # Drop top culprit
            top_culprit_idx = culprit_indices[0]
            cf_mask = np.ones(len(X_train), dtype=bool)
            cf_mask[top_culprit_idx] = False
            active_X_train = X_train[cf_mask]
            active_y_train = y_train[cf_mask]
            st.info(f"Retrained after removing culprit paper: {train_papers[top_culprit_idx].title[:30]}...")
        else:
            active_X_train = X_train
            active_y_train = y_train

        svm_model = SVMSimulator.fit(active_X_train, active_y_train, C=1.0)

        fig_mri = go.Figure()
        # Decision boundary
        fig_mri.add_trace(go.Scatter(x=svm_model.line_x, y=svm_model.line_boundary, mode="lines", line=dict(color="#0f172a", width=3), name="Decision Boundary"))
        fig_mri.add_trace(go.Scatter(x=svm_model.line_x, y=svm_model.line_gutter_pos, mode="lines", line=dict(color="#10b981", width=1.5, dash="dash"), name="CV Gutter (+1)"))
        fig_mri.add_trace(go.Scatter(x=svm_model.line_x, y=svm_model.line_gutter_neg, mode="lines", line=dict(color="#6366f1", width=1.5, dash="dash"), name="CL Gutter (-1)"))

        # Training points
        for cat, label_val, color in [("cs.CV", 1, "#10b981"), ("cs.CL", -1, "#6366f1")]:
            c_m = (active_y_train == label_val)
            fig_mri.add_trace(go.Scatter(x=active_X_train[c_m, 0], y=active_X_train[c_m, 1], mode="markers", marker=dict(size=8, color=color, opacity=0.7), name=cat))

        # Highlight Top Culprit
        if not retrain_without_culprit:
            culprit_pts = X_train[culprit_indices]
            fig_mri.add_trace(go.Scatter(x=culprit_pts[:, 0], y=culprit_pts[:, 1], mode="markers", marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="#dc2626", width=3)), name="Top Culprits (Negative Torque)"))

        # Target Test Point
        fig_mri.add_trace(go.Scatter(x=[x_test[0]], y=[x_test[1]], mode="markers", marker=dict(size=16, color="#f59e0b", symbol="star", line=dict(color="#000", width=2)), name="Test Paper Audited"))

        y_span = np.max(X_train[:, 1]) - np.min(X_train[:, 1])
        fig_mri.update_layout(
            title=f"Model MRI: Boundary {'Snapped Back (Counterfactual)' if retrain_without_culprit else 'Standard State'}",
            xaxis_title="Discriminant Feature 1",
            yaxis_title="Discriminant Feature 2",
            yaxis=dict(range=[np.min(X_train[:, 1]) - 0.4*y_span, np.max(X_train[:, 1]) + 0.4*y_span]),
            height=440,
            margin=dict(l=20, r=20, b=20, t=35),
        )
        st.plotly_chart(fig_mri, use_container_width=True)

    with col_culprits:
        st.markdown("#### 🚨 Top Influential Training Papers")
        st.caption("Influence Score: I_up,loss(z_train, z_test). Negative influence pulls the boundary toward higher test error.")

        st.markdown("**Top Culprits (Adversarial Torque):**")
        for rank, c_idx in enumerate(culprit_indices, 1):
            p = train_papers[c_idx]
            inf = influences[c_idx]
            st.markdown(
                f"""
                <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:6px; padding:8px 10px; margin-bottom:6px; font-size:0.85rem;">
                  <b>#{rank} Culprit:</b> {p.title[:35]}...<br>
                  <span style="color:#991b1b;">Influence: <b>{inf:.4f}</b></span> &nbsp;|&nbsp; <code>{p.id}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("**Top Supportive Exemplars:**")
        for rank, h_idx in enumerate(helper_indices, 1):
            p = train_papers[h_idx]
            inf = influences[h_idx]
            st.markdown(
                f"""
                <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:6px; padding:8px 10px; margin-bottom:6px; font-size:0.85rem;">
                  <b>#{rank} Helper:</b> {p.title[:35]}...<br>
                  <span style="color:#166534;">Influence: <b>{inf:+.4f}</b></span> &nbsp;|&nbsp; <code>{p.id}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
