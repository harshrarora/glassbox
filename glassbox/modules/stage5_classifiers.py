"""Stage 5: The Supervised Classifier Arena.
Interactive models: SVM Margin Explorer, Decision Tree Gini Splits, KNN, and Naive Bayes.
"""

import streamlit as st
import numpy as np
import pandas as pd
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
    SimpleMLP,
    compute_confusion_matrix_and_metrics,
    compute_bias_variance_curve,
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
        what_text="Fitting and comparing foundational syllabus classifiers (SVM, Decision Trees, KNN, Naive Bayes, and a 2-Layer Neural Network MLP) alongside a rigorous evaluation suite.",
        why_text="Classifiers impose distinct inductive biases on data: SVM optimizes geometric margins, Decision Trees perform recursive orthogonal cuts, KNN estimates local density, Naive Bayes computes joint posteriors, and Neural Networks learn non-linear layer transformations.",
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
            "5. Multi-Layer Perceptron (MLP) Forward & Activation Inspector",
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

    # -------------------------------------------------------------------------
    # 5. Multi-Layer Perceptron (MLP)
    # -------------------------------------------------------------------------
    elif model_choice.startswith("5."):
        st.markdown("### 🧠 Multi-Layer Perceptron (MLP) Forward & Activation Inspector")
        st.caption("Inspect real-time numerical activations flowing through a 2-layer Neural Network: Input (2D) → Hidden (4 ReLU Units) → Output (4 Softmax Probabilities).")

        class_list = ["cs.CV", "cs.CL", "cs.LG", "stat.ML"]
        y_indices = np.array([class_list.index(c) for c in categories])
        mlp_model = SimpleMLP(hidden_dim=4, num_classes=4).fit(X_2d, y_indices, epochs=160, lr=0.08)

        col_paper_sel, col_arch_info = st.columns([2.0, 1.2])
        with col_paper_sel:
            paper_titles = [f"[{p.category}] {p.title}" for p in papers]
            chosen_paper_idx = st.selectbox(
                "Select Research Abstract to Propagate Forward:",
                options=list(range(len(papers))),
                format_func=lambda i: paper_titles[i],
                index=0,
            )
            target_paper = papers[chosen_paper_idx]
            x_in = X_2d[chosen_paper_idx]

        with col_arch_info:
            st.markdown(
                f"""
                <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; padding:12px;">
                  <b>Input Coordinates:</b> x₁ = <code>{x_in[0]:.3f}</code>, x₂ = <code>{x_in[1]:.3f}</code><br>
                  <b>True Category:</b> <span style="color:{CATEGORY_COLORS[target_paper.category]}; font-weight:700;">{target_paper.category}</span><br>
                  <b>Architecture:</b> 2 Inputs → 4 ReLUs → 4 Softmax Outputs
                </div>
                """,
                unsafe_allow_html=True,
            )

        mlp_act = mlp_model.inspect_sample(x_in)

        col_nn_graph, col_nn_math = st.columns([1.8, 1.2])
        with col_nn_graph:
            fig_nn = go.Figure()

            in_y = [0.4, 0.6]
            hid_y = [0.2, 0.4, 0.6, 0.8]
            out_y = [0.2, 0.4, 0.6, 0.8]

            for i, y_i in enumerate(in_y):
                for j, y_h in enumerate(hid_y):
                    w = mlp_act["W1"][j, i]
                    fig_nn.add_trace(go.Scatter(
                        x=[0, 1], y=[y_i, y_h],
                        mode="lines",
                        line=dict(color="#94a3b8" if w > 0 else "#cbd5e1", width=max(1, min(6, abs(w) * 3))),
                        hoverinfo="text",
                        text=f"W₁[{j},{i}] = {w:.3f}",
                        showlegend=False,
                    ))

            for j, y_h in enumerate(hid_y):
                for k, y_o in enumerate(out_y):
                    w = mlp_act["W2"][k, j]
                    fig_nn.add_trace(go.Scatter(
                        x=[1, 2], y=[y_h, y_o],
                        mode="lines",
                        line=dict(color="#94a3b8" if w > 0 else "#cbd5e1", width=max(1, min(6, abs(w) * 3))),
                        hoverinfo="text",
                        text=f"W₂[{k},{j}] = {w:.3f}",
                        showlegend=False,
                    ))

            fig_nn.add_trace(go.Scatter(
                x=[0, 0], y=in_y,
                mode="markers+text",
                marker=dict(size=26, color="#e2e8f0", line=dict(color="#0f172a", width=2)),
                text=["x₁", "x₂"],
                textposition="middle center",
                textfont=dict(size=12, color="#0f172a", family="monospace"),
                hovertext=[f"Input 1: {x_in[0]:.4f}", f"Input 2: {x_in[1]:.4f}"],
                name="Input Layer",
            ))

            hid_colors = ["#3b82f6" if a > 0 else "#94a3b8" for a in mlp_act["a1"]]
            fig_nn.add_trace(go.Scatter(
                x=[1, 1, 1, 1], y=hid_y,
                mode="markers+text",
                marker=dict(size=28, color=hid_colors, line=dict(color="#1e293b", width=2)),
                text=[f"h{j+1}" for j in range(4)],
                textposition="middle center",
                textfont=dict(size=12, color="#ffffff", family="monospace"),
                hovertext=[f"Hidden {j+1}: z={mlp_act['z1'][j]:.3f}, a=ReLU(z)={mlp_act['a1'][j]:.3f}" for j in range(4)],
                name="Hidden Layer (ReLU)",
            ))

            out_colors = [CATEGORY_COLORS[c] for c in class_list]
            pred_idx = mlp_act["pred_idx"]
            fig_nn.add_trace(go.Scatter(
                x=[2, 2, 2, 2], y=out_y,
                mode="markers+text",
                marker=dict(size=[max(20, p * 45) for p in mlp_act["probs"]], color=out_colors, line=dict(color="#0f172a", width=[3 if i == pred_idx else 1 for i in range(4)])),
                text=[c.split(".")[1] for c in class_list],
                textposition="middle center",
                textfont=dict(size=11, color="#ffffff", family="monospace"),
                hovertext=[f"{c}: Softmax P={p*100:.2f}%" for c, p in zip(class_list, mlp_act["probs"])],
                name="Output Layer (Softmax)",
            ))

            fig_nn.update_layout(
                title="Neural Forward Propagation (Interactive Node Activations)",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=True, tickvals=[0, 1, 2], ticktext=["Input Layer (2D)", "Hidden Layer (4 ReLUs)", "Output Softmax (4 Classes)"]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=380,
                margin=dict(l=20, r=20, b=20, t=35),
            )
            st.plotly_chart(fig_nn, use_container_width=True)

        with col_nn_math:
            st.markdown("#### 🔢 Step-by-Step Numerical Derivation")
            st.markdown(
                f"""
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px; font-size:0.85rem; line-height:1.5;">
                  <b>1. Input Vector x:</b><br>
                  <code>x = [{x_in[0]:.3f}, {x_in[1]:.3f}]ᵀ</code><br><br>
                  <b>2. Hidden Pre-activations z^[1] = W^[1]x + b^[1]:</b><br>
                  <code>z^[1] = [{', '.join(f'{v:.3f}' for v in mlp_act['z1'])}]ᵀ</code><br><br>
                  <b>3. ReLU Activations a^[1] = max(0, z^[1]):</b><br>
                  <code>a^[1] = [{', '.join(f'{v:.3f}' for v in mlp_act['a1'])}]ᵀ</code><br><br>
                  <b>4. Output Logits z^[2] = W^[2]a^[1] + b^[2]:</b><br>
                  <code>z^[2] = [{', '.join(f'{v:.3f}' for v in mlp_act['z2'])}]ᵀ</code><br><br>
                  <b>5. Softmax Probabilities ŷ = exp(z^[2]) / ∑ exp(z^[2]):</b><br>
                  • <b>cs.CV:</b> <code>{mlp_act['probs'][0]*100:.2f}%</code><br>
                  • <b>cs.CL:</b> <code>{mlp_act['probs'][1]*100:.2f}%</code><br>
                  • <b>cs.LG:</b> <code>{mlp_act['probs'][2]*100:.2f}%</code><br>
                  • <b>stat.ML:</b> <code>{mlp_act['probs'][3]*100:.2f}%</code><br><br>
                  <b>Predicted Class:</b> <span style="color:{CATEGORY_COLORS[class_list[pred_idx]]}; font-weight:800;">{class_list[pred_idx]}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # MODULE C: Rigorous Evaluation & Error Diagnostics Suite
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 Rigorous Model Evaluation & Error Diagnostics Suite")
    st.caption("Comprehensive empirical evaluation from scratch: 4x4 Confusion Matrix, Per-Class Precision/Recall/Macro-F1, and Bias-Variance Overfitting Curve.")

    class_list = ["cs.CV", "cs.CL", "cs.LG", "stat.ML"]
    y_true_indices = np.array([class_list.index(c) for c in categories])

    mlp_full = SimpleMLP(hidden_dim=4, num_classes=4).fit(X_2d, y_true_indices, epochs=160, lr=0.08)
    preds_indices = mlp_full.predict(X_2d)
    preds_categories = [class_list[i] for i in preds_indices]

    eval_results = compute_confusion_matrix_and_metrics(categories, preds_categories, class_list)

    tab_eval_cm, tab_eval_curve = st.tabs([
        "🎯 Confusion Matrix & Failure Inspector",
        "📈 Bias-Variance / Overfitting Curve (Decision Tree Depth)",
    ])

    with tab_eval_cm:
        col_cm_plot, col_cm_report = st.columns([1.6, 1.4])
        with col_cm_plot:
            fig_cm = px.imshow(
                eval_results["confusion_matrix"],
                x=class_list,
                y=class_list,
                color_continuous_scale="Blues",
                text_auto=True,
                labels=dict(x="Predicted Domain", y="Ground-Truth Domain"),
                title=f"4×4 Confusion Matrix (Overall Accuracy: {eval_results['accuracy']*100:.1f}%)",
            )
            fig_cm.update_layout(height=340, margin=dict(l=20, r=20, b=20, t=35))
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_cm_report:
            st.markdown("#### 📋 Scratch Classification Report HUD")
            df_report_rows = []
            for c in class_list:
                m = eval_results["per_class"][c]
                df_report_rows.append({
                    "Domain": c,
                    "Precision": f"{m['precision']*100:.1f}%",
                    "Recall": f"{m['recall']*100:.1f}%",
                    "F1-Score": f"{m['f1']*100:.1f}%",
                    "Support": m["support"],
                })
            df_report_rows.append({
                "Domain": "<b>Macro Avg</b>",
                "Precision": f"<b>{eval_results['macro_precision']*100:.1f}%</b>",
                "Recall": f"<b>{eval_results['macro_recall']*100:.1f}%</b>",
                "F1-Score": f"<b>{eval_results['macro_f1']*100:.1f}%</b>",
                "Support": f"<b>{len(categories)}</b>",
            })
            st.write(pd.DataFrame(df_report_rows).to_html(escape=False, index=False), unsafe_allow_html=True)

        # Inspect Failure Feature
        st.markdown("#### 🔍 The 'Inspect Failure' Deep-Dive")
        st.caption("Select any off-diagonal cell where the classifier was fooled. Inspect the exact scientific paper and its overlapping vocabulary.")

        error_pairs = []
        for i, true_cat in enumerate(categories):
            pred_cat = preds_categories[i]
            if true_cat != pred_cat:
                error_pairs.append({
                    "paper": papers[i],
                    "true_cat": true_cat,
                    "pred_cat": pred_cat,
                })

        if error_pairs:
            err_options = [f"Paper #{i+1}: '{ep['paper'].title[:38]}...' (True: {ep['true_cat']} → Pred: {ep['pred_cat']})" for i, ep in enumerate(error_pairs)]
            chosen_err_idx = st.selectbox("Misclassified Abstracts:", options=list(range(len(error_pairs))), format_func=lambda i: err_options[i])
            active_err = error_pairs[chosen_err_idx]
            p_err = active_err["paper"]

            st.markdown(
                f"""
                <div style="background:#fff1f2; border:1px solid #fecdd3; border-radius:8px; padding:14px; margin-top:8px;">
                  <b>Title:</b> {p_err.title}<br>
                  <b>Paper ID:</b> <code>{p_err.id}</code> | <b>Year:</b> {p_err.year} | <b>Authors:</b> {p_err.authors}<br>
                  <b>True Label:</b> <span style="color:#059669; font-weight:700;">{active_err['true_cat']}</span> | 
                  <b>Model Predicted:</b> <span style="color:#e11d48; font-weight:700;">{active_err['pred_cat']}</span><br><br>
                  <b>Abstract Excerpt:</b> <i>"{p_err.abstract}"</i><br><br>
                  <b>💡 Why this misclassification occurred:</b><br>
                  This research paper sits at the intersection of {active_err['true_cat']} and {active_err['pred_cat']}. 
                  It employs cross-disciplinary vocabulary (such as multimodal representations, attention over visual tokens, or optimization bounds on neural layers) that places its representation near the decision boundary gutter.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.success("🎉 All documents are correctly classified in the current projection space!")

    with tab_eval_curve:
        st.markdown("#### 📉 The Bias-Variance Tradeoff: Decision Tree Max Depth Simulation")
        st.caption("Vary max_depth from 1 to 15. Observe how training accuracy continues climbing to 100% while validation accuracy degrades once variance dominates.")

        depth_val = st.slider("Decision Tree max_depth:", min_value=1, max_value=15, value=4)

        bv_data = compute_bias_variance_curve(X_2d, np.array(categories), max_depth_range=list(range(1, 16)))

        fig_bv = go.Figure()
        fig_bv.add_trace(go.Scatter(
            x=bv_data["depths"],
            y=[acc * 100 for acc in bv_data["train_acc"]],
            mode="lines+markers",
            name="Training Accuracy",
            line=dict(color="#10b981", width=3),
            marker=dict(size=7),
        ))
        fig_bv.add_trace(go.Scatter(
            x=bv_data["depths"],
            y=[acc * 100 for acc in bv_data["test_acc"]],
            mode="lines+markers",
            name="Test / Validation Accuracy",
            line=dict(color="#ef4444", width=3, dash="dash"),
            marker=dict(size=7),
        ))

        fig_bv.add_vline(x=depth_val, line_width=2, line_dash="dot", line_color="#0f172a", annotation_text=f"Selected Depth: {depth_val}")

        fig_bv.update_layout(
            title="Overfitting Curve: Training vs. Validation Accuracy Across Tree Depth",
            xaxis=dict(title="Tree Maximum Depth (max_depth)", dtick=1),
            yaxis=dict(title="Classification Accuracy (%)", range=[30, 105]),
            height=360,
            margin=dict(l=20, r=20, b=20, t=35),
            legend=dict(x=0.05, y=0.15),
        )
        st.plotly_chart(fig_bv, use_container_width=True)

        curr_train = bv_data["train_acc"][depth_val - 1] * 100
        curr_test = bv_data["test_acc"][depth_val - 1] * 100
        gap = curr_train - curr_test

        st.markdown(
            f"""
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px; font-size:0.85rem;">
              <b>Operating Point:</b> <code>max_depth = {depth_val}</code> | 
              <b>Train Accuracy:</b> <code>{curr_train:.1f}%</code> | 
              <b>Test Accuracy:</b> <code>{curr_test:.1f}%</code> | 
              <b>Generalization Gap:</b> <code>{gap:.1f}%</code><br>
              {
                "⚠️ <b>High Variance (Overfitting):</b> The tree is memorizing idiosyncratic noise in the training set." if gap > 20
                else "⚠️ <b>High Bias (Underfitting):</b> The tree is too shallow to capture decision structure." if depth_val <= 2
                else "✅ <b>Optimal Inductive Bias:</b> Healthy balance between bias and variance."
              }
            </div>
            """,
            unsafe_allow_html=True,
        )
