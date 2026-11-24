"""Four guided investigations. Each run records a prediction and its configuration."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from glassbox.core.experiments import (
    ranking_experiment, leakage_experiment, rotating_experiment, mri_dataset,
)
from glassbox.data.arxiv_corpus import get_arxiv_corpus
from glassbox.data.fixture import DEFAULT_TEACHING_CLAIM
from glassbox.utils.math_engine import compute_influence_functions_2d, compute_retrained_counterfactual
from glassbox.utils.design import (
    GREEN, RUST, MUTED, BLUE, chart, eyebrow, header, predict, live_experiment,
    verdict, receipt, source_note,
)


@st.cache_data(show_spinner=False)
def cached_leakage(seed, source):
    return leakage_experiment(seed, source)


def render_shift():
    header(1, "The paper that fell\nwithout changing.",
           "The query is identical. The paper is untouched. Three new documents arrive. Can the original paper still lose its place?",
           "RETRIEVAL / CORPUS SHIFT")
    left, right = st.columns([1, 1.55], gap="large")
    base_vec, baseline = ranking_experiment(False, False)
    with left:
        prediction = predict("shift", "What happens to DOC-03 after your intervention?",
                             ["Its rank gets worse", "Its rank stays the same", "Its rank improves"])
        eyebrow("INTERVENTION")
        refresh = st.checkbox("Refresh IDF using the incoming documents", value=True, key="shift_refresh")
        include = st.checkbox("Let incoming documents compete for a rank", value=False, key="shift_include")
        st.caption("Keep the second switch off to isolate a change in statistics. The vocabulary stays fixed in all four conditions.")
        active = live_experiment("shift", (refresh, include), prediction)
    vec, results = ranking_experiment(refresh, include) if active else (base_vec, baseline)
    target_before = next(r for r in baseline if r.doc_id == "DOC-03")
    target_after = next(r for r in results if r.doc_id == "DOC-03")
    with right:
        eyebrow("RANK MOVEMENT / LOWER IS BETTER")
        fig = go.Figure()
        previous = {r.doc_id: r for r in baseline}
        for r in results:
            b = previous.get(r.doc_id)
            color = RUST if r.doc_id == "DOC-03" else MUTED
            fig.add_trace(go.Scatter(
                x=[0, 1] if b else [1], y=[b.rank, r.rank] if b else [r.rank],
                mode="lines+markers+text", line=dict(color=color, width=3 if r.doc_id == "DOC-03" else 1),
                marker=dict(size=9 if r.doc_id == "DOC-03" else 5),
                text=[r.doc_id, r.doc_id] if b else [r.doc_id], textposition=["middle left", "middle right"] if b else "middle right",
                customdata=[r.title] * (2 if b else 1), hovertemplate="%{customdata}<br>Rank %{y}<extra></extra>",
                showlegend=False,
            ))
        fig.add_hrect(y0=.5, y1=5.5, fillcolor=GREEN, opacity=.055, line_width=0)
        fig.update_layout(xaxis=dict(tickvals=[0, 1], ticktext=["Original corpus", "After intervention" if active else "Awaiting intervention"], range=[-.3, 1.3], showgrid=False),
                          yaxis=dict(autorange="reversed", dtick=1, title="Rank"))
        chart(fig)
        st.caption("Shaded area: the top-five shortlist. Highlighted line: DOC-03.")
    with st.expander("Read the fixed query and the untouched document"):
        st.markdown(f"**Query:** {DEFAULT_TEACHING_CLAIM}")
        st.markdown(f"**DOC-03:** {target_before.title}")
        st.write(target_before.abstract)
        st.caption("Ten synthetic abstracts, deliberately constructed for this experiment. Lexical overlap is not scientific evidence.")
    if not active:
        return
    movement = "gets worse" if target_after.rank > target_before.rank else "improves" if target_after.rank < target_before.rank else "stays the same"
    verdict(prediction, f"DOC-03 {movement}: rank {target_before.rank} → {target_after.rank}. Its text and the query did not change.")
    idx = vec.vocabulary["consolidation"]
    a, b, c = st.columns(3)
    a.metric("Target rank", f"{target_before.rank} → {target_after.rank}")
    b.metric("IDF of ‘consolidation’", f"{base_vec.idf_weights[idx]:.3f} → {vec.idf_weights[idx]:.3f}")
    c.metric("Eligible documents", len(results))
    st.write("A search score belongs to a representation of a corpus, not just to a document. Refreshing IDF changes the query weights and document weights, including their normalization. Admitting candidates is a separate intervention.")
    with st.expander("Follow the numbers: IDF and the score decomposition"):
        st.latex(r"\mathrm{idf}(t)=\log\frac{1+N}{1+\mathrm{df}(t)}+1")
        st.code(f"Before: log((1 + {base_vec.n_documents}) / (1 + {base_vec.doc_frequencies[idx]})) + 1 = {base_vec.idf_weights[idx]:.6f}\nAfter:  log((1 + {vec.n_documents}) / (1 + {vec.doc_frequencies[idx]})) + 1 = {vec.idf_weights[idx]:.6f}")
        terms = sorted(set(t.term for t in target_before.matching_terms + target_after.matching_terms))
        before = {t.term: t.contribution for t in target_before.matching_terms}
        after = {t.term: t.contribution for t in target_after.matching_terms}
        contributions = go.Figure([
            go.Bar(name="Before", x=terms, y=[before.get(t, 0) for t in terms], marker_color=MUTED),
            go.Bar(name="After", x=terms, y=[after.get(t, 0) for t in terms], marker_color=GREEN),
        ])
        contributions.update_layout(barmode="group", yaxis_title="Contribution to cosine similarity")
        chart(contributions, 320)
        st.caption(f"Contributions sum to {target_after.sum_contributions:.8f}; dot-product score = {target_after.score:.8f}.")
    receipt("shift", {"refresh_idf": refresh, "include_incoming": include, "vocabulary": "frozen", "fixture": "synthetic-v1"},
            {"rank_before": target_before.rank, "rank_after": target_after.rank, "score_before": target_before.score, "score_after": target_after.score})


def render_leakage():
    header(2, "The suspiciously\ngood classifier.",
           "Two models. The same rows, the same split, the same tree. Only one has seen the answers while learning its representation.",
           "EVALUATION / DATA LEAKAGE")
    left, right = st.columns([1, 1.55], gap="large")
    with left:
        prediction = predict("leakage", "Which protocol will report higher validation accuracy?",
                             ["The honest protocol", "The leaked protocol", "They will tie"])
        eyebrow("EXPERIMENT SETTINGS")
        dataset = st.selectbox("Dataset", ["Random-label control", "Curated paper texts"], key="leakage_source")
        source = "noise" if dataset == "Random-label control" else "papers"
        seed = st.slider("Split / data seed", 0, 99, 42, key="leakage_seed")
        depth = st.slider("Tree depth", 1, 12, 4, key="leakage_depth")
        st.caption("The control has 120 rows, 40 noise features and four balanced random labels. It contains no population signal to learn.")
        active = live_experiment("leakage", (source, seed, depth), prediction, "Compare both protocols")
    with right:
        eyebrow("THE INFORMATION BOUNDARY")
        st.markdown("### Honest protocol")
        st.code("SPLIT → fit representation on TRAIN\n      → transform validation\n      → fit tree on TRAIN → evaluate", language=None)
        st.markdown("### Leaked protocol")
        st.code("fit representation on ALL LABELS\n      → SPLIT\n      → fit tree on TRAIN → evaluate", language=None)
        st.caption("For papers, the honest protocol fits vocabulary and IDF on training texts too. Both protocols use the same three term features. The noise control isolates supervised LDA leakage.")
    if not active:
        return
    result = cached_leakage(seed, source)
    honest = result["honest"]["test"][depth - 1]
    leaked = result["leaked"]["test"][depth - 1]
    fig = go.Figure()
    for name, color in [("honest", GREEN), ("leaked", RUST)]:
        fig.add_trace(go.Scatter(x=result["depths"], y=np.array(result[name]["test"]) * 100,
                                mode="lines+markers", name=f"{name.title()} validation", line=dict(color=color, width=3, dash="solid" if name == "honest" else "dash")))
    fig.add_vline(x=depth, line_dash="dot", line_color=MUTED)
    fig.add_hline(y=25, line_dash="dot", line_color=MUTED, annotation_text="25% balanced-class chance reference")
    fig.update_layout(xaxis=dict(title="Tree depth", dtick=1), yaxis=dict(title="Validation accuracy (%)", range=[0, 105]))
    chart(fig)
    a, b, c = st.columns(3)
    a.metric("Honest validation", f"{honest:.1%}")
    b.metric("Leaked validation", f"{leaked:.1%}")
    c.metric("Apparent difference", f"{(leaked-honest)*100:+.1f} pp")
    winner = "The leaked protocol scored higher" if leaked > honest else "The honest protocol scored higher" if honest > leaked else "Both protocols tied"
    verdict(prediction, f"{winner} at depth {depth}. Leakage invalidates the estimate even when it does not improve the score on this split.")
    st.write("LDA chooses directions using class labels. If validation labels help choose those directions, evaluation is no longer independent. A large score on the random-label control is a reason to investigate the protocol.")
    with st.expander("Inspect the protocol, split and limits"):
        st.write(f"Training rows: {len(result['train_indices'])}. Validation rows: {len(result['test_indices'])}. Seed: {seed}.")
        st.code(f"Training row IDs: {result['train_indices'].tolist()}\nValidation row IDs: {result['test_indices'].tolist()}")
        st.latex(r"S_B=\sum_c n_c(\mu_c-\mu)(\mu_c-\mu)^T")
        st.write("In the honest protocol every class mean in this equation comes from training rows. In the leaked protocol validation labels contribute to those means.")
        st.caption("These are exploratory validation curves, not a final test benchmark. Selecting a depth or seed after looking at this screen also uses validation information. Keep a separate final test set for a performance claim.")
        st.markdown("[Reference: scikit-learn on data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)")
    receipt("leakage", {"source": source, "seed": seed, "depth": depth},
            {"honest_validation_accuracy": honest, "leaked_validation_accuracy": leaked,
             "train_indices": result["train_indices"].tolist(), "validation_indices": result["test_indices"].tolist()})
    source_note("LDA scatter arithmetic: project implementation. Eigenproblem: NumPy. Split and decision tree: scikit-learn references.")


def boundary_trace(theta, X, name, color, dash="solid", logit=0.):
    """Draw a level set without replacing vertical lines with y=0."""
    low = np.minimum(X.min(axis=0) - .2, -.2)
    high = np.maximum(X.max(axis=0) + .2, .2)
    if np.linalg.norm(theta[:2]) < 1e-12:
        return None
    if abs(theta[1]) > abs(theta[0]):
        x = np.linspace(low[0], high[0], 60)
        y = (logit - theta[2] - theta[0] * x) / theta[1]
    else:
        y = np.linspace(low[1], high[1], 60)
        x = (logit - theta[2] - theta[1] * y) / theta[0]
    return go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, width=2, dash=dash))


def render_influence():
    header(3, "The explanation\non trial.",
           "An influence score makes a local prediction. Remove a training paper, retrain the model, and ask whether the evidence agrees.",
           "ATTRIBUTION / COUNTERFACTUALS")
    papers = [p for p in get_arxiv_corpus() if p.category in ("cs.CV", "cs.CL")]
    left, right = st.columns([1, 1.55], gap="large")
    with left:
        test_idx = st.selectbox("Paper held out before fitting", range(len(papers)),
                               format_func=lambda i: papers[i].title, key="influence_paper")
        reg = st.select_slider("L2 regularization", options=[.01, .03, .1, .3, 1.], value=.1, key="influence_reg")
        data = mri_dataset(test_idx)
        base = compute_influence_functions_2d(data["X"], data["y"], data["x_test"], data["y_test"], reg)
        choice = st.radio("Training paper to remove", ["Largest positive influence", "Most negative influence"], key="influence_remove")
        drop = int(np.argmax(base["influences"]) if choice.startswith("Largest") else np.argmin(base["influences"]))
        inf = float(base["influences"][drop])
        st.caption(f"Selected: {data['papers'][drop].title}. Upweighting influence: {inf:+.5f}.")
        prediction = predict("influence", "After removal, the held-out paper’s loss will…",
                             ["Decrease", "Increase", "Stay approximately unchanged"])
        eyebrow("COUNTERFACTUAL")
        active = live_experiment("influence", (test_idx, reg, choice), prediction, "Remove and retrain")
    cf = compute_retrained_counterfactual(data["X"], data["y"], data["x_test"], data["y_test"], drop, reg) if active else None
    with right:
        eyebrow("ONE MODEL / BEFORE AND AFTER")
        X, y = data["X"], data["y"]
        plot_X = np.vstack([X, data["x_test"]])
        fig = go.Figure()
        for label, name, color, symbol in [(1, "Vision", GREEN, "circle"), (-1, "Language", BLUE, "diamond")]:
            mask = y == label
            fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode="markers", name=name,
                                    text=[p.title for p, m in zip(data["papers"], mask) if m],
                                    hovertemplate="%{text}<extra></extra>", marker=dict(size=8, color=color, symbol=symbol, opacity=.6)))
        fig.add_trace(go.Scatter(x=[data["x_test"][0]], y=[data["x_test"][1]], mode="markers", name="Held-out paper", marker=dict(size=15, symbol="star", color="#9B7D3E")))
        fig.add_trace(go.Scatter(x=[X[drop, 0]], y=[X[drop, 1]], mode="markers", name="Removed paper" if active else "Proposed removal", marker=dict(size=16, symbol="x" if active else "circle-open", color=RUST, line_width=2)))
        for theta, name, color, dash in [(base["theta"], "Original P=0.5", MUTED, "dot" if active else "solid")] + ([(cf["theta_cf"], "Retrained P=0.5", RUST, "solid")] if active else []):
            trace = boundary_trace(theta, plot_X, name, color, dash)
            if trace is not None:
                fig.add_trace(trace)
        fig.update_layout(xaxis=dict(title="‘image’ TF-IDF × 10", range=[-.2, max(.4, plot_X[:, 0].max()+.2)]),
                          yaxis=dict(title="‘language’ TF-IDF × 10", range=[-.2, max(.4, plot_X[:, 1].max()+.2)]))
        chart(fig)
        st.caption("Two fixed term axes. Vocabulary and IDF are fitted without the held-out paper, then frozen during removal. Overlapping points share these two coordinates.")
    if not active:
        return
    actual, predicted = cf["actual_delta_loss"], cf["pred_delta_loss"]
    a, b, c = st.columns(3)
    a.metric("Classical prediction −I/n", f"{predicted:+.6f}")
    b.metric("Measured loss change", f"{actual:+.6f}")
    c.metric("Absolute approximation error", f"{abs(actual-predicted):.6f}")
    observation = "decreased" if actual < -1e-6 else "increased" if actual > 1e-6 else "was approximately unchanged"
    verdict(prediction, f"The held-out loss {observation}: {cf['test_loss_orig']:.6f} → {cf['test_loss_cf']:.6f}. The sign is {'consistent' if np.sign(actual) == np.sign(predicted) else 'not consistent'} with −I/n.")
    st.write("Positive influence means a small upweighting increases this paper’s loss. Negative influence means it decreases that loss. Neither sign guarantees the result of deleting a whole observation.")
    with st.expander("Open the derivation and the normalization check"):
        st.latex(r"R(\theta)=\frac1n\sum_i\ell_i(\theta)+\frac\lambda2\|\theta\|^2,\quad I_i=-g_{test}^{T}H^{-1}g_i")
        st.code(f"n = {len(X)}; lambda = {reg}\nI = {inf:+.8f}\n-I/n = {predicted:+.8f}\nActual leave-one-out delta = {actual:+.8f}")
        st.write("−I/n describes deleting a loss term while retaining the original 1/n scale. Actual leave-one-out retraining averages over n−1 points with the same regularization. That changes the relative weighting of data and penalty as well as taking a finite step.")
        st.latex(r"\Delta\ell_{LOO}\approx\frac{g_{test}^{T}H^{-1}(g_i+\lambda\theta)}{n-1}")
        st.metric("Prediction with leave-one-out normalization", f"{cf['pred_delta_loss_normalized']:+.6f}")
        st.caption("The intercept is regularized too. Logistic loss with positive L2 regularization is convex; disagreement here is not evidence of non-convexity. The experiment holds the representation fixed and explains only this two-feature classifier.")
        st.markdown("[Source: Koh & Liang, ICML 2017](https://proceedings.mlr.press/v70/koh17a.html)")
    receipt("influence", {"held_out_paper": data["test_paper"].id, "removed_paper": data["papers"][drop].id, "l2_reg": reg,
                          "features": ["image", "language"], "feature_scale": 10, "representation": "train-only, frozen for removal"},
            {"influence": inf, "predicted_delta": predicted, "normalized_predicted_delta": cf["pred_delta_loss_normalized"], "actual_delta": actual})
    source_note("Newton updates, logistic gradients, Hessian and influence arithmetic: project implementation using NumPy linear solves.")


def render_rotation():
    header(4, "The rotating answer.",
           "The data barely moves. The principal direction swings. Is the model unstable, or are we asking the wrong object to be stable?",
           "GEOMETRY / EIGENVALUE GAPS")
    left, right = st.columns([1, 1.55], gap="large")
    with left:
        prediction = predict("rotation", "Which object will move more under a small perturbation?",
                             ["The first principal axis", "The retained two-dimensional plane", "They will be equally stable"])
        eyebrow("PERTURBATION")
        gap_label = st.select_slider("Gap between the first two eigenvalues", options=["0.002 / nearly tied", "0.02 / close", "0.2 / separated", "1.0 / well separated"], key="rotation_gap")
        gap = float(gap_label.split(" /")[0])
        perturbation = st.slider("Off-diagonal perturbation ε", .0, .1, .01, .001, format="%.3f", key="rotation_epsilon")
        st.caption("The third eigenvalue starts at 0.15. Separating the leading pair from the third keeps the plane identifiable even when its internal axes are ambiguous.")
        active = live_experiment("rotation", (gap, perturbation), prediction, "Perturb and compare")
    result = rotating_experiment(gap, perturbation if active else 0.)
    with right:
        eyebrow("TOP VIEW / PRINCIPAL DIRECTIONS")
        fig = go.Figure()
        angles = np.linspace(0, 2*np.pi, 160)
        circle = np.vstack([np.cos(angles), np.sin(angles)])
        for values, axes, name, color, dash in [
            (result["values_before"], result["axes_before"], "Before", MUTED, "dot"),
            (result["values_after"], result["axes_after"], "After" if active else "Awaiting perturbation", GREEN, "solid"),
        ]:
            ellipse = axes[:, :2] @ (np.sqrt(values[:2])[:, None] * circle)
            fig.add_trace(go.Scatter(x=ellipse[0], y=ellipse[1], mode="lines", name=f"{name} variance ellipse", line=dict(color=color, width=1, dash=dash)))
            for i in range(2):
                v = axes[:, i] * np.sqrt(values[i])
                fig.add_trace(go.Scatter(x=[-v[0], v[0]], y=[-v[1], v[1]], mode="lines", name=f"{name} PC{i+1}", line=dict(color=color, width=3 if i == 0 else 1, dash=dash), showlegend=i == 0))
        fig.update_layout(xaxis=dict(title="x", range=[-1.9, 1.9], constrain="domain"),
                          yaxis=dict(title="y", range=[-1.9, 1.9], scaleanchor="x", scaleratio=1))
        chart(fig)
        st.caption("An almost circular variance ellipse has no strongly preferred direction. The thinner line is PC2; signs of eigenvectors are ignored.")
    if not active:
        return
    plane_angle = float(np.degrees(np.arcsin(np.clip(result["projector_distance"], 0, 1))))
    a, b, c = st.columns(3)
    a.metric("First-axis rotation", f"{result['axis_angle']:.2f}°")
    b.metric("Largest plane angle", f"{plane_angle:.3f}°")
    c.metric("Relative covariance change", f"{result['relative_perturbation']:.2%}")
    verdict(prediction, f"The axis rotated {result['axis_angle']:.2f}° while the plane moved {plane_angle:.3f}°. Compare angles, not the arbitrary sign of an eigenvector.")
    st.write("There are two different gaps. The gap between λ₁ and λ₂ controls how clearly the first axis is identified. The gap between λ₂ and λ₃ protects the leading plane. ‘Stable PCA’ needs to specify which of these objects it means.")
    with st.expander("Inspect the matrices and the subspace check"):
        st.latex(r"\Sigma=\mathrm{diag}(2+\delta,2,0.15),\quad E=\varepsilon\begin{bmatrix}0&1&0.2\\1&0&0\\0.2&0&0\end{bmatrix}")
        st.code(f"Before:\n{np.array2string(result['baseline'], precision=5)}\nAfter:\n{np.array2string(result['changed'], precision=5)}")
        st.latex(r"P=V_{1:2}V_{1:2}^{T},\quad\|P_{after}-P_{before}\|_2=\sin(\theta_{max})")
        st.write(f"Projector distance: {result['projector_distance']:.8f}. Remaining outer eigenvalue gap: {result['outer_gap']:.5f}.")
        st.caption("This is a constructed covariance experiment, not an empirical result on the paper corpus. Six symmetric observations realize each covariance exactly. Both eigensystems use the scratch Jacobi solver; the spectral norm uses NumPy. The small x-z perturbation lets the plane move too.")
    receipt("rotation", {"gap": gap, "perturbation": perturbation, "fixture": "exact-covariance-v1"},
            {"axis_angle_degrees": result["axis_angle"], "plane_angle_degrees": plane_angle,
             "projector_distance": result["projector_distance"], "relative_perturbation": result["relative_perturbation"]})
