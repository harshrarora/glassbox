"""Live reference comparisons with aligned inputs and explicit implementation boundaries."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from glassbox.data.arxiv_corpus import get_arxiv_corpus
from glassbox.core.geometry import fit_pca_3d
from glassbox.core.experiments import text_projection_split
from glassbox.utils.pipeline import vectorizer, settings, coordinates, resolve_terms
from glassbox.utils.design import chart, GREEN, RUST, BLUE
from glassbox.utils.math_engine import compute_knn_neighbors, compute_naive_bayes_posteriors


def compare_tfidf(texts, config):
    """Both paths receive exactly the same analyzer, vocabulary and TF convention."""
    vec = vectorizer(config).fit(texts)
    counts, _, scratch = vec.transform(texts)
    reference = TfidfVectorizer(analyzer=vec.tokenizer, vocabulary=vec.vocabulary,
                               norm="l2", smooth_idf=True, dtype=np.float64).fit_transform(texts).toarray()
    return vec, counts, scratch, reference


def render_comparison(stage):
    texts = [p.abstract for p in get_arxiv_corpus()]
    labels = np.array([p.category for p in get_arxiv_corpus()])
    st.subheader("Scratch ↔ reference")
    st.caption("Shared input. Matched conventions. The difference is measured, not assumed.")
    if stage == 3:
        vec, counts, scratch, reference = compare_tfidf(texts, settings())
        papers = get_arxiv_corpus()
        selected = np.argsort(counts.sum(axis=0))[-18:]
        diff = scratch - reference
        a, b, c = st.columns(3)
        a.metric("Maximum absolute error", f"{np.max(np.abs(diff)):.2e}")
        b.metric("Compared coordinates", f"{scratch.size:,}")
        c.metric("Reference", "scikit-learn")

        # Interactive visual mode selector
        view_mode = st.radio("Visualization Perspective:",
                             ["Parity Correlation (y = x)", "Residual Error Heatmap", "Side-by-Side Matrices"],
                             horizontal=True, key="compare_view_stage3")

        rep_docs = [0, 1, 2, 16, 17, 18, 32, 33, 34, 48, 49, 50]  # 3 papers per domain
        row_labels = [f"[{papers[i].category}] {papers[i].title[:20]}..." for i in rep_docs]
        col_labels = [vec.feature_names[i] for i in selected]

        if view_mode == "Parity Correlation (y = x)":
            nz_mask = (scratch > 0) | (reference > 0)
            nz_scratch = scratch[nz_mask]
            nz_ref = reference[nz_mask]
            if len(nz_scratch) > 1500:
                rng = np.random.default_rng(42)
                idx_sample = rng.choice(len(nz_scratch), size=1500, replace=False)
                nz_scratch = nz_scratch[idx_sample]
                nz_ref = nz_ref[idx_sample]

            max_val = float(max(np.max(scratch), np.max(reference), 0.1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", name="Ideal Identity (y = x)",
                line=dict(color=GREEN, width=2, dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=nz_ref, y=nz_scratch,
                mode="markers", name="Matrix Coordinates",
                marker=dict(color=BLUE, size=5, opacity=0.7, line=dict(width=0)),
                hovertemplate="scikit-learn: %{x:.6f}<br>Scratch NumPy: %{y:.6f}<br>Delta: %{customdata:.2e}<extra></extra>",
                customdata=np.abs(nz_scratch - nz_ref)
            ))
            fig.update_layout(
                title="First-Principles Parity Plot: Scratch NumPy vs. scikit-learn TfidfVectorizer",
                xaxis_title="scikit-learn TF-IDF Coordinate Value",
                yaxis_title="Scratch NumPy TF-IDF Coordinate Value",
                height=380
            )
            chart(fig, 380)
            st.caption("Every non-zero TF-IDF cell plotted against reference. Perfect alignment along the y = x line verifies identical numerical behavior across all documents and terms.")

        elif view_mode == "Residual Error Heatmap":
            sub_diff = np.abs(diff[np.ix_(rep_docs, selected)])
            fig_diff = go.Figure(go.Heatmap(
                z=sub_diff,
                x=col_labels,
                y=row_labels,
                colorscale="Reds",
                zmin=0.0,
                zmax=max(float(np.max(sub_diff)), 1e-15),
                colorbar=dict(title="|Delta|"),
                hovertemplate="Paper: %{y}<br>Term: %{x}<br>Error: %{z:.2e}<extra></extra>"
            ))
            fig_diff.update_layout(
                title="Coordinate-Wise Absolute Residual |Scratch - scikit-learn|",
                xaxis_title="Vocabulary Term",
                yaxis_title="Curated Paper",
                height=380
            )
            chart(fig_diff, 380)
            st.caption("Visual verification that residual error across landmark papers is zero within IEEE 754 float64 machine precision (<= 2.22e-16).")

        else:
            left, right = st.columns(2)
            zmax = float(max(scratch.max(), reference.max(), 0.01))
            for target, matrix, label in [(left, scratch, "Scratch NumPy"), (right, reference, "TfidfVectorizer")]:
                with target:
                    st.markdown(f"**{label}**")
                    fig = go.Figure(go.Heatmap(
                        z=matrix[np.ix_(rep_docs, selected)],
                        x=col_labels,
                        y=row_labels,
                        colorscale="Blues",
                        zmin=0,
                        zmax=zmax,
                        colorbar=dict(title="Weight"),
                        hovertemplate="Paper: %{y}<br>Term: %{x}<br>Weight: %{z:.4f}<extra></extra>"
                    ))
                    fig.update_layout(xaxis_title="Term", yaxis_title="Curated Paper", height=380)
                    chart(fig, 380)

        with st.expander("Magnify the residual and the shared contract"):
            st.code("same analyzer + vocabulary\nraw count * [log((1 + N) / (1 + df)) + 1]\nL2-normalize each document")
            st.write(f"All-coordinate Frobenius residual: {np.linalg.norm(diff):.8e}.")
            st.caption("A small residual checks this implementation on the active corpus. It is not a universal guarantee. The current preprocessing and n-gram controls apply to both paths.")
    elif stage == 4:
        vec = vectorizer().fit(texts)
        X = coordinates(vec, vec.transform(texts)[2], settings()["terms"])
        active_terms = resolve_terms(vec, settings()["terms"])
        result = fit_pca_3d(X, active_terms)
        reference = PCA(n_components=3, svd_solver="full").fit(X)
        P = result.eigenvectors[:, :2] @ result.eigenvectors[:, :2].T
        P_ref = reference.components_[:2].T @ reference.components_[:2]
        a,b,c = st.columns(3)
        a.metric("Eigenvalue residual", f"{np.max(np.abs(result.eigenvalues-reference.explained_variance_)):.2e}")
        b.metric("Plane projector residual", f"{np.linalg.norm(P-P_ref, ord=2):.2e}")
        c.metric("Jacobi rotations", result.jacobi_iterations)
        fig = go.Figure([go.Bar(x=["PC1","PC2","PC3"], y=result.eigenvalues, name="Scratch Jacobi", marker_color=BLUE),
                         go.Bar(x=["PC1","PC2","PC3"], y=reference.explained_variance_, name="scikit-learn PCA", marker_color=GREEN)])
        fig.update_layout(barmode="group", yaxis_title="Explained variance")
        chart(fig)
        st.caption("Active coordinates: " + ", ".join(active_terms) + ". Change the triplet in Playground. Both paths center the same three columns and use sample covariance (n-1).")
        st.write("Compare subspaces rather than raw eigenvector signs. At a tied truncation boundary, different valid planes can exist; a larger projector residual then needs interpretation.")
        with st.expander("Inspect the Jacobi eigenpair residual"):
            residual = result.covariance_matrix @ result.eigenvectors - result.eigenvectors * result.eigenvalues
            st.code(np.array2string(residual, precision=8))
            st.latex(r"\Sigma V-V\Lambda=0,\quad P=V_{1:2}V_{1:2}^{T}")
    else:
        model = st.radio("Compare", ["K-nearest neighbors", "Multinomial Naive Bayes"], horizontal=True, key="compare_classifier")
        train,test = train_test_split(np.arange(len(texts)), test_size=.3, random_state=42, stratify=labels)
        vec = vectorizer().fit([texts[i] for i in train])
        if model.startswith("K-"):
            k = st.slider("Neighbors", 1, 11, 5, key="compare_k")
            split = text_projection_split(texts, labels, train, test, analyzer=vec.tokenizer)
            Xtr, Xte = split["train"], split["test"]
            scratch = [compute_knn_neighbors(Xtr, labels[train], x, k)["prediction"] for x in Xte]
            reference = KNeighborsClassifier(n_neighbors=k, algorithm="brute").fit(Xtr, labels[train]).predict(Xte)
            st.caption("Same train-only TF-IDF → three terms → train-only LDA. Uniform votes, Euclidean distance. Equal-distance neighbor ordering can differ between implementations.")
        else:
            alpha = st.select_slider("Laplace smoothing α", [.1,.5,1.,2.], value=1., key="compare_alpha")
            C = vec.transform_raw_counts(texts)
            classes = sorted(set(labels))
            scratch = [compute_naive_bayes_posteriors(C[train], labels[train], np.repeat(np.arange(C.shape[1]), C[i].astype(int)).tolist(), classes, alpha)["predicted_class"] for i in test]
            reference = MultinomialNB(alpha=alpha).fit(C[train], labels[train]).predict(C[test])
            st.caption("Same training vocabulary, raw term counts, empirical class priors, smoothing and repeated query tokens. No validation labels enter fitting.")
        scratch=np.asarray(scratch)
        a,b,c = st.columns(3)
        a.metric("Scratch validation accuracy", f"{np.mean(scratch==labels[test]):.0%}")
        b.metric("Reference validation accuracy", f"{np.mean(reference==labels[test]):.0%}")
        c.metric("Prediction agreement", f"{np.mean(scratch==reference):.0%}")
        classes=sorted(set(labels)); lookup={c:i for i,c in enumerate(classes)}
        fig=go.Figure()
        for predictions,name,color,symbol in [(scratch,"Scratch",BLUE,"circle"),(reference,"Reference",GREEN,"x")]:
            fig.add_trace(go.Scatter(x=[str(i) for i in test], y=[lookup[c] for c in predictions], mode="markers", name=name, marker=dict(color=color,symbol=symbol,size=12)))
        fig.update_layout(xaxis_title="Held-out row", yaxis=dict(tickvals=list(range(len(classes))),ticktext=classes,title="Predicted domain"))
        chart(fig,350)
        st.caption("A fixed 44/20 stratified split (seed 42). Agreement checks implementations; it does not establish strong predictive performance.")
