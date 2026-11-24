"""Reactive vectorization playground, cell microscope and aligned reference comparison."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from glassbox.data.arxiv_corpus import get_arxiv_corpus
from glassbox.utils.pipeline import settings, vectorizer, lens_selector
from glassbox.utils.design import chart


def render_stage3():
    lens = lens_selector(3, ["Playground", "Magnify a coordinate", "Compare implementations", "Failure spotlight"])
    if lens == "Failure spotlight":
        from glassbox.modules.investigations import render_shift
        render_shift()
        return
    cfg = settings()
    cfg["bigrams"] = st.checkbox("Include adjacent word pairs (bigrams)", value=cfg["bigrams"], key="vector_bigrams")
    if lens == "Compare implementations":
        from glassbox.modules.comparisons import render_comparison
        render_comparison(3)
        return
    papers = get_arxiv_corpus()
    texts = [p.abstract for p in papers]
    vec = vectorizer().fit(texts)
    C, Z, X = vec.transform(texts)
    a,b,c = st.columns(3)
    a.metric("Vocabulary", f"{len(vec.feature_names):,}")
    b.metric("Matrix shape", f"{len(papers)} × {len(vec.feature_names):,}")
    c.metric("Zero coordinates", f"{1-np.count_nonzero(C)/C.size:.1%}")
    left,right = st.columns([1,2.1],gap="large")
    with left:
        doc = st.selectbox("Document to inspect", range(len(papers)), format_func=lambda i: papers[i].title, key="vector_doc")
        available = np.flatnonzero(C[doc])
        term = st.selectbox("Term to magnify", [vec.feature_names[i] for i in available], key="vector_term")
        index = vec.vocabulary[term]
        mode = st.radio("Show matrix values as", ["Normalized TF-IDF", "Raw counts", "Count × IDF"], key="vector_mode")
        st.caption("Downstream geometry uses normalized TF-IDF. This switch inspects intermediate representations of the same data.")
        with st.expander("Inspect the input text"):
            st.write(papers[doc].abstract)
            st.caption("Preprocessing choices come from Stage 2.")
    with right:
        if lens == "Playground":
            matrix = X if mode == "Normalized TF-IDF" else C if mode == "Raw counts" else Z
            selected = list(np.argsort(C.sum(axis=0))[-19:])
            if index not in selected:
                selected.append(index)
            rows = list(range(max(0,doc-5),min(len(papers),doc+7)))
            fig=go.Figure(go.Heatmap(z=matrix[np.ix_(rows,selected)],x=[vec.feature_names[i] for i in selected],y=[str(i) for i in rows],colorscale="Blues",colorbar=dict(title="value"),hovertemplate="Document %{y}<br>Term %{x}<br>Value %{z:.5f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[term],y=[str(doc)],mode="markers",marker=dict(symbol="square-open",size=23,line=dict(color="#f43f5e",width=3)),showlegend=False))
            fig.update_layout(xaxis_title="Term",yaxis_title="Document row")
            chart(fig,400)
            st.caption("The rose outline marks your selected coordinate. The window follows your document; the full matrix is used for all calculations.")
        else:
            st.subheader(f'Inside the coordinate: “{term}”')
            count,df,idf = int(C[doc,index]),int(vec.doc_frequencies[index]),float(vec.idf_weights[index])
            norm=float(np.linalg.norm(Z[doc]))
            st.latex(rf"\mathrm{{tf}}={count},\qquad\mathrm{{df}}={df},\qquad N={len(papers)}")
            st.latex(rf"\mathrm{{idf}}=\log\frac{{1+{len(papers)}}}{{1+{df}}}+1={idf:.6f}")
            st.latex(rf"z={count}\times {idf:.6f}={Z[doc,index]:.6f}")
            st.latex(rf"x=\frac{{{Z[doc,index]:.6f}}}{{{norm:.6f}}}={X[doc,index]:.6f}")
            st.caption("The denominator depends on every term in this document, not just the selected term.")
            with st.expander("Inspect the implementation"):
                import inspect
                from glassbox.core.vectorizer import TextVectorizer
                st.code(inspect.getsource(TextVectorizer.transform),language="python")
    if lens == "Playground":
        st.info(f"Selected cell: count {int(C[doc,index])} × IDF {vec.idf_weights[index]:.4f} ÷ document norm {np.linalg.norm(Z[doc]):.4f} = {X[doc,index]:.6f}. Open ‘Magnify a coordinate’ to follow each operation.")
