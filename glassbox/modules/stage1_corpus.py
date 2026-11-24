"""Corpus browser: inspect a paper alongside its position in the dataset."""
import streamlit as st
import plotly.graph_objects as go
from glassbox.data.arxiv_corpus import get_arxiv_corpus
from glassbox.utils.design import chart

COLORS = {"cs.CV":"#10b981", "cs.CL":"#6366f1", "cs.LG":"#f59e0b", "stat.ML":"#f43f5e"}

def render_stage1():
    papers=get_arxiv_corpus()
    lengths=[len(p.abstract.split()) for p in papers]
    a,b,c,d=st.columns(4)
    a.metric("Papers",len(papers))
    b.metric("Domains",4)
    c.metric("Raw vocabulary",f"{len(set(' '.join(p.abstract for p in papers).lower().split())):,}")
    d.metric("Mean words / text",f"{sum(lengths)/len(lengths):.0f}")
    left,right=st.columns([1.05,1.7],gap="large")
    with left:
        category=st.selectbox("Domain",["All domains"]+list(COLORS),key="corpus_category")
        query=st.text_input("Find a paper",placeholder="Title, author or keyword",key="corpus_search")
        options=[i for i,p in enumerate(papers) if (category=="All domains" or p.category==category) and query.lower() in (p.title+' '+p.abstract+' '+p.authors).lower()]
        if not options:
            st.info("No matching papers. Try a shorter query or another domain.")
            return
        selected=st.selectbox("Paper",options,format_func=lambda i:papers[i].title,key="corpus_paper")
        paper=papers[selected]
        st.caption(f"{paper.category} · {paper.year} · {paper.id}")
        st.write(f"**{paper.title}**")
        st.caption(paper.authors)
        st.write(paper.abstract)
    with right:
        st.subheader("Where this paper sits")
        fig=go.Figure()
        for cat,color in COLORS.items():
            indices=[i for i,p in enumerate(papers) if p.category==cat]
            fig.add_trace(go.Scatter(x=indices,y=[lengths[i] for i in indices],mode="markers",name=cat,text=[papers[i].title for i in indices],marker=dict(size=9,color=color),hovertemplate="%{text}<br>%{y} words<extra></extra>"))
        fig.add_trace(go.Scatter(x=[selected],y=[lengths[selected]],mode="markers",name="Selected paper",marker=dict(size=18,symbol="circle-open",color="#0f172a",line_width=2)))
        fig.update_layout(xaxis_title="Corpus row",yaxis_title="Text length (words)")
        chart(fig,410)
        st.caption("16 teaching records per domain. Selecting a paper highlights it on the canvas.")
    with st.expander("About this corpus"):
        st.write("These are 64 curated teaching texts associated with research papers. Source abstracts and bibliographic metadata have not been comprehensively verified. Categories are teaching assignments, not a claim about official arXiv classifications.")
