"""GlassBox: a reactive six-stage machine-learning playground."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st
from glassbox.utils.design import install_design, eyebrow
from glassbox.utils.pipeline import settings
from glassbox.modules.stage1_corpus import render_stage1
from glassbox.modules.stage2_preprocess import render_stage2
from glassbox.modules.stage3_vectorize import render_stage3
from glassbox.modules.stage4_reduction import render_stage4
from glassbox.modules.stage5_classifiers import render_stage5
from glassbox.modules.stage6_model_mri import render_stage6

st.set_page_config(page_title="GlassBox | ML Playground", page_icon="◩", layout="wide", initial_sidebar_state="collapsed")
install_design()
STAGES = ["Corpus", "Preprocess", "Vectorize", "Geometry", "Classify", "Diagnose"]
DESCRIPTIONS = ["Start with the evidence.", "See what survives preprocessing.", "Turn words into measurable geometry.", "Rotate, compress and separate.", "Compare how models draw the line.", "Put the explanation to the test."]
st.session_state.setdefault("current_stage", 1)
st.session_state.setdefault("show_methods", False)
if "query_loaded" not in st.session_state:
    st.session_state.query_loaded = True
    presets = {"shift": (3, "Failure spotlight"), "rotation": (4, "Failure spotlight"), "leakage": (5, "Failure spotlight"), "influence": (6, "Playground")}
    preset = presets.get(st.query_params.get("view", ""))
    if preset:
        st.session_state.current_stage = preset[0]
        st.session_state[f"lens_{preset[0]}"] = preset[1]


def navigate(stage):
    st.session_state.current_stage = stage
    st.session_state.show_methods = False


def toggle_methods():
    st.session_state.show_methods = not st.session_state.show_methods

brand, context, notes = st.columns([1.4, 4, 1.1], vertical_alignment="center")
with brand:
    st.markdown('<div class="brand"><span class="brand-mark">◩</span> GlassBox</div>', unsafe_allow_html=True)
with context:
    st.caption("MACHINE LEARNING PLAYGROUND  /  manipulate · magnify · compare")
with notes:
    st.button("Methods & sources", on_click=toggle_methods, width="stretch")

with st.container(key="pipeline_navigation"):
    for index, column in enumerate(st.columns(6), 1):
        with column:
            st.button(f"{index:02d}  {STAGES[index-1]}", key=f"stage_{index}", on_click=navigate, args=(index,),
                      type="primary" if st.session_state.current_stage == index else "secondary", width="stretch")

cfg = settings()
pipeline_summary = st.empty()

def methods():
    eyebrow("METHODS / CLAIMS YOU CAN CHECK")
    st.title("Show the work.")
    st.write("GlassBox makes mathematical operations inspectable. It does not claim to reimplement every numerical primitive or to benchmark real-world scientific classification.")
    st.markdown("""
| Component | Implementation boundary |
| :--- | :--- |
| Token counts, smoothed IDF, normalization, cosine contributions | Project Python / NumPy |
| Covariance and 3D PCA | Project arithmetic and Jacobi eigensolver |
| LDA | Project scatter matrices; NumPy linear algebra |
| Logistic regression and influence | Project Newton updates, gradients and Hessian; NumPy linear solve |
| K-means, MLP, KNN, Bayes, split impurity, metrics | Project teaching implementations |
| SVD decomposition | NumPy; reconstruction and residuals exposed |
| SVM and fitted decision trees | scikit-learn reference estimators |
| Hierarchical clustering | SciPy linkage |
| Morphology | NLTK when available; explicitly labeled fallback rules otherwise |
""")
    st.subheader("Data and scope")
    st.write("The ranking investigation uses ten synthetic abstracts. The rotation investigation uses exact constructed covariance matrices. The leakage control uses independent noise and random labels. The paper corpus contains 64 curated teaching texts associated with research papers; source abstracts and bibliographic metadata have not been fully verified. Category labels are teaching assignments, not a claim about official arXiv categories.")
    st.subheader("Numerical checks")
    st.caption("Four scoped reference / invariant checks on the bundled corpus. These are observations on this input, not a universal precision guarantee.")
    if st.button("Run numerical checks", key="run_audit"):
        from glassbox.data.arxiv_corpus import get_arxiv_corpus
        from glassbox.utils.math_engine import verify_first_principles_vs_sklearn
        papers = get_arxiv_corpus()
        st.session_state.audit = verify_first_principles_vs_sklearn([p.abstract for p in papers], [p.category for p in papers])
    for record in st.session_state.get("audit", []):
        with st.expander(f"{'PASS' if record['passed'] else 'FAIL'} / {record['test']}"):
            st.write(f"Measured difference: {record['diff']:.3e}. Tolerance: {record['tol']:.1e}.")
            st.write(f"Implementation: {record['scratch_desc']}. Comparison: {record['sklearn_desc']}.")
    st.subheader("Reading")
    st.markdown("[Influence functions: Koh & Liang (2017)](https://proceedings.mlr.press/v70/koh17a.html)  \n[Data leakage: scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)")


if st.session_state.show_methods:
    methods()
else:
    current = st.session_state.current_stage
    eyebrow(f"STAGE {current:02d} / {STAGES[current-1].upper()}")
    st.title(DESCRIPTIONS[current-1])
    [render_stage1, render_stage2, render_stage3, render_stage4, render_stage5, render_stage6][current-1]()
    st.divider()
    prev, hint, nxt = st.columns([1, 3, 1])
    with prev:
        st.button("← Previous", on_click=navigate, args=(max(1, current-1),), disabled=current == 1, width="stretch")
    with hint:
        st.caption("Move freely. Charts respond immediately; explanations are optional.")
    with nxt:
        st.button("Next →", on_click=navigate, args=(min(6, current+1),), disabled=current == 6, width="stretch")

pipeline_summary.caption(f"64 teaching texts  ·  Regex {'on' if cfg['regex'] else 'off'}  ·  Stopwords {'removed' if cfg['stopwords'] else 'kept'}  ·  {cfg['morphology']} morphology  ·  {'Unigrams + bigrams' if cfg['bigrams'] else 'Unigrams'}")
