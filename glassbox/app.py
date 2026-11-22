"""GlassBox: Lesson 1 — Vectorization, Geometric PCA, Evidence Retrieval, and Shift.

Interactive mathematical teaching lab for university machine learning and data science instruction.
"""

import sys
import os

# Ensure local glassbox package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from glassbox.core.vectorizer import TextVectorizer
from glassbox.core.geometry import fit_pca_3d, PCA3DResult
from glassbox.core.retrieval import (
    EvidenceRetrieval,
    SearchResult,
    QueryStatus,
)
from glassbox.data.fixture import (
    ResearchAbstract,
    DEFAULT_TEACHING_CLAIM,
    DEFAULT_PCA_TERMS,
    get_teaching_fixture,
    get_original_corpus,
    get_incoming_batch,
)

# -----------------------------------------------------------------------------
# Configuration and Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GlassBox — From Text to Evidence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Educational Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .step-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #e0e7ff;
        color: #3730a3;
        margin-right: 8px;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .highlight-doc {
        border-left: 4px solid #2563eb;
        background-color: #eff6ff;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    .highlight-term {
        background-color: #fef08a;
        color: #854d0e;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        border: 1px solid #facc15;
    }
    .annotation-trap {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
        padding: 10px;
        border-radius: 6px;
        font-size: 0.9rem;
    }
    .annotation-support {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #166534;
        padding: 10px;
        border-radius: 6px;
        font-size: 0.9rem;
    }
    .annotation-comp {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        color: #334155;
        padding: 10px;
        border-radius: 6px;
        font-size: 0.9rem;
    }
    .question-box {
        background-color: #fefce8;
        border: 1px solid #fef08a;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 14px;
        color: #713f12;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1

if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = "DOC-01"

if "selected_term" not in st.session_state:
    st.session_state.selected_term = "synaptic"

if "matrix_view_type" not in st.session_state:
    st.session_state.matrix_view_type = "L2-Normalized TF-IDF"

if "pca_stage" not in st.session_state:
    st.session_state.pca_stage = 1

if "query_input" not in st.session_state:
    st.session_state.query_input = DEFAULT_TEACHING_CLAIM

if "reveal_annotations" not in st.session_state:
    st.session_state.reveal_annotations = False

if "shift_include_incoming" not in st.session_state:
    st.session_state.shift_include_incoming = False

if "shift_refresh_idf" not in st.session_state:
    st.session_state.shift_refresh_idf = True

if "shift_target_focus" not in st.session_state:
    st.session_state.shift_target_focus = "DOC-03"


# -----------------------------------------------------------------------------
# Cached Data & Vectorizer Fitters
# -----------------------------------------------------------------------------
@st.cache_data
def get_corpora():
    all_abstracts = get_teaching_fixture()
    orig_abstracts = get_original_corpus()
    incoming_abstracts = get_incoming_batch()
    return all_abstracts, orig_abstracts, incoming_abstracts


all_docs, orig_docs, incoming_docs = get_corpora()
orig_texts = [d.abstract for d in orig_docs]
all_texts = [d.abstract for d in all_docs]
orig_ids = [d.doc_id for d in orig_docs]
all_ids = [d.doc_id for d in all_docs]
doc_id_to_doc = {d.doc_id: d for d in all_docs}


@st.cache_resource
def get_fitted_baseline_vectorizer():
    vec = TextVectorizer()
    vec.fit(orig_texts)
    c, z, x = vec.transform(orig_texts)
    return vec, c, z, x


vec_orig, c_orig, z_orig, x_orig = get_fitted_baseline_vectorizer()


@st.cache_resource
def get_refreshed_idf_vectorizer():
    # Frozen vocabulary from baseline, refreshed DF and IDF across all 10 texts
    vec_ref = TextVectorizer()
    vec_ref.feature_names = list(vec_orig.feature_names)
    vec_ref.vocabulary = dict(vec_orig.vocabulary)
    vec_ref.n_documents = len(all_texts)

    v_size = len(vec_orig.feature_names)
    df_ref = np.zeros(v_size, dtype=np.int64)
    for t_doc in [vec_ref.tokenizer(d) for d in all_texts]:
        for term in set(t_doc):
            if term in vec_ref.vocabulary:
                df_ref[vec_ref.vocabulary[term]] += 1
    vec_ref.doc_frequencies = df_ref
    vec_ref.idf_weights = np.log((1.0 + vec_ref.n_documents) / (1.0 + df_ref)) + 1.0
    vec_ref.is_fitted = True
    return vec_ref


vec_refreshed = get_refreshed_idf_vectorizer()


# -----------------------------------------------------------------------------
# Sidebar Navigation and Global Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔬 GlassBox Lab")
    st.markdown("**Course Modules:** UML501 (ML) / UCS551 (ADS)")
    st.markdown("---")

    st.markdown("#### Lesson 1 Roadmap")
    step_labels = [
        "1. Text to Matrix",
        "2. 3D Geometric PCA",
        "3. Evidence Retrieval",
        "4. Evidence Under Shift",
    ]

    selected_step_radio = st.radio(
        "Current Step:",
        options=[1, 2, 3, 4],
        format_func=lambda s: step_labels[s - 1],
        index=st.session_state.step - 1,
    )
    if selected_step_radio != st.session_state.step:
        st.session_state.step = selected_step_radio
        st.rerun()

    st.markdown("---")
    st.markdown("#### Classroom Navigation")
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("⬅️ Back", disabled=(st.session_state.step == 1), use_container_width=True):
            st.session_state.step -= 1
            st.rerun()
    with col_nav2:
        if st.button("Next ➡️", disabled=(st.session_state.step == 4), use_container_width=True):
            st.session_state.step += 1
            st.rerun()

    if st.button("🔄 Reset to Preset", use_container_width=True):
        st.session_state.step = 1
        st.session_state.selected_doc_id = "DOC-01"
        st.session_state.selected_term = "synaptic"
        st.session_state.matrix_view_type = "L2-Normalized TF-IDF"
        st.session_state.pca_stage = 1
        st.session_state.query_input = DEFAULT_TEACHING_CLAIM
        st.session_state.reveal_annotations = False
        st.session_state.shift_include_incoming = False
        st.session_state.shift_refresh_idf = True
        st.session_state.shift_target_focus = "DOC-03"
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.8rem; color:#64748b; line-height:1.4;">
        <b>Teaching Fixture:</b><br>
        • 7 Original Abstracts (N=7)<br>
        • 3 Incoming Abstracts<br>
        • Synthetic corpus for rigorous mathematical transparency.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Step Header
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown(
        f'<div class="main-header"><span class="step-pill">Step {st.session_state.step} of 4</span>'
        f'{step_labels[st.session_state.step - 1]}</div>',
        unsafe_allow_html=True,
    )

with header_col2:
    st.markdown(
        f'<div style="text-align:right; padding-top:10px; color:#64748b; font-size:0.9rem;">'
        f'Active Document: <b>{st.session_state.selected_doc_id}</b> | Term: <b>{st.session_state.selected_term}</b></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# =============================================================================
# STEP 1: Turn text into a matrix
# =============================================================================
if st.session_state.step == 1:
    st.markdown(
        '<div class="sub-header">Select any document and vocabulary term to inspect exact counts, '
        "relative frequency, smoothed IDF, and L2 normalization side by side.</div>",
        unsafe_allow_html=True,
    )

    # Top Controls for Step 1
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 1.6])
    with ctrl_col1:
        doc_options = orig_ids
        doc_idx_current = doc_options.index(st.session_state.selected_doc_id) if st.session_state.selected_doc_id in doc_options else 0
        new_doc_id = st.selectbox(
            "Select Document (Abstract):",
            options=doc_options,
            index=doc_idx_current,
            format_func=lambda did: f"{did}: {doc_id_to_doc[did].title[:35]}...",
        )
        if new_doc_id != st.session_state.selected_doc_id:
            st.session_state.selected_doc_id = new_doc_id
            st.rerun()

    with ctrl_col2:
        vocab_options = vec_orig.feature_names
        term_idx_current = vocab_options.index(st.session_state.selected_term) if st.session_state.selected_term in vocab_options else 0
        new_term = st.selectbox("Select Vocabulary Term:", options=vocab_options, index=term_idx_current)
        if new_term != st.session_state.selected_term:
            st.session_state.selected_term = new_term
            st.rerun()

    with ctrl_col3:
        st.session_state.matrix_view_type = st.radio(
            "Matrix Numerical View:",
            options=["Raw Counts", "TF-IDF (Unnormalized)", "L2-Normalized TF-IDF"],
            index=["Raw Counts", "TF-IDF (Unnormalized)", "L2-Normalized TF-IDF"].index(st.session_state.matrix_view_type),
            horizontal=True,
        )

    col_doc_view, col_matrix_view, col_inspector = st.columns([1.1, 1.5, 1.4])

    active_doc = doc_id_to_doc[st.session_state.selected_doc_id]
    active_doc_idx = orig_ids.index(st.session_state.selected_doc_id)

    # 1. Document Text Panel with Linked Term Highlighting
    with col_doc_view:
        st.markdown(f"#### 📄 {active_doc.doc_id}: Title & Text")
        st.markdown(f"**{active_doc.title}**")
        st.caption(f"Category: `{active_doc.category}` | Synthetic Corpus")

        # Highlight occurrences of selected term in text
        highlighted_text = active_doc.abstract
        sel_term = st.session_state.selected_term
        import re
        pattern = re.compile(rf"\b({re.escape(sel_term)})\b", re.IGNORECASE)
        highlighted_text = pattern.sub(r'<mark class="highlight-term">\1</mark>', highlighted_text)

        st.markdown(
            f'<div class="highlight-doc">{highlighted_text}</div>',
            unsafe_allow_html=True,
        )
        st.info(f"Term **'{sel_term}'** occurrences are highlighted above in yellow.")

    # 2. Document-Term Matrix Panel
    with col_matrix_view:
        st.markdown(f"#### 📊 Document-Term Matrix: {st.session_state.matrix_view_type}")

        # Choose between featured teaching columns vs full vocabulary
        show_all_cols = st.checkbox("Show full vocabulary (all 90+ terms)", value=False)
        featured_terms = [
            "synaptic", "consolidation", "decay", "sleep", "associative",
            "stabilizes", "memory", "slow", "spindles", "plasticity"
        ]
        # Ensure selected term is always in displayed columns
        if sel_term not in featured_terms and not show_all_cols:
            display_terms = [sel_term] + featured_terms
        else:
            display_terms = vec_orig.feature_names if show_all_cols else featured_terms

        # Select data matrix according to view toggle
        if st.session_state.matrix_view_type == "Raw Counts":
            matrix_data = c_orig
            format_str = "{:.0f}"
        elif st.session_state.matrix_view_type == "TF-IDF (Unnormalized)":
            matrix_data = z_orig
            format_str = "{:.3f}"
        else:
            matrix_data = x_orig
            format_str = "{:.4f}"

        # Build DataFrame for display
        df_display = pd.DataFrame(
            matrix_data,
            index=orig_ids,
            columns=vec_orig.feature_names,
        )[display_terms]

        # Style selected row, selected column, and selected cell
        def highlight_matrix_cells(df):
            styler_df = pd.DataFrame("", index=df.index, columns=df.columns)
            for r in df.index:
                for c in df.columns:
                    is_row = (r == st.session_state.selected_doc_id)
                    is_col = (c == sel_term)
                    if is_row and is_col:
                        styler_df.loc[r, c] = "background-color: #facc15; color: #000; font-weight: bold; border: 2px solid #ca8a04;"
                    elif is_row:
                        styler_df.loc[r, c] = "background-color: #dbeafe; font-weight: 500;"
                    elif is_col:
                        styler_df.loc[r, c] = "background-color: #fef08a;"
            return styler_df

        st.dataframe(
            df_display.style.apply(highlight_matrix_cells, axis=None).format(format_str),
            use_container_width=True,
            height=320,
        )
        st.caption(
            f"Highlighted: Row = **{st.session_state.selected_doc_id}** (blue), "
            f"Column = **{sel_term}** (yellow), Intersecting Cell = Gold."
        )

    # 3. Mathematical Inspector Panel
    with col_inspector:
        st.markdown("#### 📐 Mathematical Cell Inspector")

        info = vec_orig.inspect_cell(
            active_doc.abstract,
            sel_term,
            c_orig,
            z_orig,
            x_orig,
            active_doc_idx,
        )

        st.markdown(
            f"""
            <div class="metric-card">
              <div style="font-weight:600; font-size:1.05rem; margin-bottom:8px;">
                Cell: [{st.session_state.selected_doc_id}, '{sel_term}']
              </div>
              <table style="width:100%; font-size:0.9rem; border-collapse:collapse;">
                <tr><td><b>Raw Count</b> <code>c(t, d)</code>:</td><td style="text-align:right;"><b>{info['raw_count']}</b></td></tr>
                <tr><td><b>Relative Frequency</b> <code>f(t, d)</code>:</td><td style="text-align:right;">{info['raw_count']} / {info['doc_token_count']} = <b>{info['relative_freq']:.4f}</b></td></tr>
                <tr><td colspan="2" style="font-size:0.75rem; color:#64748b; padding-bottom:6px;">Denominator: <i>{info['denominator_name']}</i></td></tr>
                <tr style="border-top:1px solid #e2e8f0;"><td><b>Corpus Size</b> <code>N</code>:</td><td style="text-align:right;"><b>{info['N']}</b> documents</td></tr>
                <tr><td><b>Document Frequency</b> <code>DF(t)</code>:</td><td style="text-align:right;"><b>{info['DF']}</b> documents</td></tr>
                <tr style="border-top:1px solid #e2e8f0;"><td><b>Smoothed IDF</b> <code>idf(t)</code>:</td><td style="text-align:right;"><b>{info['idf']:.4f}</b></td></tr>
                <tr><td colspan="2" style="font-size:0.75rem; color:#475569; padding-bottom:6px;"><code>{info['idf_formula']}</code></td></tr>
                <tr style="border-top:1px solid #e2e8f0;"><td><b>Unnormalized TF-IDF</b> <code>z(t, d)</code>:</td><td style="text-align:right;">{info['raw_count']} × {info['idf']:.4f} = <b>{info['unnormalized_tfidf']:.4f}</b></td></tr>
                <tr><td><b>Document L2 Norm</b> <code>||z(:, d)||₂</code>:</td><td style="text-align:right;"><b>{info['doc_l2_norm']:.4f}</b></td></tr>
                <tr style="border-top:1px solid #cbd5e1; font-weight:bold; color:#1d4ed8;">
                  <td><b>L2-Normalized TF-IDF</b> <code>x(t, d)</code>:</td>
                  <td style="text-align:right;">{info['unnormalized_tfidf']:.4f} / {info['doc_l2_norm']:.4f} = <b>{info['normalized_tfidf']:.4f}</b></td>
                </tr>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="question-box">
          <b>Pedagogical Observation for Students:</b><br>
          Notice how the unnormalized TF-IDF weight <code>z(t, d) = c(t, d) × idf(t)</code> scales linearly with document length.
          L2 normalization divides each row by its Euclidean length <code>||z(:, d)||₂</code>, transforming documents into unit-length vectors on a hypersphere.
          This ensures cosine similarity later measures conceptual topic direction rather than superficial document verbosity.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# STEP 2: Project three term coordinates into two dimensions
# =============================================================================
elif st.session_state.step == 2:
    st.markdown(
        '<div class="sub-header">Dimensionality Reduction: Projecting 3 normalized term coordinates '
        "('synaptic', 'consolidation', 'decay') onto their top 2 Principal Components (PC1 & PC2).</div>",
        unsafe_allow_html=True,
    )

    pca_terms = DEFAULT_PCA_TERMS
    pca_indices = [vec_orig.vocabulary[t] for t in pca_terms]
    X_3d = x_orig[:, pca_indices]

    pca_result: PCA3DResult = fit_pca_3d(X_3d, pca_terms)

    # Stage Selector for Teacher
    stage_names = [
        "1. Original 3D Positions",
        "2. Mean Vector & Centering",
        "3. PC1–PC2 Subspace Plane",
        "4. Orthogonal Projections",
        "5. 2D Projected Scores",
    ]
    st.session_state.pca_stage = st.radio(
        "Interactive Classroom Stage:",
        options=[1, 2, 3, 4, 5],
        format_func=lambda s: stage_names[s - 1],
        index=st.session_state.pca_stage - 1,
        horizontal=True,
    )

    col_geom_plot, col_geom_math = st.columns([1.6, 1.4])

    doc_colors = {
        "DOC-01": "#2563eb",  # Blue
        "DOC-02": "#16a34a",  # Green
        "DOC-03": "#9333ea",  # Purple (Target)
        "DOC-04": "#dc2626",  # Red
        "DOC-05": "#ea580c",  # Orange
        "DOC-06": "#0891b2",  # Cyan
        "DOC-07": "#475569",  # Slate
    }

    with col_geom_plot:
        st.markdown(f"#### 🌐 Visualization: Stage {st.session_state.pca_stage}")

        if st.session_state.pca_stage in [1, 2, 3, 4]:
            fig = go.Figure()

            # 1. Plot original or centered points
            is_centered_view = (st.session_state.pca_stage >= 2)
            points = pca_result.X_centered if is_centered_view else pca_result.X_original
            frame_title = "Centered Coordinate Frame (Xc = X - μ)" if is_centered_view else "Original Coordinate Frame (Normalized TF-IDF)"

            for i, d_id in enumerate(orig_ids):
                is_selected = (d_id == st.session_state.selected_doc_id)
                pt = points[i]
                fig.add_trace(
                    go.Scatter3d(
                        x=[pt[0]],
                        y=[pt[1]],
                        z=[pt[2]],
                        mode="markers+text",
                        marker=dict(
                            size=12 if is_selected else 8,
                            color=doc_colors[d_id],
                            symbol="circle",
                            line=dict(color="#000" if is_selected else "#fff", width=3 if is_selected else 1),
                        ),
                        text=[f"<b>{d_id}</b>" if is_selected else d_id],
                        textposition="top center",
                        name=f"{d_id} ({doc_id_to_doc[d_id].category})",
                    )
                )

            # Stage 2+: Mean Vector
            if st.session_state.pca_stage == 2:
                # Mark origin in centered frame
                fig.add_trace(
                    go.Scatter3d(
                        x=[0], y=[0], z=[0],
                        mode="markers+text",
                        marker=dict(size=10, color="#f59e0b", symbol="diamond"),
                        text=["Origin (Mean μ)"],
                        textposition="bottom center",
                        name="Centered Mean μ",
                    )
                )

            # Stage 3 & 4: Subspace plane and PC1/PC2 vectors
            if st.session_state.pca_stage in [3, 4]:
                # Draw PC1 and PC2 direction vectors from origin
                v1 = pca_result.eigenvectors[:, 0] * 0.25
                v2 = pca_result.eigenvectors[:, 1] * 0.25

                fig.add_trace(
                    go.Scatter3d(
                        x=[0, v1[0]], y=[0, v1[1]], z=[0, v1[2]],
                        mode="lines+text",
                        line=dict(color="#10b981", width=6),
                        text=["", f"PC1 (52.4%)"],
                        name="PC1 Vector",
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=[0, v2[0]], y=[0, v2[1]], z=[0, v2[2]],
                        mode="lines+text",
                        line=dict(color="#6366f1", width=6),
                        text=["", f"PC2 (42.8%)"],
                        name="PC2 Vector",
                    )
                )

                # Plane mesh grid spanning [-0.25, 0.25]
                grid_u = np.linspace(-0.28, 0.28, 11)
                grid_v = np.linspace(-0.28, 0.28, 11)
                U, V_mesh = np.meshgrid(grid_u, grid_v)
                v1_full = pca_result.eigenvectors[:, 0]
                v2_full = pca_result.eigenvectors[:, 1]
                plane_x = U * v1_full[0] + V_mesh * v2_full[0]
                plane_y = U * v1_full[1] + V_mesh * v2_full[1]
                plane_z = U * v1_full[2] + V_mesh * v2_full[2]

                fig.add_trace(
                    go.Surface(
                        x=plane_x, y=plane_y, z=plane_z,
                        opacity=0.35,
                        colorscale=[[0, "#93c5fd"], [1, "#c7d2fe"]],
                        showscale=False,
                        name="PC1-PC2 Subspace Plane",
                    )
                )

            # Stage 4: Orthogonal Projection Lines
            if st.session_state.pca_stage == 4:
                # In centered frame, projected point is scores @ V2^T
                proj_centered = pca_result.scores_2d @ pca_result.eigenvectors[:, :2].T
                for i, d_id in enumerate(orig_ids):
                    is_selected = (d_id == st.session_state.selected_doc_id)
                    pt_orig = pca_result.X_centered[i]
                    pt_proj = proj_centered[i]

                    # Line connecting 3D point to projected point
                    fig.add_trace(
                        go.Scatter3d(
                            x=[pt_orig[0], pt_proj[0]],
                            y=[pt_orig[1], pt_proj[1]],
                            z=[pt_orig[2], pt_proj[2]],
                            mode="lines",
                            line=dict(
                                color="#d97706" if is_selected else "#94a3b8",
                                width=7 if is_selected else 2,
                                dash="solid" if is_selected else "dash",
                            ),
                            showlegend=False,
                        )
                    )
                    # Marker at projected point on plane
                    fig.add_trace(
                        go.Scatter3d(
                            x=[pt_proj[0]],
                            y=[pt_proj[1]],
                            z=[pt_proj[2]],
                            mode="markers",
                            marker=dict(
                                size=10 if is_selected else 5,
                                color="#f59e0b" if is_selected else doc_colors[d_id],
                                symbol="cross",
                            ),
                            showlegend=False,
                        )
                    )

            # Layout with stable ranges
            r_limit = 0.42 if is_centered_view else 0.45
            axis_min = -r_limit if is_centered_view else 0.0
            axis_max = r_limit if is_centered_view else 0.45
            fig.update_layout(
                title=frame_title,
                scene=dict(
                    xaxis=dict(title=f"'{pca_terms[0]}'", range=[axis_min, axis_max]),
                    yaxis=dict(title=f"'{pca_terms[1]}'", range=[axis_min, axis_max]),
                    zaxis=dict(title=f"'{pca_terms[2]}'", range=[axis_min, axis_max]),
                    camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            # Stage 5: 2D Projected Scores
            fig_2d = go.Figure()
            for i, d_id in enumerate(orig_ids):
                is_selected = (d_id == st.session_state.selected_doc_id)
                sc = pca_result.scores_2d[i]
                fig_2d.add_trace(
                    go.Scatter(
                        x=[sc[0]],
                        y=[sc[1]],
                        mode="markers+text",
                        marker=dict(
                            size=16 if is_selected else 10,
                            color=doc_colors[d_id],
                            line=dict(color="#000" if is_selected else "#fff", width=2),
                        ),
                        text=[f"<b>{d_id}</b>" if is_selected else d_id],
                        textposition="top right",
                        name=f"{d_id}: {doc_id_to_doc[d_id].title[:25]}",
                    )
                )

            fig_2d.update_layout(
                title="2D Projected Scores on PC1 & PC2 (scores = Xc @ V2)",
                xaxis=dict(
                    title=f"PC1 (Explained Variance: {pca_result.explained_variance_ratio[0]*100:.2f}%)",
                    zeroline=True,
                    zerolinecolor="#cbd5e1",
                ),
                yaxis=dict(
                    title=f"PC2 (Explained Variance: {pca_result.explained_variance_ratio[1]*100:.2f}%)",
                    zeroline=True,
                    zerolinecolor="#cbd5e1",
                ),
                height=480,
                margin=dict(l=20, r=20, b=20, t=40),
            )
            st.plotly_chart(fig_2d, use_container_width=True)

    with col_geom_math:
        st.markdown("#### 📐 Mathematical Derivation & Values")

        st.latex(r"X_c = X - \mu, \quad \Sigma = \frac{X_c^T X_c}{n - 1}")
        st.latex(r"\text{scores} = X_c V_2, \quad \tilde{X} = \text{scores} V_2^T + \mu")

        st.markdown(
            f"""
            <div class="metric-card">
              <b>1. Mean Vector μ:</b><br>
              <code>['{pca_terms[0]}', '{pca_terms[1]}', '{pca_terms[2]}'] = [{pca_result.mean_vector[0]:.4f}, {pca_result.mean_vector[1]:.4f}, {pca_result.mean_vector[2]:.4f}]</code><br><br>
              <b>2. Sample Covariance Matrix Σ (3×3):</b>
              <pre style="margin:4px 0; font-size:0.8rem;">{np.array2string(pca_result.covariance_matrix, precision=4, suppress_small=True)}</pre>
              <b>3. Eigenvalues & Explained Variance:</b><br>
              • λ₁ = <b>{pca_result.eigenvalues[0]:.4f}</b> ({pca_result.explained_variance_ratio[0]*100:.2f}%)<br>
              • λ₂ = <b>{pca_result.eigenvalues[1]:.4f}</b> ({pca_result.explained_variance_ratio[1]*100:.2f}%)<br>
              • λ₃ = <b>{pca_result.eigenvalues[2]:.4f}</b> ({pca_result.explained_variance_ratio[2]*100:.2f}%)<br>
              <i>Cumulative 2D Variance Captured:</i> <b>{(pca_result.explained_variance_ratio[0]+pca_result.explained_variance_ratio[1])*100:.2f}%</b><br><br>
              <b>4. Eigenvectors (Principal Component Loadings V):</b>
              <pre style="margin:4px 0; font-size:0.8rem;">{np.array2string(pca_result.eigenvectors, precision=4, suppress_small=True)}</pre>
              <span style="font-size:0.75rem; color:#64748b;">
                Jacobi Solver: Converged in {pca_result.jacobi_iterations} sweeps (tol=1e-12). Subspace isotropic: {pca_result.is_degenerate}.
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Selected document inspector
        active_idx = orig_ids.index(st.session_state.selected_doc_id)
        pt_orig = pca_result.X_original[active_idx]
        pt_cent = pca_result.X_centered[active_idx]
        pt_scores = pca_result.scores_2d[active_idx]
        pt_recon = pca_result.projected_3d[active_idx]
        dist_residual = np.linalg.norm(pca_result.residuals_3d[active_idx])

        st.markdown(
            f"""
            <div class="metric-card" style="border-left:4px solid #2563eb;">
              <b>Focus Document: {st.session_state.selected_doc_id}</b><br>
              • Original coordinates: <code>[{pt_orig[0]:.4f}, {pt_orig[1]:.4f}, {pt_orig[2]:.4f}]</code><br>
              • Centered coordinates: <code>[{pt_cent[0]:.4f}, {pt_cent[1]:.4f}, {pt_cent[2]:.4f}]</code><br>
              • 2D Projected Scores (PC1, PC2): <code>[{pt_scores[0]:.4f}, {pt_scores[1]:.4f}]</code><br>
              • 3D Reconstructed point: <code>[{pt_recon[0]:.4f}, {pt_recon[1]:.4f}, {pt_recon[2]:.4f}]</code><br>
              • Orthogonal Residual Distance: <b>{dist_residual:.4f}</b> (distance to 2D plane)
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="question-box">
          <b>Pedagogical Observation for Students:</b><br>
          Notice that PC1 and PC2 together capture <b>95.22%</b> of the total variance across the 7 papers, meaning
          the orthogonal residual error (distance from each 3D point to the plane) is remarkably small.
          The 2D projection preserves almost all geometric relational information between papers.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# STEP 3: Rank documents and explain one score
# =============================================================================
elif st.session_state.step == 3:
    st.markdown(
        '<div class="sub-header">Full-Vector Cosine Retrieval: Ranking literature against a hypothesis claim, '
        "reconciling exact term contributions, and uncovering semantic vs lexical traps.</div>",
        unsafe_allow_html=True,
    )

    # Claim Input Box
    claim_col1, claim_col2 = st.columns([3.5, 1.0])
    with claim_col1:
        user_query = st.text_input(
            "Hypothesis Claim to Retrieve Evidence:",
            value=st.session_state.query_input,
            help="Type any claim or keep the default teaching hypothesis.",
        )
        if user_query != st.session_state.query_input:
            st.session_state.query_input = user_query
            st.rerun()

    with claim_col2:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("Reset Default Claim", use_container_width=True):
            st.session_state.query_input = DEFAULT_TEACHING_CLAIM
            st.rerun()

    st.session_state.reveal_annotations = st.checkbox(
        "👁️ Reveal Fixture Expert Annotation (Compare Lexical Score vs Ground Truth)",
        value=st.session_state.reveal_annotations,
    )

    retrieval_engine = EvidenceRetrieval(vec_orig)
    status, msg, search_results = retrieval_engine.search(
        st.session_state.query_input,
        orig_ids,
        [d.title for d in orig_docs],
        orig_texts,
        [d.annotation for d in orig_docs],
        x_orig,
        c_orig,
    )

    if status == QueryStatus.BLANK:
        st.warning(f"⚠️ {msg}")
    elif status == QueryStatus.NO_VOCAB_OVERLAP:
        st.error(f"❌ {msg}")
        st.info("The vocabulary was fitted on the 7 baseline abstracts. Try terms like 'synaptic', 'consolidation', 'sleep', 'associative', or 'memory'.")
    else:
        # Valid retrieval results
        col_rank_list, col_breakdown = st.columns([1.5, 1.5])

        with col_rank_list:
            st.markdown("#### 🏆 Ranked Document Candidates (Cosine Similarity)")

            # Render ranked items
            for res in search_results:
                is_selected = (res.doc_id == st.session_state.selected_doc_id)
                badge_color = "#2563eb" if res.score > 0.4 else ("#0284c7" if res.score > 0.2 else "#64748b")

                with st.container():
                    st.markdown(
                        f"""
                        <div style="border: 1px solid {'#2563eb' if is_selected else '#e2e8f0'};
                                    background-color: {'#eff6ff' if is_selected else '#ffffff'};
                                    border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                          <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:700; font-size:1.0rem;">
                              Rank #{res.rank} — {res.doc_id}
                            </span>
                            <span style="background-color:{badge_color}; color:#fff; padding:2px 8px; border-radius:12px; font-size:0.85rem; font-weight:600;">
                              Similarity: {res.score:.4f}
                            </span>
                          </div>
                          <div style="font-size:0.9rem; color:#1e293b; margin:4px 0;">{res.title}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button(f"Inspect {res.doc_id}", key=f"btn_inspect_{res.doc_id}"):
                        st.session_state.selected_doc_id = res.doc_id
                        st.rerun()

        with col_breakdown:
            st.markdown(f"#### 🔍 Term Contribution Waterfall: {st.session_state.selected_doc_id}")

            # Find selected result
            selected_res = next((r for r in search_results if r.doc_id == st.session_state.selected_doc_id), search_results[0])
            sel_doc_obj = doc_id_to_doc[selected_res.doc_id]

            st.markdown(f"**{sel_doc_obj.title}**")
            st.markdown(f"*{sel_doc_obj.abstract}*")

            # Show annotation if requested
            if st.session_state.reveal_annotations:
                if sel_doc_obj.category == "lexical_trap":
                    st.markdown(
                        f'<div class="annotation-trap"><b>⚠️ Lexical Trap:</b> {sel_doc_obj.annotation}</div>',
                        unsafe_allow_html=True,
                    )
                elif sel_doc_obj.category == "supporting":
                    st.markdown(
                        f'<div class="annotation-support"><b>✅ Ground Truth Support:</b> {sel_doc_obj.annotation}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="annotation-comp"><b>ℹ️ Comparison Abstract:</b> {sel_doc_obj.annotation}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown(
                f"""
                <div style="margin:10px 0; font-size:0.9rem;">
                  <b>Cosine Similarity Score:</b> <code>{selected_res.score:.4f}</code><br>
                  <b>Sum of Term Contributions:</b> <code>{selected_res.sum_contributions:.4f}</code>
                  (Reconciliation Delta: <code>{abs(selected_res.score - selected_res.sum_contributions):.2e}</code>)
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Waterfall or horizontal bar chart of term contributions
            if selected_res.matching_terms:
                t_terms = [t.term for t in selected_res.matching_terms]
                t_contribs = [t.contribution for t in selected_res.matching_terms]

                fig_contrib = px.bar(
                    x=t_contribs[::-1],
                    y=t_terms[::-1],
                    orientation="h",
                    labels={"x": "Contribution: q_norm(t) × d_norm(t)", "y": "Term"},
                    title=f"Per-Term Cosine Contribution Breakdown for {selected_res.doc_id}",
                    color=t_contribs[::-1],
                    color_continuous_scale="Blues",
                )
                fig_contrib.update_layout(height=260, margin=dict(l=20, r=20, b=20, t=35), showlegend=False)
                st.plotly_chart(fig_contrib, use_container_width=True)

                # Detailed contribution table
                df_tc = pd.DataFrame([
                    {
                        "Term": tc.term,
                        "Query Count": tc.query_raw_count,
                        "Query Weight q̂": f"{tc.query_normalized_weight:.4f}",
                        "Doc Count": tc.doc_raw_count,
                        "Doc Weight d̂": f"{tc.doc_normalized_weight:.4f}",
                        "Product (Contribution)": f"{tc.contribution:.4f}",
                    }
                    for tc in selected_res.matching_terms
                ])
                st.dataframe(df_tc, use_container_width=True, hide_index=True)
            else:
                st.info("No matching vocabulary terms between this document and the query claim.")

        st.markdown(
            """
            <div class="question-box">
              <b>Pedagogical Observation for Students:</b><br>
              Look at <b>DOC-04</b> ("Awake Synaptic Consolidation During Novelty Exploration Destabilizes Memory Traces").
              It achieves a very high cosine similarity score (Rank #3 or #4) purely because it matches multiple query words
              ('synaptic', 'consolidation', 'during', 'stabilizes', 'associative', 'memory').
              Yet its actual finding directly contradicts the hypothesis claim!
              This illustrates why lexical similarity is not semantic truth.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# STEP 4: Evidence Under Shift
# =============================================================================
elif st.session_state.step == 4:
    st.markdown(
        '<div class="sub-header">Evidence Under Shift: Can the same paper fall out of the shortlist '
        "when its own text and candidate eligibility remain fixed?</div>",
        unsafe_allow_html=True,
    )

    # Preset & Control Panel
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.2, 1.2])

    with ctrl_col1:
        if st.button("⭐ Apply Teaching Preset: Hide new papers, refresh IDF", use_container_width=True):
            st.session_state.shift_include_incoming = False
            st.session_state.shift_refresh_idf = True
            st.session_state.shift_target_focus = "DOC-03"
            st.rerun()

    with ctrl_col2:
        if st.button("Reset to Baseline (7 candidates, frozen IDF)", use_container_width=True):
            st.session_state.shift_include_incoming = False
            st.session_state.shift_refresh_idf = False
            st.rerun()

    with ctrl_col3:
        st.session_state.shift_target_focus = st.selectbox(
            "Focus Document to Inspect:",
            options=orig_ids,
            index=orig_ids.index(st.session_state.shift_target_focus) if st.session_state.shift_target_focus in orig_ids else 2,
        )

    # Independent Toggles
    tog_col1, tog_col2 = st.columns(2)
    with tog_col1:
        st.session_state.shift_include_incoming = st.checkbox(
            "Include incoming papers in candidate search results",
            value=st.session_state.shift_include_incoming,
            help="If checked, the 3 incoming papers (DOC-08 to DOC-10) are eligible to appear in search rankings.",
        )
    with tog_col2:
        st.session_state.shift_refresh_idf = st.checkbox(
            "Recompute IDF using expanded 10-document corpus statistics",
            value=st.session_state.shift_refresh_idf,
            help="If checked, DF and IDF are recalculated across all 10 papers, diluting terms like 'consolidation'.",
        )

    # Compute Baseline Results
    ret_baseline = EvidenceRetrieval(vec_orig)
    _, _, results_baseline = ret_baseline.search(
        DEFAULT_TEACHING_CLAIM,
        orig_ids,
        [d.title for d in orig_docs],
        orig_texts,
        [d.annotation for d in orig_docs],
        x_orig,
        c_orig,
    )

    # Compute Updated Shift Results
    active_shift_vec = vec_refreshed if st.session_state.shift_refresh_idf else vec_orig
    shift_candidate_docs = all_docs if st.session_state.shift_include_incoming else orig_docs
    shift_candidate_ids = [d.doc_id for d in shift_candidate_docs]
    shift_candidate_texts = [d.abstract for d in shift_candidate_docs]
    shift_candidate_titles = [d.title for d in shift_candidate_docs]
    shift_candidate_annotations = [d.annotation for d in shift_candidate_docs]

    c_shift, z_shift, x_shift = active_shift_vec.transform(shift_candidate_texts)
    ret_shift = EvidenceRetrieval(active_shift_vec)
    _, _, results_updated = ret_shift.search(
        DEFAULT_TEACHING_CLAIM,
        shift_candidate_ids,
        shift_candidate_titles,
        shift_candidate_texts,
        shift_candidate_annotations,
        x_shift,
        c_shift,
    )

    # Side-by-Side Comparison
    col_base_top5, col_upd_top5 = st.columns(2)

    with col_base_top5:
        st.markdown("#### 🏛️ Baseline Top 5 (Frozen IDF, N=7)")
        for r in results_baseline[:5]:
            is_focus = (r.doc_id == st.session_state.shift_target_focus)
            st.markdown(
                f"""
                <div style="border:1px solid {'#9333ea' if is_focus else '#cbd5e1'};
                            background-color:{'#faf5ff' if is_focus else '#ffffff'};
                            border-radius:6px; padding:8px 12px; margin-bottom:6px;">
                  <b>Rank #{r.rank} — {r.doc_id}</b> (Score: <code>{r.score:.4f}</code>)
                  {' <span style=\"color:#9333ea; font-weight:bold;\">★ FOCUS TARGET</span>' if is_focus else ''}<br>
                  <span style="font-size:0.85rem; color:#475569;">{r.title}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_upd_top5:
        st.markdown(
            f"#### ⚡ Experimental Top 5 "
            f"({'Expanded Candidates' if st.session_state.shift_include_incoming else 'Fixed 7 Candidates'}, "
            f"{'Refreshed IDF' if st.session_state.shift_refresh_idf else 'Frozen IDF'})"
        )
        base_rank_map = {r.doc_id: r.rank for r in results_baseline}

        for r in results_updated[:5]:
            is_focus = (r.doc_id == st.session_state.shift_target_focus)
            b_rank = base_rank_map.get(r.doc_id)
            if b_rank is None:
                mov_badge = "<span style='color:#059669; font-weight:bold;'>[NEW CANDIDATE]</span>"
            elif b_rank > r.rank:
                mov_badge = f"<span style='color:#059669; font-weight:bold;'>[Rose {b_rank - r.rank} place{'s' if b_rank - r.rank > 1 else ''}]</span>"
            elif b_rank < r.rank:
                mov_badge = f"<span style='color:#dc2626; font-weight:bold;'>[Fell {r.rank - b_rank} place{'s' if r.rank - b_rank > 1 else ''}]</span>"
            else:
                mov_badge = "<span style='color:#64748b;'>[Unchanged]</span>"

            st.markdown(
                f"""
                <div style="border:1px solid {'#9333ea' if is_focus else '#cbd5e1'};
                            background-color:{'#faf5ff' if is_focus else '#ffffff'};
                            border-radius:6px; padding:8px 12px; margin-bottom:6px;">
                  <b>Rank #{r.rank} — {r.doc_id}</b> (Score: <code>{r.score:.4f}</code>) {mov_badge}<br>
                  <span style="font-size:0.85rem; color:#475569;">{r.title}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Deep-Dive on Target Document Demotion
    st.markdown("---")
    focus_did = st.session_state.shift_target_focus
    st.markdown(f"#### 🔬 Detailed Impact Analysis for Focus Document: **{focus_did}**")

    # Metrics before and after
    b_res = next((r for r in results_baseline if r.doc_id == focus_did), None)
    u_res = next((r for r in results_updated if r.doc_id == focus_did), None)

    b_rank_str = f"Rank #{b_res.rank} (in Top 5)" if b_res and b_res.rank <= 5 else (f"Rank #{b_res.rank}" if b_res else "N/A")
    u_rank_str = f"Rank #{u_res.rank} (in Top 5)" if u_res and u_res.rank <= 5 else (f"<b style='color:#dc2626;'>Rank #{u_res.rank} (OUT OF TOP 5)</b>" if u_res else "N/A")

    shift_col1, shift_col2 = st.columns([1.5, 1.5])
    with shift_col1:
        # Check DF and IDF of key term 'consolidation'
        idx_cons = vec_orig.vocabulary.get("consolidation", 0)
        df_base_cons = int(vec_orig.doc_frequencies[idx_cons])
        df_upd_cons = int(active_shift_vec.doc_frequencies[idx_cons])
        idf_base_cons = float(vec_orig.idf_weights[idx_cons])
        idf_upd_cons = float(active_shift_vec.idf_weights[idx_cons])

        table_data = [
            {"Parameter / Metric": "Corpus Documents N", "Baseline": "7", "Shifted": f"{active_shift_vec.n_documents}", "Net Change": f"+{active_shift_vec.n_documents - 7}"},
            {"Parameter / Metric": "DF('consolidation')", "Baseline": f"{df_base_cons}", "Shifted": f"{df_upd_cons}", "Net Change": f"+{df_upd_cons - df_base_cons} (diluted)"},
            {"Parameter / Metric": "IDF('consolidation')", "Baseline": f"{idf_base_cons:.4f}", "Shifted": f"{idf_upd_cons:.4f}", "Net Change": f"{idf_upd_cons - idf_base_cons:.4f}"},
            {"Parameter / Metric": f"{focus_did} Cosine Score", "Baseline": f"{b_res.score:.4f}" if b_res else "-", "Shifted": f"{u_res.score:.4f}" if u_res else "-", "Net Change": f"{(u_res.score - b_res.score):.4f}" if (b_res and u_res) else "-"},
            {"Parameter / Metric": f"{focus_did} Standing", "Baseline": b_rank_str, "Shifted": u_rank_str, "Net Change": f"Fell {u_res.rank - b_res.rank} places" if (b_res and u_res and u_res.rank > b_res.rank) else "Unchanged"},
        ]
        st.markdown(pd.DataFrame(table_data).to_html(classes="table", index=False, escape=False), unsafe_allow_html=True)

    with shift_col2:
        st.markdown(
            f"""
            <div class="metric-card">
              <div style="font-weight:700; margin-bottom:6px; color:#1e293b;">Mathematical Explanation:</div>
              <ol style="margin-bottom:0; font-size:0.88rem; line-height:1.5;">
                <li><b>IDF Dilution:</b> The incoming batch (DOC-08 to DOC-10) contains unrelated discussions of 'consolidation' (orthopedic bone consolidation and fiscal policy).
                    This drives <code>DF('consolidation')</code> up from 4 to 7, reducing its IDF from <b>{idf_base_cons:.4f}</b> to <b>{idf_upd_cons:.4f}</b>.</li>
                <li><b>Vector Redistribution:</b> Because <b>{focus_did}</b> relies primarily on 'consolidation', its unnormalized coordinate decreases.
                    L2 normalization redistributes vector energy, weakening its match with the query.</li>
                <li><b>Shortlist Displacement:</b> Meanwhile, competitor document <b>DOC-06</b> has terms like 'stabilizes', 'associative', and 'memory' whose DF was not diluted.
                    DOC-06's score rises relative to {focus_did}, overtaking it to claim the 5th spot!</li>
              </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Full Ranked Candidate Explorer
    st.markdown("---")
    with st.expander("📋 View Complete Candidate Rankings Table (All Positions)"):
        all_comp_data = []
        for r in results_updated:
            b_r = base_rank_map.get(r.doc_id)
            all_comp_data.append({
                "Doc ID": r.doc_id,
                "Updated Rank": r.rank,
                "Baseline Rank": b_r if b_r else "None (New)",
                "Score": f"{r.score:.4f}",
                "Title": r.title,
                "Category": doc_id_to_doc[r.doc_id].category,
            })
        st.dataframe(pd.DataFrame(all_comp_data), use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="question-box">
          <b>Key Pedagogical Takeaway for Students:</b><br>
          <i>"Can a paper fall out of the top five when its own text and the candidate list stay fixed?"</i><br>
          <b>YES.</b> Evidence retrieval is not an absolute judgment of a document in isolation; it is a relative judgment
          conditioned on global corpus statistics. Diluting an informative term's rarity globally shifts vector geometry
          and reshuffles the ranking of untouched papers.
        </div>
        """,
        unsafe_allow_html=True,
    )
