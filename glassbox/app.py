"""GlassBox: The Interactive First-Principles ML Simulator.

Master 6-Stage Pipeline Stepper & Session State Router.
"""

import sys
import os

# Ensure local package importability
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import streamlit as st
from glassbox.utils.visual_components import render_stepper_bar
from glassbox.modules.stage1_corpus import render_stage1
from glassbox.modules.stage2_preprocess import render_stage2
from glassbox.modules.stage3_vectorize import render_stage3
from glassbox.modules.stage4_reduction import render_stage4
from glassbox.modules.stage5_classifiers import render_stage5
from glassbox.modules.stage6_model_mri import render_stage6

# Global Page Config
st.set_page_config(
    page_title="GlassBox — First-Principles ML Simulator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "current_stage" not in st.session_state:
    st.session_state.current_stage = 1

# Sidebar Controls
with st.sidebar:
    st.markdown("## 🔬 GlassBox")
    st.markdown("**First-Principles ML Simulator**")
    st.caption("UML501 (Machine Learning) & UCS551 (Accelerated Data Science)")
    st.markdown("---")

    st.markdown("#### Pipeline Stepper Navigation")
    stage_options = [
        "Stage 1: The Raw Corpus Hub",
        "Stage 2: Preprocessing & Linguistics Lab",
        "Stage 3: Vectorization & Feature Engineering",
        "Stage 4: Linear Algebra & Reduction",
        "Stage 5: Supervised Classifier Arena",
        "Stage 6: Model MRI & Failure Debugger",
    ]

    selected_idx = st.radio(
        "Current Pipeline Stage:",
        options=list(range(1, 7)),
        format_func=lambda i: stage_options[i - 1],
        index=st.session_state.current_stage - 1,
    )
    if selected_idx != st.session_state.current_stage:
        st.session_state.current_stage = selected_idx
        st.rerun()

    st.markdown("---")
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("⬅️ Previous", disabled=(st.session_state.current_stage == 1), use_container_width=True):
            st.session_state.current_stage -= 1
            st.rerun()
    with col_next:
        if st.button("Next ➡️", disabled=(st.session_state.current_stage == 6), use_container_width=True):
            st.session_state.current_stage += 1
            st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.8rem; color:#64748b; line-height:1.5;">
          <b>Syllabus Integration:</b><br>
          • Tokenization & Stemming vs Lemmatization<br>
          • Sparsity & Smoothed TF-IDF<br>
          • Covariance, PCA & SVD Rank-k<br>
          • Multiclass LDA (S_W & S_B)<br>
          • SVM Margin Gutters & C Slack<br>
          • Decision Tree Gini Splits & KNN<br>
          • Empirical Influence Functions
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render Top Horizontal Stepper Bar
render_stepper_bar(st.session_state.current_stage)

# Route to Active Pipeline Stage
if st.session_state.current_stage == 1:
    render_stage1()
elif st.session_state.current_stage == 2:
    render_stage2()
elif st.session_state.current_stage == 3:
    render_stage3()
elif st.session_state.current_stage == 4:
    render_stage4()
elif st.session_state.current_stage == 5:
    render_stage5()
elif st.session_state.current_stage == 6:
    render_stage6()
