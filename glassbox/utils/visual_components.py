"""Reusable visual UI components for GlassBox interactive simulation.

Implements:
1. Top Pipeline Stepper Bar
2. 'What & Why' Hero Cards
3. Living Visual Text Diff Highlighters (strikethrough red, root green)
4. Vocabulary Contraction Meter
5. Styled Metric Badges and Plotly visual layouts
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional


def render_stepper_bar(current_stage: int) -> None:
    """Renders the top horizontal pipeline stepper."""
    stages = [
        (1, "1. Raw Corpus Hub"),
        (2, "2. Preprocessing Lab"),
        (3, "3. Vectorization"),
        (4, "4. Dimensionality Reduction"),
        (5, "5. Classifier Arena"),
        (6, "6. Model MRI"),
    ]

    cols = st.columns(len(stages))
    for idx, (stage_num, stage_label) in enumerate(stages):
        with cols[idx]:
            is_active = (stage_num == current_stage)
            is_past = (stage_num < current_stage)

            bg_color = "#2563eb" if is_active else ("#10b981" if is_past else "#f1f5f9")
            text_color = "#ffffff" if (is_active or is_past) else "#64748b"
            border = "2px solid #1d4ed8" if is_active else ("1px solid #059669" if is_past else "1px solid #cbd5e1")
            icon = "🎯" if is_active else ("✓" if is_past else f"{stage_num}")

            st.markdown(
                f"""
                <div style="background:{bg_color}; color:{text_color}; border:{border};
                            border-radius:8px; padding:8px 4px; text-align:center;
                            font-size:0.8rem; font-weight:700; margin-bottom:12px;">
                  {icon} {stage_label}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_hero_card(stage_title: str, what_text: str, why_text: str) -> None:
    """Renders the golden 2-sentence 'What & Why' Hero Card."""
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                    border: 1px solid #334155; border-radius: 12px; padding: 18px 24px;
                    color: #f8fafc; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
          <div style="font-size: 1.5rem; font-weight: 800; color: #60a5fa; margin-bottom: 8px;">
            {stage_title}
          </div>
          <div style="font-size: 0.95rem; line-height: 1.6; margin-bottom: 8px;">
            <b style="color: #38bdf8;">💡 What is happening:</b> {what_text}
          </div>
          <div style="font-size: 0.95rem; line-height: 1.6;">
            <b style="color: #4ade80;">🎯 Why we are doing this:</b> {why_text}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_contraction_meter(contraction_stats: Dict[str, Any]) -> None:
    """Renders the dynamic visual bar showing vocabulary shrinking in real time."""
    v_raw = contraction_stats["v_raw"]
    v_regex = contraction_stats["v_regex"]
    v_stop = contraction_stats["v_stop"]
    v_lemma = contraction_stats["v_lemma"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("1. Raw Words", f"{v_raw:,}", delta="Unfiltered")
    with col2:
        st.metric("2. After Regex", f"{v_regex:,}", delta=f"-{v_raw - v_regex} words", delta_color="inverse")
    with col3:
        st.metric("3. Non-Stopwords", f"{v_stop:,}", delta=f"-{v_regex - v_stop} stop", delta_color="inverse")
    with col4:
        st.metric("4. Clean Lemmas", f"{v_lemma:,}", delta=f"Compressed {contraction_stats['total_compression_pct']:.1f}%", delta_color="normal")

    # Contraction Progress Visualization Bar
    pct_regex = (v_regex / v_raw) * 100 if v_raw else 100
    pct_stop = (v_stop / v_raw) * 100 if v_raw else 100
    pct_lemma = (v_lemma / v_raw) * 100 if v_raw else 100

    fig_bar = go.Figure()
    stages = ["Raw Words", "Post-Regex", "Post-Stopwords", "Post-Lemmas"]
    counts = [v_raw, v_regex, v_stop, v_lemma]
    colors = ["#94a3b8", "#38bdf8", "#f59e0b", "#10b981"]

    fig_bar.add_trace(
        go.Bar(
            x=stages,
            y=counts,
            text=[f"<b>{c:,}</b> ({c/v_raw*100:.1f}%)" for c in counts],
            textposition="auto",
            marker=dict(color=colors, line=dict(color="#0f172a", width=1.5)),
        )
    )
    fig_bar.update_layout(
        title="Vocabulary Contraction Funnel (Feature Dimension Reduction)",
        height=240,
        margin=dict(l=20, r=20, b=20, t=35),
        yaxis=dict(title="Unique Vocabulary Tokens"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


def render_living_diff_text(
    original_text: str,
    cleaned_text: str,
    removed_regex: List[str],
    discarded_stopwords: List[str],
    show_regex_diff: bool = True,
    show_stopword_diff: bool = True,
) -> str:
    """Creates a living visual diff highlighting stripped punctuation/latex in red strikethrough,
    and removed stopwords in amber/red tags.
    """
    import re

    # Tokenize words to inspect
    tokens = re.findall(r"\b\w+\b|[^\w\s]|\s+", original_text)
    html_spans = []

    stopword_set = set(discarded_stopwords)

    for tok in tokens:
        clean_t = tok.lower().strip()
        is_punct_or_sym = bool(re.match(r"[^a-zA-Z\s]", tok))
        is_stop = clean_t in stopword_set

        if show_regex_diff and is_punct_or_sym and tok.strip():
            html_spans.append(
                f'<del style="color:#ef4444; background:#fee2e2; padding:1px 3px; border-radius:3px;" '
                f'title="Stripped by Regex">{tok}</del>'
            )
        elif show_stopword_diff and is_stop:
            html_spans.append(
                f'<span style="color:#b45309; background:#fef3c7; text-decoration:line-through; padding:1px 4px; border-radius:3px;" '
                f'title="Filtered Stopword">{tok}</span>'
            )
        else:
            html_spans.append(tok)

    return "".join(html_spans)
