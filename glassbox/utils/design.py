"""Shared editorial styling and interaction primitives for the investigation UI."""

import html
import json
import hashlib
import platform
from importlib.metadata import version
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

INK = "#0F172A"
GREEN = "#059669"
RUST = "#E11D48"
MUTED = "#64748B"
PAPER = "#F8FAFC"
BLUE = "#4F46E5"


def install_design():
    st.markdown(f"<style>{Path(__file__).with_name('theme.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    pio.templates["glassbox"] = go.layout.Template(layout={
        "font": {"family": "Arial, sans-serif", "size": 13, "color": INK},
        "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
        "colorway": [GREEN, RUST, BLUE, "#D97706"],
        "xaxis": {"gridcolor": "#E2E8F0", "zerolinecolor": "#CBD5E1", "automargin": True},
        "yaxis": {"gridcolor": "#E2E8F0", "zerolinecolor": "#CBD5E1", "automargin": True},
        "margin": {"l": 35, "r": 30, "t": 25, "b": 40},
        "legend": {"orientation": "h", "y": -0.18, "x": 0},
        "hoverlabel": {"bgcolor": "#FFFFFF", "font_color": INK},
    })
    pio.templates.default = "glassbox"


def eyebrow(text):
    st.markdown(f'<div class="eyebrow">{html.escape(text)}</div>', unsafe_allow_html=True)


def header(number, title, premise, concept):
    eyebrow(f"FAILURE SPOTLIGHT {number:02d} / {concept}")
    st.subheader(title.replace("\n", " "))
    st.markdown(f'<p class="standfirst">{html.escape(premise)}</p>', unsafe_allow_html=True)
    st.markdown('<div class="method-line">PREDICT <span>→</span> INTERVENE <span>→</span> EXPLAIN <span>→</span> VERIFY</div>', unsafe_allow_html=True)


def chart(fig, height=430):
    fig.update_layout(template="glassbox", height=height)
    st.plotly_chart(fig, use_container_width=True, theme=None,
                    config={"displaylogo": False, "displayModeBar": False, "scrollZoom": False})


def predict(key, question, choices):
    with st.expander("Thought prompt (optional)"):
        return st.radio(question, choices, index=None, key=f"{key}_prediction")


def live_experiment(key, signature, prediction, label=None):
    """Capture the currently rendered configuration. Never gate exploration."""
    st.session_state[f"{key}_run_record"] = {"signature": signature, "prediction": prediction}
    return True


def verdict(prediction, observation):
    prefix = f'<small>YOUR EXPECTATION</small><p>{html.escape(prediction)}</p>' if prediction else ""
    st.markdown(f'<div class="finding">{prefix}<small>OBSERVED NOW</small><p>{html.escape(observation)}</p></div>', unsafe_allow_html=True)


def receipt(key, parameters, results):
    completed = set(st.session_state.get("completed", []))
    completed.add(key)
    st.session_state["completed"] = sorted(completed)
    notes = ""
    record = st.session_state[f"{key}_run_record"]
    package = Path(__file__).resolve().parents[1]
    source_files = sorted(package.rglob("*.py"))
    source_hash = hashlib.sha256()
    for source in source_files:
        source_hash.update(source.relative_to(package).as_posix().encode())
        source_hash.update(source.read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode())
    payload = {"schema_version": 1, "experiment": key,
               "prediction": record["prediction"], "parameters": parameters,
               "results": results, "reflection": notes,
               "source_sha256": source_hash.hexdigest(),
               "environment": {"python": platform.python_version(), **{
                   name: version(name) for name in ["numpy", "scikit-learn", "streamlit", "plotly"]}}}
    st.download_button("Download experiment record", json.dumps(payload, indent=2),
                       file_name=f"glassbox-{key}.json", mime="application/json", key=f"{key}_download")


def source_note(text):
    st.caption(text)
