"""Persistent preprocessing configuration shared by the pipeline stages."""
from functools import partial
import streamlit as st
import numpy as np

from glassbox.core.vectorizer import TextVectorizer, default_tokenizer
from glassbox.utils.math_engine import clean_regex_text, filter_stopwords, LEMMATIZER, STEMMER

DEFAULTS = {"regex": True, "stopwords": True, "morphology": "None", "bigrams": False,
            "terms": ["image", "language", "optimization"], "rank": 2}


def settings():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = {**DEFAULTS, "terms": list(DEFAULTS["terms"])}
    return st.session_state.pipeline


def update_setting(name, widget_key):
    settings()[name] = st.session_state[widget_key]


def analyze(text, regex=True, stopwords=True, morphology="None", bigrams=False):
    cleaned = clean_regex_text(text)[0] if regex else text
    tokens = default_tokenizer(cleaned)
    if stopwords:
        tokens = filter_stopwords(tokens)[0]
    if morphology == "Lemma":
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    elif morphology == "Stem":
        tokens = [STEMMER.stem(t) for t in tokens]
    if bigrams:
        return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    return tokens


def vectorizer(config=None):
    cfg = settings() if config is None else config
    return TextVectorizer(tokenizer=partial(analyze, **{k: cfg[k] for k in ("regex", "stopwords", "morphology", "bigrams")}))


def resolve_terms(vec, terms):
    return [t if t in vec.vocabulary else next(iter(vec.tokenizer(t)), "") for t in terms]


def coordinates(vec, X, terms):
    """Never silently substitute a different column for an absent feature."""
    return np.column_stack([X[:, vec.vocabulary[t]] if t in vec.vocabulary else np.zeros(len(X)) for t in resolve_terms(vec, terms)])


def lens_selector(stage, options):
    """Keep per-stage lenses independent and persistent through navigation."""
    state_key = f"lens_{stage}"
    st.session_state.setdefault(state_key, options[0])
    widget = f"lens_widget_{stage}"
    def remember():
        st.session_state[state_key] = st.session_state[widget]
    return st.radio("Explore this stage", options, index=options.index(st.session_state[state_key]),
                    horizontal=True, key=widget, on_change=remember, label_visibility="collapsed")
