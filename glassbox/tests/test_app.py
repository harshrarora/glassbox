"""Real Streamlit interaction contracts for the open pipeline playground."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
APP = Path(__file__).resolve().parents[1] / "app.py"


def app_at(stage):
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    if stage != 1:
        app.button(key=f"stage_{stage}").click().run()
    return app


@pytest.mark.parametrize("stage", range(1,7))
def test_all_six_stages_are_directly_accessible(stage):
    app = app_at(stage)
    assert not app.exception
    assert app.session_state["current_stage"] == stage
    assert all(app.button(key=f"stage_{i}") is not None for i in range(1,7))


@pytest.mark.parametrize("stage,key", [(3,"shift"),(4,"rotation"),(5,"leakage"),(6,"influence")])
def test_failure_spotlights_are_live_without_a_prediction(stage,key):
    app = app_at(stage)
    if stage != 6:
        app.radio(key=f"lens_widget_{stage}").set_value("Failure spotlight").run()
    assert not app.exception
    assert len(app.metric) >= 3
    assert app.radio(key=f"{key}_prediction").value is None
    assert not any(button.key == f"{key}_run" for button in app.button)
    assert app.session_state[f"{key}_run_record"]["prediction"] is None


def test_intervention_updates_results_immediately():
    app = app_at(3)
    app.radio(key="lens_widget_3").set_value("Failure spotlight").run()
    before = app.metric[0].value
    app.checkbox(key="shift_refresh").uncheck().run()
    assert not app.exception
    assert app.metric[0].value != before
    assert len(app.metric) == 3


def test_pipeline_configuration_survives_navigation():
    app = app_at(2)
    app.checkbox(key="prep_stop").uncheck().run()
    app.button(key="stage_3").click().run()
    assert app.session_state["pipeline"]["stopwords"] is False
    app.checkbox(key="vector_bigrams").check().run()
    app.button(key="stage_4").click().run()
    assert not app.exception
    assert app.session_state["pipeline"]["bigrams"] is True
    app.button(key="stage_2").click().run()
    assert app.checkbox(key="prep_stop").value is False


@pytest.mark.parametrize("stage", [3,4,5])
def test_live_reference_views_render(stage):
    app = app_at(stage)
    app.radio(key=f"lens_widget_{stage}").set_value("Compare implementations").run()
    assert not app.exception
    assert len(app.metric) == 3


def test_cell_microscope_and_reference_use_same_ngram_setting():
    app = app_at(3)
    app.checkbox(key="vector_bigrams").check().run()
    app.radio(key="lens_widget_3").set_value("Magnify a coordinate").run()
    assert not app.exception
    assert len(app.latex) >= 4
    app.radio(key="lens_widget_3").set_value("Compare implementations").run()
    assert app.checkbox(key="vector_bigrams").value is True
    assert not app.exception


def test_direct_spotlight_link():
    app = AppTest.from_file(str(APP),default_timeout=45)
    app.query_params["view"]="rotation"
    app.run()
    assert not app.exception
    assert app.session_state["current_stage"] == 4
    assert app.radio(key="lens_widget_4").value == "Failure spotlight"


def test_classifier_evaluation_has_a_separate_view():
    app=app_at(5)
    app.radio(key="lens_widget_5").set_value("Evaluation").run()
    assert not app.exception
    assert any("Evaluation Split" in radio.label for radio in app.radio)


def test_naive_bayes_reference_comparison():
    app=app_at(5)
    app.radio(key="lens_widget_5").set_value("Compare implementations").run()
    app.radio(key="compare_classifier").set_value("Multinomial Naive Bayes").run()
    assert not app.exception
    assert app.metric[2].value=="100%"
