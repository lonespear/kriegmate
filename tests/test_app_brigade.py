from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app_brigade.py")


def test_brigade_app_runs_and_solves_game():
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert not at.exception
    reps = [s for s in at.sidebar.select_slider if s.label == "Replications"][0]
    reps.set_value(10).run()
    [s for s in at.sidebar.slider if s.label == "Attacking brigades"][0].set_value(1).run()
    [b for b in at.sidebar.button if b.label == "Run this pairing"][0].click().run()
    assert not at.exception, at.exception
    assert at.session_state.bres is not None
    at.slider[-1].set_value(30).run()
    assert not at.exception
    [b for b in at.sidebar.button if b.label == "Solve the 3×3 COA game"][0].click().run()
    assert not at.exception, at.exception
    assert 0 <= at.session_state.bgame[0] <= 1
