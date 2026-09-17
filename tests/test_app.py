from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_runs_offline_and_plays():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Offline demo").run()
    at.sidebar.button[0].click().run()          # Load ground
    assert not at.exception
    run_btn = [b for b in at.sidebar.button if b.label == "Run both COAs"][0]
    run_btn.click().run()
    assert not at.exception, at.exception
    assert at.session_state.runs is not None
    assert at.session_state.runs["manifest"]["code_version"]
    at.slider(key="tick").set_value(5).run()
    assert not at.exception
    at.segmented_control(key="quick_pick").set_value("A and B split").run()
    assert not at.exception
    at.segmented_control(key="loss_coa").set_value("B").run()
    assert not at.exception
    toggles = [tg for tg in at.toggle if "Simultaneous" in tg.label]
    toggles[0].set_value(True).run()
    assert not at.exception
    play = [b for b in at.button if b.label == "Play"][0]
    play.click().run()
    assert not at.exception, at.exception


def test_app_multi_plan_run():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    plans = [s for s in at.sidebar.slider if s.label == "Red plans sampled"][0]
    plans.set_value(2).run()
    reps = [s for s in at.sidebar.select_slider if s.label == "Replications per COA"][0]
    reps.set_value(200).run()
    run_btn = [b for b in at.sidebar.button if b.label == "Run both COAs"][0]
    run_btn.click().run()
    assert not at.exception, at.exception
    assert at.session_state.runs["res"]["A"].n_plans == 2
