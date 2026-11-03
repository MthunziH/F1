"""Offline tests on a synthetic championship (no network)."""
import json

import numpy as np
import pandas as pd
import pytest

from f1predict import config, predictor
from f1predict.features import FEATURES, build_features
from f1predict.model import PodiumModel, _normalise, predicted_podium
from f1predict.sources import HISTORY_COLUMNS, best_quali_time, lap_time_seconds, team_to_constructor_id

TEAMS = ["red_bull", "mclaren", "ferrari", "mercedes", "aston_martin",
         "alpine", "williams", "rb", "sauber", "haas"]


def synthetic_history(seasons=(2022, 2023, 2024), rounds=12, seed=0):
    rng = np.random.default_rng(seed)
    drivers = [(f"D{i:02d}", f"d{i:02d}", TEAMS[i // 2]) for i in range(20)]
    skill = {code: 2.0 - i * 0.1 for i, (code, _, _) in enumerate(drivers)}
    pts_table = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
    rows = []
    for season in seasons:
        for rnd in range(1, rounds + 1):
            pace = {c: skill[c] + rng.normal(0, 0.4) for c, _, _ in drivers}
            quali_order = sorted(pace, key=pace.get, reverse=True)
            race_perf = {c: pace[c] + rng.normal(0, 0.5) for c in pace}
            dnf = {c: rng.random() < 0.08 for c in pace}
            finish = sorted(pace, key=lambda c: (dnf[c], -race_perf[c]))
            pole_time = 80.0
            for code, did, team in drivers:
                pos = finish.index(code) + 1
                qpos = quali_order.index(code) + 1
                rows.append({
                    "season": season, "round": rnd, "date": f"{season}-01-01",
                    "eventName": f"GP {rnd}", "circuitId": f"c{rnd}",
                    "driverId": did, "driverCode": code, "driverName": f"Driver {code}",
                    "constructorId": team, "constructorName": team.title(),
                    "quali_pos": qpos, "q_best_s": pole_time + 0.1 * (qpos - 1),
                    "grid": qpos, "position": pos,
                    "positionText": "R" if dnf[code] else str(pos),
                    "points": 0 if dnf[code] or pos > 10 else pts_table[pos - 1],
                    "status": "Retired" if dnf[code] else "Finished", "source": "ergast",
                })
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


@pytest.fixture
def tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PREDICTIONS_DIR", tmp_path / "predictions")
    monkeypatch.setattr(config, "SITE_DIR", tmp_path / "site")
    monkeypatch.setattr(config, "BACKTEST_JSON", tmp_path / "backtest.json")
    return tmp_path


def test_parsing_helpers():
    assert lap_time_seconds("1:23.456") == pytest.approx(83.456)
    assert lap_time_seconds(pd.Timedelta(seconds=90.5)) == pytest.approx(90.5)
    assert np.isnan(lap_time_seconds(pd.NaT))
    assert best_quali_time({"Q1": "1:30.0", "Q2": "1:29.5", "Q3": None}) == pytest.approx(89.5)
    assert team_to_constructor_id("Red Bull Racing") == "red_bull"
    assert team_to_constructor_id("Haas F1 Team") == "haas"


def test_normalise():
    p = _normalise(np.array([0.9, 0.9, 0.9, 0.5, 0.1]), 3.0)
    assert p.sum() == pytest.approx(3.0)
    assert p.max() <= 1.0


def test_features_use_only_past():
    hist = synthetic_history(seasons=(2023,), rounds=3)
    f = build_features(hist)
    first = f[f["race_key"] == 202301]
    assert first["d_finish5"].isna().all()  # no prior races
    second = f[(f["race_key"] == 202302) & (f["driverCode"] == "D00")]
    expected = first.loc[first["driverCode"] == "D00", "finish"].iloc[0]
    assert second["d_finish5"].iloc[0] == expected
    assert set(FEATURES) <= set(f.columns)


def test_model_learns_and_predicts(tmp_dirs):
    hist = synthetic_history()
    target_race = hist[(hist["season"] == 2024) & (hist["round"] == 12)]
    quali = target_race[["driverCode", "driverId", "driverName", "constructorId",
                         "constructorName", "quali_pos", "q_best_s"]]
    train_hist = hist[~((hist["season"] == 2024) & (hist["round"] == 12))]
    target = predictor.prepare_target(train_hist, quali, 2024, 12, "GP 12", "c12")
    pred, n = predictor.predict_race(train_hist, target)
    assert n == 35
    assert pred["p_win"].sum() == pytest.approx(1.0, abs=1e-6)
    assert pred["p_podium"].sum() == pytest.approx(3.0, abs=0.05)
    assert (pred["p_podium"] >= pred["p_win"] - 1e-9).all()
    # Pole sitter should be among the favourites in a world where pace matters.
    pole = quali.sort_values("quali_pos")["driverCode"].iloc[0]
    assert pole in pred["driverCode"].head(3).tolist()

    rec = predictor.prediction_record(pred, {"season": 2024, "round": 12, "eventName": "GP 12"})
    assert len(set(rec["podium"])) == 3
    predictor.save_prediction(rec)
    scored = predictor.score_pending(hist)
    assert len(scored) == 1 and scored[0]["score"] is not None


def test_backtest_beats_random_and_site_builds(tmp_dirs):
    from f1predict.site import build_site

    hist = synthetic_history()
    bt = predictor.backtest(hist, from_season=2024)
    assert bt["summary"]["overall"]["model"]["races"] == 12
    assert bt["summary"]["overall"]["model"]["podium_hits_avg"] > 1.0
    config.BACKTEST_JSON.write_text(json.dumps(bt))
    build_site({"season": 2025, "round": 1, "eventName": "GP 1", "quali_utc": "2025-03-01T05:00:00Z"})
    html = (config.SITE_DIR / "index.html").read_text()
    assert "Backtest" in html and "GP 1" in html
