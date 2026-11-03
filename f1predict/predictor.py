"""Build predictions, score them against real results, and backtest."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from f1predict import config
from f1predict.features import build_features
from f1predict.model import PodiumModel, predicted_podium

log = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- target race

def prepare_target(hist: pd.DataFrame, quali: pd.DataFrame, season: int, rnd: int,
                   event_name: str, circuit_id: str | None) -> pd.DataFrame:
    """Turn a qualifying table into history-shaped rows for the target race,
    resolving driver and team ids to the ones used in the history."""
    t = quali.copy()
    t["season"], t["round"] = season, rnd
    t["eventName"], t["circuitId"] = event_name, circuit_id
    t["source"] = "target"

    by_code = (hist.sort_values(["season", "round"])
               .drop_duplicates("driverCode", keep="last").set_index("driverCode"))
    this_season = hist[hist["season"] == season]
    season_team = (this_season.sort_values("round")
                   .drop_duplicates("driverCode", keep="last").set_index("driverCode")["constructorId"])

    if "driverId" not in t:
        t["driverId"] = np.nan
    t["driverId"] = t["driverId"].fillna(t["driverCode"].map(by_code["driverId"]))
    t["driverId"] = t["driverId"].fillna(t["driverCode"].str.lower())

    # Team id: the driver's team earlier this season, else a teammate's, else
    # the id derived from the team name.
    team_from_hist = t["driverCode"].map(season_team)
    by_team_name = {}
    for name, grp in t.assign(_cid=team_from_hist).groupby("constructorName"):
        known = grp["_cid"].dropna()
        if not known.empty:
            by_team_name[name] = known.iloc[0]
    t["constructorId"] = team_from_hist.fillna(t["constructorName"].map(by_team_name)).fillna(t["constructorId"])
    return t


def predict_race(hist: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    key = int(target["season"].iloc[0]) * 100 + int(target["round"].iloc[0])
    past = hist[hist["season"] * 100 + hist["round"] < key]
    feats = build_features(past, target)
    model = PodiumModel().fit(feats[~feats["is_target"]])
    pred = model.predict_race(feats[feats["is_target"]])
    return pred, model.n_races


def prediction_record(pred: pd.DataFrame, meta: dict) -> dict:
    podium_idx = predicted_podium(pred)
    drivers = []
    for i, r in pred.iterrows():
        drivers.append({
            "code": r["driverCode"], "name": r["driverName"],
            "team": r["constructorName"], "teamId": r["constructorId"],
            "quali_pos": int(r["quali_pos"]) if pd.notna(r["quali_pos"]) else None,
            "p_win": round(float(r["p_win"]), 4), "p_podium": round(float(r["p_podium"]), 4),
        })
    return {
        **meta,
        "generated_utc": utc_now_iso(),
        "podium": [drivers[i]["code"] for i in podium_idx],
        "drivers": drivers,
        "actual": None,
        "score": None,
    }


# ---------------------------------------------------------------- storage

def prediction_path(season: int, rnd: int):
    return config.PREDICTIONS_DIR / f"{season}-{rnd:02d}.json"


def save_prediction(rec: dict) -> None:
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path(rec["season"], rec["round"]).write_text(json.dumps(rec, indent=2) + "\n")


def load_predictions() -> list[dict]:
    if not config.PREDICTIONS_DIR.exists():
        return []
    recs = [json.loads(p.read_text()) for p in sorted(config.PREDICTIONS_DIR.glob("*.json"))]
    return sorted(recs, key=lambda r: (r["season"], r["round"]))


# ---------------------------------------------------------------- scoring

def actual_podium(hist: pd.DataFrame, season: int, rnd: int) -> list[str] | None:
    race = hist[(hist["season"] == season) & (hist["round"] == rnd)].copy()
    if race.empty:
        return None
    race["position"] = pd.to_numeric(race["position"], errors="coerce")
    top = race[race["position"] <= 3].sort_values("position")
    return top["driverCode"].tolist() if len(top) == 3 else None


def score(predicted: list[str], actual: list[str]) -> dict:
    return {
        "winner": predicted[0] == actual[0],
        "podium_hits": len(set(predicted) & set(actual)),
        "exact": predicted == actual,
    }


def score_pending(hist: pd.DataFrame) -> list[dict]:
    """Attach real results to predictions whose race has finished."""
    newly_scored = []
    for rec in load_predictions():
        if rec.get("actual"):
            continue
        actual = actual_podium(hist, rec["season"], rec["round"])
        if not actual:
            continue
        rec["actual"] = {"podium": actual}
        rec["score"] = score(rec["podium"], actual)
        save_prediction(rec)
        newly_scored.append(rec)
    return newly_scored


# ---------------------------------------------------------------- backtest

def backtest(hist: pd.DataFrame, from_season: int = config.BACKTEST_FROM_SEASON) -> dict:
    """Replay each race since `from_season`: train only on earlier races,
    predict from that weekend's qualifying, compare with the real result."""
    feats = build_features(hist)
    keys = sorted(feats.loc[feats["season"] >= from_season, "race_key"].unique())
    races = []
    for key in keys:
        race = feats[feats["race_key"] == key]
        actual = actual_podium(hist, int(key // 100), int(key % 100))
        train = feats[feats["race_key"] < key]
        if not actual or train["race_key"].nunique() < 20:
            continue
        model = PodiumModel().fit(train)
        pred = model.predict_race(race)
        podium = pred.loc[predicted_podium(pred), "driverCode"].tolist()
        grid = race.sort_values("quali_pos")["driverCode"].head(3).tolist()
        winner_p = pred.loc[pred["driverCode"] == actual[0], "p_win"]
        races.append({
            "season": int(key // 100), "round": int(key % 100),
            "eventName": race["eventName"].iloc[0],
            "predicted": podium, "actual": actual, "grid_top3": grid,
            "p_actual_winner": round(float(winner_p.iloc[0]), 4) if len(winner_p) else None,
            "model": score(podium, actual),
            "baseline": score(grid, actual),
        })
        log.info("Backtest %s R%s: predicted %s, actual %s", key // 100, key % 100, podium, actual)
    return {"generated_utc": utc_now_iso(), "from_season": from_season,
            "races": races, "summary": summarise(races)}


def summarise(races: list[dict]) -> dict:
    def block(rs, which):
        n = len(rs)
        if not n:
            return None
        return {
            "races": n,
            "winner_pct": round(100 * sum(r[which]["winner"] for r in rs) / n, 1),
            "podium_hits_avg": round(sum(r[which]["podium_hits"] for r in rs) / n, 2),
            "exact_pct": round(100 * sum(r[which]["exact"] for r in rs) / n, 1),
        }

    out = {"overall": {"model": block(races, "model"), "baseline": block(races, "baseline")},
           "by_season": {}}
    for season in sorted({r["season"] for r in races}):
        rs = [r for r in races if r["season"] == season]
        out["by_season"][str(season)] = {"model": block(rs, "model"), "baseline": block(rs, "baseline")}
    return out


def live_summary(recs: list[dict]) -> dict | None:
    scored = [{"model": r["score"]} for r in recs if r.get("score")]
    if not scored:
        return None
    n = len(scored)
    return {
        "races": n,
        "winner_pct": round(100 * sum(r["model"]["winner"] for r in scored) / n, 1),
        "podium_hits_avg": round(sum(r["model"]["podium_hits"] for r in scored) / n, 2),
        "exact_pct": round(100 * sum(r["model"]["exact"] for r in scored) / n, 1),
        "winners_right": sum(r["model"]["winner"] for r in scored),
    }
