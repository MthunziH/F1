"""Feature engineering. Every feature for a race uses only information that is
known after qualifying and before the race starts."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "quali_pos",         # qualifying position (≈ grid; penalties are not known yet)
    "quali_gap_pct",     # % slower than pole lap
    "team_best_quali",   # best qualifying position of the team (car pace)
    "teammate_delta",    # quali position minus teammate's
    "d_finish5",         # driver: avg finishing position, last 5 races
    "d_pts5",            # driver: avg points, last 5 races
    "d_podium10",        # driver: podium rate, last 10 races
    "d_win10",           # driver: win rate, last 10 races
    "d_dnf10",           # driver: retirement rate, last 10 races
    "d_gain10",          # driver: avg places gained from quali to finish, last 10
    "d_season_share",    # driver: season points so far / leader's points
    "t_pts5",            # team: avg points per race, last 5 races
    "t_best5",           # team: avg best finish, last 5 races
    "d_track_finish",    # driver: avg finish at this circuit in earlier years
    "circuit_pole_win",  # circuit: how often pole has converted to a win
]

# -1 = probability can only go down as the feature grows.
MONOTONIC = {"quali_pos": -1, "quali_gap_pct": -1}

DNF_FINISH = 20.0


def _prior_rolling(df, key, col, window):
    return df.groupby(key, sort=False)[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean())


def _prior_expanding(df, key, col):
    return df.groupby(key, sort=False)[col].transform(
        lambda s: s.shift(1).expanding().mean())


def build_features(hist: pd.DataFrame, target: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one row per driver per race with FEATURES and labels.

    `target` rows (an upcoming race) have qualifying data but no result.
    """
    df = hist.copy()
    df["is_target"] = False
    if target is not None and not target.empty:
        t = target.copy()
        t["is_target"] = True
        df = pd.concat([df, t], ignore_index=True)

    for c in ("season", "round"):
        df[c] = df[c].astype(int)
    for c in ("quali_pos", "q_best_s", "grid", "position", "points"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["race_key"] = df["season"] * 100 + df["round"]
    df = df.sort_values(["race_key", "quali_pos"]).reset_index(drop=True)

    # ---- labels / outcomes
    pos_text = df["positionText"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["classified"] = pos_text.str.isdigit()
    df["has_result"] = ~df["is_target"] & df["position"].notna()
    has_result = df["has_result"]
    df["finish"] = np.where(df["classified"], df["position"], DNF_FINISH)
    df.loc[~has_result, "finish"] = np.nan
    df["win"] = np.where(has_result, (df["classified"] & (df["position"] == 1)).astype(float), np.nan)
    df["podium"] = np.where(has_result, (df["classified"] & (df["position"] <= 3)).astype(float), np.nan)
    df["dnf"] = np.where(has_result, (~df["classified"]).astype(float), np.nan)

    # ---- weekend (qualifying) features
    size = df.groupby("race_key")["driverId"].transform("size")
    df["quali_pos"] = df["quali_pos"].fillna(df["grid"].where(df["grid"] > 0)).fillna(size)
    pole = df.groupby("race_key")["q_best_s"].transform("min")
    df["quali_gap_pct"] = ((df["q_best_s"] / pole - 1) * 100).clip(0, 10)
    df["team_best_quali"] = df.groupby(["race_key", "constructorId"])["quali_pos"].transform("min")
    team_sum = df.groupby(["race_key", "constructorId"])["quali_pos"].transform("sum")
    team_n = df.groupby(["race_key", "constructorId"])["quali_pos"].transform("count")
    df["teammate_delta"] = np.where(team_n == 2, df["quali_pos"] - (team_sum - df["quali_pos"]), np.nan)

    # ---- driver form (prior races only)
    df["gain"] = df["quali_pos"] - df["finish"]
    df["d_finish5"] = _prior_rolling(df, "driverId", "finish", 5)
    df["d_pts5"] = _prior_rolling(df, "driverId", "points", 5)
    df["d_podium10"] = _prior_rolling(df, "driverId", "podium", 10)
    df["d_win10"] = _prior_rolling(df, "driverId", "win", 10)
    df["d_dnf10"] = _prior_rolling(df, "driverId", "dnf", 10)
    df["d_gain10"] = _prior_rolling(df, "driverId", "gain", 10)

    pts = df["points"].fillna(0)
    cum_before = pts.groupby([df["driverId"], df["season"]]).cumsum() - pts
    leader = cum_before.groupby(df["race_key"]).transform("max")
    df["d_season_share"] = np.where(leader > 0, cum_before / leader.where(leader > 0, 1), np.nan)

    # ---- team form
    team = (df[has_result].groupby(["race_key", "constructorId"])
            .agg(team_pts=("points", "sum"), team_best=("finish", "min")).reset_index())
    keys = df[["race_key", "constructorId"]].drop_duplicates()
    team = keys.merge(team, on=["race_key", "constructorId"], how="left").sort_values("race_key")
    team["t_pts5"] = _prior_rolling(team, "constructorId", "team_pts", 5)
    team["t_best5"] = _prior_rolling(team, "constructorId", "team_best", 5)
    df = df.merge(team[["race_key", "constructorId", "t_pts5", "t_best5"]],
                  on=["race_key", "constructorId"], how="left")

    # ---- track history
    df = df.sort_values(["race_key", "quali_pos"]).reset_index(drop=True)
    df["d_track_finish"] = _prior_expanding(df, [df["driverId"], df["circuitId"].fillna("?")], "finish")
    race = (df[df["has_result"]]
            .groupby("race_key").apply(_pole_converted, include_groups=False)
            .rename("pole_won").reset_index())
    races = df[["race_key", "circuitId"]].drop_duplicates("race_key").merge(race, on="race_key", how="left")
    races["circuit_pole_win"] = _prior_expanding(races, races["circuitId"].fillna("?"), "pole_won")
    df = df.merge(races[["race_key", "circuit_pole_win"]], on="race_key", how="left")

    return df


def _pole_converted(g: pd.DataFrame) -> float:
    winner = g[g["win"] == 1]
    if winner.empty:
        return np.nan
    return float(winner["quali_pos"].iloc[0] == 1)
