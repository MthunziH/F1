"""Data access: FastF1 (live timing + its Ergast/Jolpica interface).

History (training data) comes from the Ergast interface because it returns a
whole season of results in a handful of requests. The current weekend's
qualifying comes from FastF1 live timing first (available minutes after the
session), with Ergast as a fallback.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import timedelta

import numpy as np
import pandas as pd

from f1predict import config

log = logging.getLogger(__name__)

HISTORY_COLUMNS = [
    "season", "round", "date", "eventName", "circuitId",
    "driverId", "driverCode", "driverName", "constructorId", "constructorName",
    "quali_pos", "q_best_s", "grid", "position", "positionText", "points",
    "status", "source",
]

# FastF1 / live-timing team names -> Ergast constructorId.
TEAM_ALIASES = {
    "red bull racing": "red_bull", "red bull": "red_bull",
    "racing bulls": "rb", "rb": "rb", "visa cash app rb": "rb",
    "alphatauri": "alphatauri", "toro rosso": "toro_rosso",
    "kick sauber": "sauber", "sauber": "sauber", "alfa romeo": "alfa",
    "audi": "audi", "cadillac": "cadillac",
    "haas f1 team": "haas", "haas": "haas",
    "alpine": "alpine", "renault": "renault",
    "aston martin": "aston_martin", "racing point": "racing_point",
    "force india": "force_india",
    "williams": "williams", "mclaren": "mclaren",
    "ferrari": "ferrari", "mercedes": "mercedes",
}


def _fastf1():
    import fastf1

    logging.getLogger("fastf1").setLevel(logging.WARNING)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(config.CACHE_DIR))
    return fastf1


def team_to_constructor_id(team_name: str) -> str:
    key = re.sub(r"\s+", " ", str(team_name).lower()).strip()
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    for alias, cid in TEAM_ALIASES.items():
        if alias in key:
            return cid
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_")


def lap_time_seconds(value) -> float:
    """Convert Timedelta / '1:23.456' / NaN to seconds (NaN if missing)."""
    if value is None:
        return np.nan
    if isinstance(value, (pd.Timedelta, timedelta)):
        return value.total_seconds() if not pd.isna(value) else np.nan
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return np.nan
    try:
        parts = text.split(":")
        secs = float(parts[-1])
        if len(parts) > 1:
            secs += 60 * float(parts[-2])
        return secs
    except ValueError:
        return np.nan


def best_quali_time(row) -> float:
    times = [lap_time_seconds(row.get(q)) for q in ("Q1", "Q2", "Q3")]
    times = [t for t in times if not np.isnan(t)]
    return min(times) if times else np.nan


# ---------------------------------------------------------------- schedule

def event_schedule(season: int) -> pd.DataFrame:
    ff1 = _fastf1()
    sched = ff1.get_event_schedule(season, include_testing=False)
    rows = []
    for _, ev in sched.iterrows():
        sessions = {ev[f"Session{i}"]: ev[f"Session{i}DateUtc"] for i in range(1, 6)}
        rows.append({
            "season": season,
            "round": int(ev["RoundNumber"]),
            "eventName": ev["EventName"],
            "location": ev["Location"],
            "country": ev["Country"],
            "format": ev["EventFormat"],
            "quali_utc": pd.Timestamp(sessions.get("Qualifying")) if sessions.get("Qualifying") is not None else pd.NaT,
            "race_utc": pd.Timestamp(sessions.get("Race")) if sessions.get("Race") is not None else pd.NaT,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Ergast

def _ergast():
    _fastf1()
    from fastf1.ergast import Ergast

    return Ergast(result_type="pandas", auto_cast=True, limit=100)


def _with_retry(fn, *args, **kwargs):
    """Jolpica (Ergast) allows ~4 requests/s and 500/hour; back off on 429."""
    delay = 5
    for attempt in range(6):
        try:
            time.sleep(0.4)
            return fn(*args, **kwargs)
        except Exception as exc:
            if "Too Many Requests" not in str(exc) or attempt == 5:
                raise
            log.info("Ergast rate limit hit, waiting %ss", delay)
            time.sleep(delay)
            delay *= 2


def _collect_pages(resp) -> pd.DataFrame:
    frames = []
    while True:
        for i, desc in resp.description.iterrows():
            part = resp.content[i].copy()
            part["season"] = int(desc["season"])
            part["round"] = int(desc["round"])
            part["eventName"] = desc.get("raceName")
            part["circuitId"] = desc.get("circuitId")
            part["date"] = pd.Timestamp(desc.get("raceDate")).date().isoformat() if pd.notna(desc.get("raceDate")) else None
            frames.append(part)
        h = resp._response_headers
        if int(h.get("offset", 0)) + int(h.get("limit", 0)) >= int(h.get("total", 0)):
            break
        resp = _with_retry(resp.get_next_result_page)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def ergast_season(season: int) -> pd.DataFrame:
    """All race results + qualifying for one season, in HISTORY_COLUMNS form."""
    erg = _ergast()
    races = _collect_pages(_with_retry(erg.get_race_results, season=season))
    if races.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    races = races.drop_duplicates(["season", "round", "driverId"])
    quali = _collect_pages(_with_retry(erg.get_qualifying_results, season=season))
    if not quali.empty:
        quali = quali.drop_duplicates(["season", "round", "driverId"])
        quali["q_best_s"] = quali.apply(best_quali_time, axis=1)
        quali = quali.rename(columns={"position": "quali_pos"})[
            ["season", "round", "driverId", "quali_pos", "q_best_s"]]
        races = races.merge(quali, on=["season", "round", "driverId"], how="left")
    else:
        races["quali_pos"] = np.nan
        races["q_best_s"] = np.nan
    races["driverName"] = races["givenName"].astype(str) + " " + races["familyName"].astype(str)
    races["source"] = "ergast"
    return races.reindex(columns=HISTORY_COLUMNS)


def ergast_qualifying(season: int, rnd: int) -> pd.DataFrame | None:
    erg = _ergast()
    q = _collect_pages(_with_retry(erg.get_qualifying_results, season=season, round=rnd))
    if q.empty:
        return None
    q["q_best_s"] = q.apply(best_quali_time, axis=1)
    return pd.DataFrame({
        "driverCode": q["driverCode"],
        "driverId": q["driverId"],
        "driverName": q["givenName"].astype(str) + " " + q["familyName"].astype(str),
        "constructorId": q["constructorId"],
        "constructorName": q["constructorName"],
        "quali_pos": q["position"].astype(float),
        "q_best_s": q["q_best_s"],
    })


# ---------------------------------------------------------------- FastF1 live timing

def _load_session(season: int, rnd: int, kind: str):
    ff1 = _fastf1()
    session = ff1.get_session(season, rnd, kind)
    # laps=True: for very recent sessions positions are computed from lap data.
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    res = session.results
    if res is None or res.empty or res["Position"].isna().all():
        return None, session
    return res, session


def fastf1_qualifying(season: int, rnd: int) -> pd.DataFrame | None:
    try:
        res, _ = _load_session(season, rnd, "Q")
    except Exception as exc:  # network / data not yet published
        log.warning("FastF1 qualifying load failed: %s", exc)
        return None
    if res is None:
        return None
    res = res.copy()
    res["q_best_s"] = res.apply(best_quali_time, axis=1)
    return pd.DataFrame({
        "driverCode": res["Abbreviation"],
        "driverId": res.get("DriverId", pd.Series(index=res.index, dtype=str)).replace("", np.nan),
        "driverName": res["FullName"],
        "constructorId": res["TeamName"].map(team_to_constructor_id),
        "constructorName": res["TeamName"],
        "quali_pos": res["Position"].astype(float),
        "q_best_s": res["q_best_s"],
    }).reset_index(drop=True)


def fastf1_race_round(season: int, rnd: int, event: dict) -> pd.DataFrame | None:
    """A completed race (+ its qualifying) from live timing, in HISTORY_COLUMNS form.

    Used when Ergast has not published a just-finished race yet.
    """
    try:
        res, _ = _load_session(season, rnd, "R")
    except Exception as exc:
        log.warning("FastF1 race load failed: %s", exc)
        return None
    if res is None:
        return None
    quali = fastf1_qualifying(season, rnd)
    out = pd.DataFrame({
        "season": season, "round": rnd,
        "date": pd.Timestamp(event["race_utc"]).date().isoformat(),
        "eventName": event["eventName"],
        "circuitId": event.get("circuitId"),
        "driverId": res.get("DriverId", pd.Series(index=res.index, dtype=str)).replace("", np.nan),
        "driverCode": res["Abbreviation"],
        "driverName": res["FullName"],
        "constructorId": res["TeamName"].map(team_to_constructor_id),
        "constructorName": res["TeamName"],
        "grid": res["GridPosition"],
        "position": res["Position"],
        "positionText": res["ClassifiedPosition"],
        "points": res["Points"],
        "status": res["Status"],
        "source": "fastf1",
    }).reset_index(drop=True)
    if quali is not None:
        out = out.merge(quali[["driverCode", "quali_pos", "q_best_s"]], on="driverCode", how="left")
    return out.reindex(columns=HISTORY_COLUMNS)


def get_qualifying(season: int, rnd: int) -> tuple[pd.DataFrame | None, str]:
    q = fastf1_qualifying(season, rnd)
    if q is not None and len(q) >= 10:
        return q, "fastf1"
    try:
        q = ergast_qualifying(season, rnd)
    except Exception as exc:
        log.warning("Ergast qualifying load failed: %s", exc)
        q = None
    if q is not None and len(q) >= 10:
        return q, "ergast"
    return None, ""
