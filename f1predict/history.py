"""Keeps data/history.csv (one row per driver per race) up to date."""
from __future__ import annotations

import logging

import pandas as pd

from f1predict import config, sources

log = logging.getLogger(__name__)


def load_history() -> pd.DataFrame:
    if not config.HISTORY_CSV.exists():
        return pd.DataFrame(columns=sources.HISTORY_COLUMNS)
    return pd.read_csv(config.HISTORY_CSV)


def save_history(df: pd.DataFrame) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["season", "round", "position"]).reset_index(drop=True)
    df.to_csv(config.HISTORY_CSV, index=False)


def circuit_ids(season: int) -> dict[int, str]:
    try:
        erg = sources._ergast()
        sched = sources._with_retry(erg.get_race_schedule, season=season)
        return {int(r["round"]): r["circuitId"] for _, r in sched.iterrows()}
    except Exception as exc:
        log.warning("Could not load circuit ids for %s: %s", season, exc)
        return {}


def completed_rounds(schedule: pd.DataFrame, now: pd.Timestamp) -> list[int]:
    """Rounds whose race finished (start + 3h) before `now` (naive UTC)."""
    done = schedule[schedule["race_utc"].notna()
                    & (schedule["race_utc"] + pd.Timedelta(hours=3) < now)]
    return sorted(done["round"].astype(int).tolist())


def update_history(now: pd.Timestamp) -> tuple[pd.DataFrame, bool]:
    """Fetch any missing races. Returns (history, changed)."""
    hist = load_history()
    changed = False
    current = now.year

    have_seasons = set(hist["season"].astype(int)) if not hist.empty else set()

    for season in range(config.FIRST_SEASON, current + 1):
        recent = season >= current - 1  # may still be gaining races
        if season in have_seasons and not recent:
            continue
        if season not in have_seasons:
            log.info("Fetching season %s from Ergast", season)
            try:
                df = sources.ergast_season(season)
            except Exception as exc:
                log.warning("Season %s fetch failed (will retry next run): %s", season, exc)
                continue
            if not df.empty:
                hist = pd.concat([hist, df], ignore_index=True)
                changed = True
        if recent:
            hist, c = _refresh_recent_season(hist, season, now)
            changed |= c
    if changed:
        hist = fill_driver_ids(hist)
        save_history(hist)
    return hist, changed


def _refresh_recent_season(hist, season, now):
    """Fill finished races of `season` that are missing (or only have
    provisional live-timing data)."""
    changed = False
    schedule = sources.event_schedule(season)
    done = completed_rounds(schedule, now)
    cur = hist[(hist["season"] == season) & (hist["source"] == "ergast")]
    missing = [r for r in done if r not in set(cur["round"].astype(int))]
    if not missing:
        return hist, False

    log.info("Season %s: missing rounds %s, refreshing from Ergast", season, missing)
    try:
        df = sources.ergast_season(season)
    except Exception as exc:
        log.warning("Ergast season fetch failed: %s", exc)
        df = pd.DataFrame()
    if not df.empty:
        new_rounds = set(df["round"].astype(int)) - set(cur["round"].astype(int))
        if new_rounds:
            hist = hist[~((hist["season"] == season) & hist["round"].isin(df["round"].unique()))]
            hist = pd.concat([hist, df], ignore_index=True)
            changed = True

    have_now = set(hist.loc[hist["season"] == season, "round"].astype(int))
    still_missing = [r for r in done if r not in have_now]
    if still_missing:
        cids = circuit_ids(season)
        for rnd in still_missing:
            ev = schedule[schedule["round"] == rnd].iloc[0].to_dict()
            ev["circuitId"] = cids.get(rnd)
            log.info("Round %s not on Ergast yet, trying FastF1 live timing", rnd)
            df = sources.fastf1_race_round(season, rnd, ev)
            if df is not None and not df.empty:
                hist = pd.concat([hist, df], ignore_index=True)
                changed = True
    return hist, changed


def fill_driver_ids(hist: pd.DataFrame) -> pd.DataFrame:
    """Live-timing rows lack Ergast driver ids; map them via driver code."""
    hist = hist.copy()
    missing = hist["driverId"].isna()
    if missing.any():
        known = (hist[~missing].sort_values(["season", "round"])
                 .drop_duplicates("driverCode", keep="last")
                 .set_index("driverCode")["driverId"])
        hist.loc[missing, "driverId"] = hist.loc[missing, "driverCode"].map(known)
        still = hist["driverId"].isna()
        hist.loc[still, "driverId"] = hist.loc[still, "driverCode"].str.lower()
    return hist
