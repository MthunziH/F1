"""Command line entry point.

    python -m f1predict auto                       # what the scheduled job runs
    python -m f1predict predict --season 2026 --round 20 [--force]
    python -m f1predict backtest
    python -m f1predict site
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from f1predict import config, history, notify, predictor, sources
from f1predict.site import build_site

log = logging.getLogger("f1predict")


def now_utc() -> pd.Timestamp:
    """Naive UTC timestamp (FastF1 schedule times are naive UTC)."""
    return pd.Timestamp.now(tz="UTC").tz_convert(None)


def _iso(ts) -> str | None:
    return None if pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def _event_dict(ev) -> dict:
    return {"season": int(ev["season"]), "round": int(ev["round"]), "eventName": ev["eventName"],
            "quali_utc": _iso(ev["quali_utc"]), "race_utc": _iso(ev["race_utc"])}


def weekend_to_predict(schedule: pd.DataFrame, now: pd.Timestamp):
    """The event whose qualifying is over but whose race hasn't started."""
    ready = now - pd.Timedelta(minutes=config.QUALI_READY_AFTER_MIN)
    ev = schedule[(schedule["quali_utc"] <= ready) & (schedule["race_utc"] > now)]
    return None if ev.empty else ev.iloc[0]


def next_event(schedule: pd.DataFrame, now: pd.Timestamp):
    ev = schedule[schedule["race_utc"] > now]
    return None if ev.empty else _event_dict(ev.iloc[0])


def run_prediction(hist, season: int, rnd: int, event_name: str, race_utc: str | None,
                   force: bool = False) -> dict | None:
    path = predictor.prediction_path(season, rnd)
    if path.exists() and not force:
        log.info("Prediction for %s round %s already exists", season, rnd)
        return None
    quali, source = sources.get_qualifying(season, rnd)
    if quali is None:
        log.warning("Qualifying results for %s round %s are not available yet", season, rnd)
        return None
    cids = history.circuit_ids(season)
    circuit = cids.get(rnd)
    if circuit is None:
        past = hist[hist["eventName"] == event_name]
        circuit = past["circuitId"].iloc[-1] if not past.empty else None
    target = predictor.prepare_target(hist, quali, season, rnd, event_name, circuit)
    pred, n_races = predictor.predict_race(hist, target)
    rec = predictor.prediction_record(pred, {
        "season": season, "round": rnd, "eventName": event_name, "circuitId": circuit,
        "race_utc": race_utc, "quali_source": source, "trained_on_races": n_races,
    })
    if path.exists():  # keep the notification issue when re-running
        rec["issue_number"] = json.loads(path.read_text()).get("issue_number")
    predictor.save_prediction(rec)
    _print(rec)
    return rec


def _print(rec: dict) -> None:
    print(f"\n{rec['season']} {rec['eventName']} — predicted podium")
    for i, code in enumerate(rec["podium"], 1):
        d = next(x for x in rec["drivers"] if x["code"] == code)
        print(f"  P{i}  {d['name']:<22} {d['team']:<18} win {d['p_win']:>5.1%}  podium {d['p_podium']:>5.1%}")
    print("\n  All drivers:")
    for d in rec["drivers"]:
        print(f"    {str(d['quali_pos']):>2}  {d['code']}  win {d['p_win']:>5.1%}  podium {d['p_podium']:>5.1%}")


def cmd_auto(args) -> None:
    now = now_utc()
    hist, changed = history.update_history(now)
    log.info("History: %s races, %s rows", hist.groupby(["season", "round"]).ngroups, len(hist))

    for rec in predictor.score_pending(hist):
        log.info("Scored %s: %s", rec["eventName"], rec["score"])
        notify.announce_result(rec)

    if changed or not config.BACKTEST_JSON.exists():
        bt = predictor.backtest(hist)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.BACKTEST_JSON.write_text(json.dumps(bt, indent=1) + "\n")
        log.info("Backtest: %s", json.dumps(bt["summary"]["overall"]))

    schedule = sources.event_schedule(now.year)
    ev = weekend_to_predict(schedule, now)
    if ev is not None:
        rec = run_prediction(hist, int(ev["season"]), int(ev["round"]), ev["eventName"], _iso(ev["race_utc"]))
        if rec is not None and not rec.get("issue_number"):
            rec["issue_number"] = notify.announce_prediction(rec)
            predictor.save_prediction(rec)
    else:
        log.info("No race weekend waiting for a prediction right now")

    build_site(next_event(schedule, now))


def cmd_predict(args) -> None:
    now = now_utc()
    hist, _ = history.update_history(now)
    schedule = sources.event_schedule(args.season)
    ev = schedule[schedule["round"] == args.round]
    if ev.empty:
        sys.exit(f"No round {args.round} in {args.season}")
    ev = ev.iloc[0]
    rec = run_prediction(hist, args.season, args.round, ev["eventName"], _iso(ev["race_utc"]), force=args.force)
    if rec is None and not args.force:
        print("Nothing new (use --force to redo an existing prediction).")
    predictor.score_pending(hist)
    build_site(next_event(sources.event_schedule(now.year), now))


def cmd_backtest(args) -> None:
    hist, _ = history.update_history(now_utc())
    bt = predictor.backtest(hist, args.from_season)
    config.BACKTEST_JSON.write_text(json.dumps(bt, indent=1) + "\n")
    print(json.dumps(bt["summary"], indent=2))


def cmd_site(args) -> None:
    now = now_utc()
    try:
        nxt = next_event(sources.event_schedule(now.year), now)
    except Exception:
        nxt = None
    build_site(nxt)
    print(f"Wrote {config.SITE_DIR / 'index.html'}")


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="f1predict", description="Predict the F1 podium after qualifying.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("auto", help="update data, score finished races, predict if qualifying just ended, build site")
    pp = sub.add_parser("predict", help="predict a specific race (its qualifying must be done)")
    pp.add_argument("--season", type=int, required=True)
    pp.add_argument("--round", type=int, required=True)
    pp.add_argument("--force", action="store_true", help="overwrite an existing prediction")
    bp = sub.add_parser("backtest", help="replay past races to measure accuracy")
    bp.add_argument("--from-season", type=int, default=config.BACKTEST_FROM_SEASON)
    sub.add_parser("site", help="rebuild site/index.html")
    args = p.parse_args(argv)
    {"auto": cmd_auto, "predict": cmd_predict, "backtest": cmd_backtest, "site": cmd_site}[args.cmd](args)


if __name__ == "__main__":
    main()
