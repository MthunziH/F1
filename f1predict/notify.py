"""Phone notifications via GitHub issues.

Opening an issue that @mentions the repo owner triggers a GitHub notification
(push in the GitHub mobile app and/or email, per the owner's settings).
Only runs inside GitHub Actions, where GITHUB_TOKEN is available.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request

log = logging.getLogger(__name__)


def _enabled() -> bool:
    return bool(os.environ.get("GITHUB_TOKEN") and os.environ.get("GITHUB_REPOSITORY"))


def _api(method: str, path: str, body: dict | None = None) -> dict:
    url = f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}{path}"
    req = urllib.request.Request(
        url, method=method, data=json.dumps(body).encode() if body else None,
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "f1-podium-predictor",
        })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read() or b"{}")


def site_url() -> str:
    if os.environ.get("SITE_URL"):
        return os.environ["SITE_URL"]
    owner, _, repo = os.environ.get("GITHUB_REPOSITORY", "/").partition("/")
    return f"https://{owner.lower()}.github.io/{repo}/"


def _owner_mention() -> str:
    owner = os.environ.get("GITHUB_REPOSITORY_OWNER")
    return f"@{owner} " if owner else ""


def _driver_line(rec: dict, code: str) -> str:
    d = next(x for x in rec["drivers"] if x["code"] == code)
    return f"{d['name']} ({d['team']}) · win {d['p_win']:.0%} · podium {d['p_podium']:.0%}"


def announce_prediction(rec: dict) -> int | None:
    if not _enabled():
        return None
    p = rec["podium"]
    title = f"🏁 {rec['season']} {rec['eventName']}: {p[0]} · {p[1]} · {p[2]}"
    body = "\n".join([
        f"{_owner_mention()}new prediction is ready.",
        "",
        f"🥇 **P1** {_driver_line(rec, p[0])}",
        f"🥈 **P2** {_driver_line(rec, p[1])}",
        f"🥉 **P3** {_driver_line(rec, p[2])}",
        "",
        f"Full table: {site_url()}",
        "",
        "_This issue is closed automatically with the real result after the race._",
    ])
    try:
        issue = _api("POST", "/issues", {"title": title, "body": body, "labels": ["prediction"]})
        return issue.get("number")
    except Exception as exc:
        log.warning("Could not open notification issue: %s", exc)
        return None


def announce_result(rec: dict) -> None:
    number = rec.get("issue_number")
    if not _enabled() or not number:
        return
    s, a = rec["score"], rec["actual"]["podium"]
    verdict = ("🎯 Exact podium!" if s["exact"]
               else f"Winner {'✅ right' if s['winner'] else '❌ wrong'} · {s['podium_hits']}/3 podium drivers right")
    body = "\n".join([
        f"{_owner_mention()}race result is in.",
        "",
        f"Actual podium: **{a[0]} · {a[1]} · {a[2]}**",
        f"Predicted: {' · '.join(rec['podium'])}",
        "",
        verdict,
        "",
        f"Season record: {site_url()}",
    ])
    try:
        _api("POST", f"/issues/{number}/comments", {"body": body})
        _api("PATCH", f"/issues/{number}", {"state": "closed", "state_reason": "completed"})
    except Exception as exc:
        log.warning("Could not post result to issue #%s: %s", number, exc)
