import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PREDICTIONS_DIR = ROOT / "predictions"
SITE_DIR = ROOT / "site"
CACHE_DIR = Path(os.environ.get("FASTF1_CACHE", ROOT / ".fastf1-cache"))

HISTORY_CSV = DATA_DIR / "history.csv"
BACKTEST_JSON = DATA_DIR / "backtest.json"

# Seasons used as training history. Older seasons still teach the model how
# grid position, form and car pace translate into results.
FIRST_SEASON = 2018
# Seasons replayed race-by-race in the backtest.
BACKTEST_FROM_SEASON = 2024

# Wait this long after qualifying is scheduled to start before predicting,
# so the session is over and its data has been published.
QUALI_READY_AFTER_MIN = 90
