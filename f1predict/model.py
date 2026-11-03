"""P(win) and P(podium): gradient-boosted classifiers blended with a
grid-position prior.

The prior is simply "how often does the driver starting Pn win / reach the
podium", learned from history. Blending 50/50 keeps the model from
over-ruling the pole sitter on recent form alone, which the backtest showed
it did too eagerly, while keeping its better-calibrated probabilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from f1predict.features import FEATURES, MONOTONIC


def _classifier(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=200,
        max_leaf_nodes=8,
        min_samples_leaf=60,
        l2_regularization=3.0,
        monotonic_cst=[MONOTONIC.get(f, 0) for f in FEATURES],
        random_state=seed,
    )


MODEL_WEIGHT = 0.5  # share of the ML model vs the grid-position prior
PRIOR_MAX_POS = 11  # positions 11+ are pooled for the prior


def _grid_rates(train: pd.DataFrame, col: str) -> pd.Series:
    pos = train["quali_pos"].clip(upper=PRIOR_MAX_POS)
    agg = train.groupby(pos)[col].agg(["sum", "count"])
    return (agg["sum"] + 0.5) / (agg["count"] + 1)  # smoothed


class PodiumModel:
    def fit(self, train: pd.DataFrame) -> "PodiumModel":
        train = train[train["has_result"]]
        X = train[FEATURES].astype(float)
        self.win = _classifier(1).fit(X, train["win"].astype(int))
        self.podium = _classifier(2).fit(X, train["podium"].astype(int))
        self.win_prior = _grid_rates(train, "win")
        self.podium_prior = _grid_rates(train, "podium")
        self.n_races = train["race_key"].nunique()
        return self

    def _prior(self, race: pd.DataFrame, rates: pd.Series) -> np.ndarray:
        pos = race["quali_pos"].clip(upper=PRIOR_MAX_POS)
        return pos.map(rates).fillna(rates.min()).to_numpy(dtype=float)

    def predict_race(self, race: pd.DataFrame) -> pd.DataFrame:
        """Probabilities for one race, normalised so they add up to 1 (win)
        and 3 (podium)."""
        X = race[FEATURES].astype(float)
        out = race.copy()
        w = MODEL_WEIGHT
        out["p_win"] = (w * _normalise(self.win.predict_proba(X)[:, 1], 1.0)
                        + (1 - w) * _normalise(self._prior(race, self.win_prior), 1.0))
        out["p_podium"] = (w * _normalise(self.podium.predict_proba(X)[:, 1], 3.0)
                           + (1 - w) * _normalise(self._prior(race, self.podium_prior), 3.0))
        # A driver can't be likelier to win than to finish on the podium.
        out["p_podium"] = np.maximum(out["p_podium"], out["p_win"])
        return out.sort_values(["p_win", "p_podium"], ascending=False).reset_index(drop=True)


def _normalise(p: np.ndarray, total: float) -> np.ndarray:
    """Scale probabilities to sum to `total`, capping each at 1."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, None)
    total = min(total, len(p))
    capped = np.zeros(len(p), dtype=bool)
    for _ in range(len(p)):
        free = ~capped
        remaining = total - capped.sum()
        scaled = p.copy()
        scaled[free] = p[free] / p[free].sum() * remaining
        scaled[capped] = 1.0
        over = free & (scaled > 1.0)
        if not over.any():
            return scaled
        capped |= over
    return np.minimum(scaled, 1.0)


def predicted_podium(pred: pd.DataFrame) -> list[int]:
    """Row indices for P1, P2, P3: most likely winner first, then the two
    most likely podium finishers among the rest."""
    p1 = int(pred["p_win"].idxmax())
    rest = pred.drop(index=p1)
    rest = rest.assign(_score=rest["p_podium"] + rest["p_win"])
    others = rest.sort_values("_score", ascending=False).index[:2].tolist()
    return [p1, *map(int, others)]
