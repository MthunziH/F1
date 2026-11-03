"""Gradient-boosted classifiers for P(win) and P(podium)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from f1predict.features import FEATURES, MONOTONIC


def _classifier(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        monotonic_cst=[MONOTONIC.get(f, 0) for f in FEATURES],
        random_state=seed,
    )


class PodiumModel:
    def fit(self, train: pd.DataFrame) -> "PodiumModel":
        train = train[train["has_result"]]
        X = train[FEATURES].astype(float)
        self.win = _classifier(1).fit(X, train["win"].astype(int))
        self.podium = _classifier(2).fit(X, train["podium"].astype(int))
        self.n_races = train["race_key"].nunique()
        return self

    def predict_race(self, race: pd.DataFrame) -> pd.DataFrame:
        """Probabilities for one race, normalised so they add up to 1 (win)
        and 3 (podium)."""
        X = race[FEATURES].astype(float)
        out = race.copy()
        out["p_win"] = _normalise(self.win.predict_proba(X)[:, 1], 1.0)
        out["p_podium"] = _normalise(self.podium.predict_proba(X)[:, 1], 3.0)
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
