"""Tests de la validacion cruzada de origen movil
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fifa26.domain.entities import TeamStrength
from fifa26.evaluation.cross_validation import rolling_origin_evaluate
from fifa26.features.dixon_coles import DixonColesEstimator
from fifa26.prediction.outcome import OutcomeCalculator


class _ConstantModel:
    """Modelo trivial que devuelve goles esperados fijos, barato para los tests."""

    def __init__(self, name: str, lam_home: float, lam_away: float) -> None:
        self.name = name
        self._lh = lam_home
        self._la = lam_away

    def fit(self, matches, strengths):
        return self

    def predict_expected_goals(self, fixtures):
        n = len(fixtures)
        return np.full(n, self._lh), np.full(n, self._la)


def _synthetic_matches():
    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(6)]
    rows = []
    for year in (2021, 2022, 2023):
        for _ in range(120):
            h, a = rng.choice(teams, size=2, replace=False)
            rows.append([
                f"{year}-06-01", h, a, int(rng.poisson(1.4)), int(rng.poisson(1.0)),
                "Friendly", False, year,
            ])
    df = pd.DataFrame(rows, columns=[
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "neutral", "year",
    ])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_rolling_origin_evaluate_promedia_por_modelo():
    matches = _synthetic_matches()
    models = [
        _ConstantModel("home-lean", 1.6, 0.9),
        _ConstantModel("away-lean", 0.9, 1.6),
    ]
    results = rolling_origin_evaluate(
        matches,
        DixonColesEstimator(),
        models,
        OutcomeCalculator(),
        test_years=[2022, 2023],
    )
    assert set(results) == {"home-lean", "away-lean"}
    for result in results.values():
        # Dos pliegues con datos, ambos con partidos evaluados.
        assert result.n_matches > 0
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.rps <= 1.0
