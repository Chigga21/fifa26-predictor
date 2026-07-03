
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from fifa26.domain.entities import TeamStrength
from fifa26.domain.interfaces import GoalModel
from fifa26.features.dixon_coles import DixonColesEstimator


class CalibratedGoalModel(GoalModel):
    """Decorador que calibra la media de goles de un GoalModel base.
    """

    def __init__(
        self,
        make_base: Callable[[], GoalModel],
        dixon_coles_factory: Callable[[], DixonColesEstimator],
        k_folds: int = 4,
        clip: tuple[float, float] = (0.8, 1.25),
        min_train_years: int = 2,
    ) -> None:
        self._make_base = make_base
        self._dixon_coles_factory = dixon_coles_factory
        self._k_folds = k_folds
        self._clip = clip
        self._min_train_years = min_train_years
        self._base = make_base()
        self.factor: float = 1.0
        # Conserva el nombre del base para no romper la seleccion por nombre ni la UI.
        self.name = self._base.name

    def fit(
        self,
        matches: pd.DataFrame,
        strengths: dict[str, TeamStrength],
    ) -> "CalibratedGoalModel":
        self.factor = self._rolling_factor(matches)
        self._base.fit(matches, strengths)
        return self

    def predict_expected_goals(
        self, fixtures: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        lambda_home, lambda_away = self._base.predict_expected_goals(fixtures)
        return lambda_home * self.factor, lambda_away * self.factor

    def _rolling_factor(self, matches: pd.DataFrame) -> float:
        """Factor real sobre los esperados acumulados
        """
        years = sorted(matches["year"].unique())
        min_year = years[0]
        folds = [
            year
            for year in years
            if year - min_year >= self._min_train_years
        ][-self._k_folds:]
        total_expected = 0.0
        total_actual = 0.0
        for year in folds:
            train = matches[matches["year"] < year]
            holdout = matches[matches["year"] == year]
            if train.empty or holdout.empty:
                continue
            dixon_coles = self._dixon_coles_factory()
            dixon_coles.fit(train)
            model = self._make_base()
            model.fit(train, dixon_coles.strengths)
            lambda_home, lambda_away = model.predict_expected_goals(holdout)
            total_expected += float(lambda_home.sum() + lambda_away.sum())
            total_actual += float(
                holdout["home_score"].sum() + holdout["away_score"].sum()
            )
        if total_expected <= 0.0:
            return 1.0
        return float(np.clip(total_actual / total_expected, *self._clip))
