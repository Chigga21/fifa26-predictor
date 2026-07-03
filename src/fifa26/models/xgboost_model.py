"""Modelo de regresion de goles con XGBoost, una estrategia GoalModel.
@author Chigga21
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from fifa26.domain.entities import TeamStrength
from fifa26.domain.interfaces import GoalModel
from fifa26.models.features import SIDE_FEATURES, side_features

_FEATURES = SIDE_FEATURES


class XGBoostGoalModel(GoalModel):
    """Regresion Tweedie de goles con gradient boosting.

    El objetivo reg:tweedie penaliza el error de forma que tolera la sobre-dispersion
    de los goles al aprender la media. El ancho real de la cola del marcador lo aporta
    despues la matriz Negative Binomial, no este objetivo.
    """

    name = "XGBoost"

    def __init__(self, **xgb_kwargs) -> None:
        params = dict(
            objective="reg:tweedie",
            tweedie_variance_power=1.4,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        params.update(xgb_kwargs)
        self._model = XGBRegressor(**params)
        self._strengths: dict[str, TeamStrength] = {}

    def fit(self, matches: pd.DataFrame, strengths: dict[str, TeamStrength]) -> "XGBoostGoalModel":
        self._strengths = strengths
        home = self._side_frame(matches, scoring="home")
        away = self._side_frame(matches, scoring="away")
        long_df = pd.concat([home, away], ignore_index=True)
        self._model.fit(long_df[_FEATURES], long_df["goals"])
        return self

    def predict_expected_goals(self, fixtures: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        home_feats = side_features(fixtures, self._strengths, scoring="home")
        away_feats = side_features(fixtures, self._strengths, scoring="away")
        lambda_home = self._model.predict(home_feats[_FEATURES])
        lambda_away = self._model.predict(away_feats[_FEATURES])
        return np.asarray(lambda_home, float), np.asarray(lambda_away, float)

    def _side_frame(self, matches: pd.DataFrame, scoring: str) -> pd.DataFrame:
        df = side_features(matches, self._strengths, scoring=scoring)
        df["goals"] = matches["home_score" if scoring == "home" else "away_score"].to_numpy()
        return df
