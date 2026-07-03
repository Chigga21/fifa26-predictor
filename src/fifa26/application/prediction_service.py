"""Servicio de prediccion que convierte un partido
elegido en un pronóstico completo.

Envuelve los artefactos del Trainer tras una única llamada
al método predict.

@author Chigga21
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fifa26.application.training import TrainedArtifacts
from fifa26.domain.entities import MatchPrediction, ScoreMatrix
from fifa26.domain.interfaces import DispersionModel, GoalModel
from fifa26.prediction.negative_binomial_matrix import NegativeBinomialMatrixBuilder
from fifa26.prediction.outcome import OutcomeCalculator


@dataclass(frozen=True)
class MatchForecast:
    """El pronostico completo de un partido."""

    score_matrix: ScoreMatrix
    prediction: MatchPrediction
    top_scorelines: list[tuple[str, float]]


class PredictionService:
    def __init__(
        self,
        models: list[GoalModel],
        matrix_builder: NegativeBinomialMatrixBuilder,
        dispersion: DispersionModel,
        outcome_calculator: OutcomeCalculator,
        teams: list[str],
    ) -> None:
        self._models = list(models)
        self._matrix_builder = matrix_builder
        self._dispersion = dispersion
        self._outcome = outcome_calculator
        self._teams = list(teams)

    @classmethod
    def from_artifacts(
        cls, artifacts: TrainedArtifacts, outcome_calculator: OutcomeCalculator
    ) -> "PredictionService":
        return cls(
            models=artifacts.models,
            matrix_builder=artifacts.matrix_builder,
            dispersion=artifacts.dispersion,
            outcome_calculator=outcome_calculator,
            teams=artifacts.teams,
        )

    @property
    def teams(self) -> list[str]:
        return self._teams

    @property
    def model_names(self) -> list[str]:
        return [m.name for m in self._models]

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        tournament: str = "FIFA World Cup",
        top_n: int = 10,
    ) -> MatchForecast:
        """Pronostico del primer modelo, conservado por compatibilidad."""
        return self._forecast(self._models[0], home_team, away_team, neutral, tournament, top_n)

    def predict_all(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        tournament: str = "FIFA World Cup",
        top_n: int = 10,
    ) -> list[tuple[str, MatchForecast]]:
        """Pronostico de cada modelo para mostrarlos todos a la vez."""
        return [
            (model.name, self._forecast(model, home_team, away_team, neutral, tournament, top_n))
            for model in self._models
        ]

    def _forecast(
        self,
        model: GoalModel,
        home_team: str,
        away_team: str,
        neutral: bool,
        tournament: str,
        top_n: int,
    ) -> MatchForecast:
        fixture = pd.DataFrame(
            [
                {
                    "home_team": home_team,
                    "away_team": away_team,
                    "neutral": neutral,
                    "tournament": tournament,
                }
            ]
        )
        lambda_home, lambda_away = model.predict_expected_goals(fixture)
        d_home, d_away = self._dispersion.predict_dispersion(fixture)
        score_matrix = self._matrix_builder.build(
            home_team,
            away_team,
            float(lambda_home[0]),
            float(lambda_away[0]),
            float(d_home[0]),
            float(d_away[0]),
        )
        prediction = self._outcome.to_prediction(score_matrix)
        top = self._outcome.top_scorelines(score_matrix, top_n=top_n)
        return MatchForecast(
            score_matrix=score_matrix, prediction=prediction, top_scorelines=top
        )
