"""Servicio de prediccion que convierte un partido elegido en un pronostico completo.
Autor Chigga21
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fifa26.domain import MatchPrediction, ScoreMatrix
from fifa26.models import GoalModel
from fifa26.score_matrix import to_prediction, top_scorelines
from fifa26.training import TrainedArtifacts


@dataclass(frozen=True)
class MatchForecast:
    """El pronostico completo de un partido."""

    score_matrix: ScoreMatrix
    prediction: MatchPrediction
    top_scorelines: list[tuple[str, float]]


class PredictionService:
    """Genera el pronostico completo de un partido con cada modelo."""

    def __init__(self, artifacts: TrainedArtifacts) -> None:
        self._models = list(artifacts.models)
        self._matrix_builder = artifacts.matrix_builder
        self._dispersion = artifacts.dispersion
        self._teams = list(artifacts.teams)

    @property
    def teams(self) -> list[str]:
        """Equipos disponibles para predecir."""
        return self._teams

    def predict_all(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        tournament: str = "FIFA World Cup",
        top_n: int = 10,
    ) -> list[tuple[str, MatchForecast]]:
        """Genera el pronostico de cada modelo para el mismo partido.

        Args:
            home_team (str): Equipo local.
            away_team (str): Equipo visitante.
            neutral (bool): Si la sede es neutral.
            tournament (str): Torneo del partido.
            top_n (int): Marcadores mas probables a incluir.

        Returns:
            list[tuple[str, MatchForecast]]: Nombre y pronostico por modelo.
        """
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
        """Arma el pronostico completo de un modelo para un partido.

        Args:
            model (GoalModel): Modelo de goles a usar.
            home_team (str): Equipo local.
            away_team (str): Equipo visitante.
            neutral (bool): Si la sede es neutral.
            tournament (str): Torneo del partido.
            top_n (int): Marcadores mas probables a incluir.

        Returns:
            MatchForecast: Matriz, probabilidades 1X2 y marcadores top.
        """
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
        prediction = to_prediction(score_matrix)
        top = top_scorelines(score_matrix, top_n=top_n)
        return MatchForecast(
            score_matrix=score_matrix, prediction=prediction, top_scorelines=top
        )
