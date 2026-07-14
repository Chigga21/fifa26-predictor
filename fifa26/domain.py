"""Entidades y objetos de valor del dominio.
Autor Chigga21
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Outcome(Enum):
    """Los tres resultados 1X2 posibles de un partido."""

    HOME_WIN = "1"
    DRAW = "X"
    AWAY_WIN = "2"

    @staticmethod
    def from_scores(home_score: int, away_score: int) -> "Outcome":
        """Deriva el resultado 1X2 de un marcador.

        Args:
            home_score (int): Goles del local.
            away_score (int): Goles del visitante.

        Returns:
            Outcome: Resultado del partido.
        """
        if home_score > away_score:
            return Outcome.HOME_WIN
        if home_score < away_score:
            return Outcome.AWAY_WIN
        return Outcome.DRAW


@dataclass(frozen=True)
class TeamStrength:
    """Ratings ofensivo y defensivo Dixon-Coles de un equipo."""

    team: str
    attack: float
    defense: float


@dataclass(frozen=True)
class ScoreMatrix:
    """Probabilidad conjunta de cada marcador exacto de un partido."""

    home_team: str
    away_team: str
    lambda_home: float
    lambda_away: float
    matrix: np.ndarray


@dataclass(frozen=True)
class ShootoutPrediction:
    """Probabilidad de cada equipo de ganar la tanda de penales."""

    home_team: str
    away_team: str
    prob_home: float
    prob_away: float


@dataclass(frozen=True)
class MatchPrediction:
    """Probabilidades 1X2 de un partido y su resultado mas probable."""

    home_team: str
    away_team: str
    prob_home_win: float
    prob_draw: float
    prob_away_win: float

    @property
    def predicted_outcome(self) -> Outcome:
        """Resultado con la probabilidad mas alta."""
        probs = {
            Outcome.HOME_WIN: self.prob_home_win,
            Outcome.DRAW: self.prob_draw,
            Outcome.AWAY_WIN: self.prob_away_win,
        }
        return max(probs, key=probs.get)
