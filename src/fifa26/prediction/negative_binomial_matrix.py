"""Matriz conjunta de marcadores con sobre-dispersion Binomial negativa
"""
from __future__ import annotations

import numpy as np
from scipy.stats import nbinom, poisson

from fifa26.domain.entities import ScoreMatrix


class NegativeBinomialMatrixBuilder:
    """Convierte goles esperados y dispersion por lado en una matriz de marcadores."""

    def __init__(self, max_goals: int = 10, rho: float = 0.0) -> None:
        self._max_goals = max_goals
        self._rho = rho

    def build(
        self,
        home_team: str,
        away_team: str,
        lambda_home: float,
        lambda_away: float,
        d_home: float = 1.0,
        d_away: float = 1.0,
    ) -> ScoreMatrix:
        goals = np.arange(self._max_goals + 1)
        home_pmf = self._side_pmf(goals, lambda_home, d_home)
        away_pmf = self._side_pmf(goals, lambda_away, d_away)
        matrix = np.outer(home_pmf, away_pmf)

        matrix = self._apply_tau(matrix, lambda_home, lambda_away)
        matrix = matrix / matrix.sum()
        return ScoreMatrix(
            home_team=home_team,
            away_team=away_team,
            lambda_home=float(lambda_home),
            lambda_away=float(lambda_away),
            matrix=matrix,
        )

    @staticmethod
    def _side_pmf(goals: np.ndarray, lam: float, dispersion: float) -> np.ndarray:
        """PMF de goles de un lado, Poisson sin dispersion o Negative Binomial con ella"""
        lam = max(float(lam), 1e-9)
        if dispersion <= 1.0 + 1e-9:
            return poisson.pmf(goals, lam)
        r = lam / (dispersion - 1.0)
        p = r / (r + lam)
        return nbinom.pmf(goals, r, p)

    def _apply_tau(self, matrix: np.ndarray, lh: float, la: float) -> np.ndarray:
        rho = self._rho
        if rho == 0.0:
            return matrix
        m = matrix.copy()
        m[0, 0] *= 1 - lh * la * rho
        m[0, 1] *= 1 + lh * rho
        m[1, 0] *= 1 + la * rho
        m[1, 1] *= 1 - rho
        return np.clip(m, 0.0, None)
