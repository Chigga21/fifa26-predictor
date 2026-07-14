"""Matriz conjunta de marcadores Negative Binomial y su lectura 1X2.
Autor Chigga21
"""
from __future__ import annotations

import numpy as np
from scipy.stats import nbinom, poisson

from fifa26.domain import MatchPrediction, ScoreMatrix


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
        """Construye la matriz de marcadores de un partido.

        Args:
            home_team (str): Equipo local.
            away_team (str): Equipo visitante.
            lambda_home (float): Goles esperados del local.
            lambda_away (float): Goles esperados del visitante.
            d_home (float): Factor de dispersion del local.
            d_away (float): Factor de dispersion del visitante.

        Returns:
            ScoreMatrix: Probabilidad de cada marcador exacto.
        """
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
        """Calcula la PMF de goles de un lado.

        Args:
            goals (np.ndarray): Conteos de goles a evaluar.
            lam (float): Goles esperados del lado.
            dispersion (float): Factor de dispersion, uno es Poisson.

        Returns:
            np.ndarray: Probabilidad de cada conteo de goles.
        """
        lam = max(float(lam), 1e-9)
        if dispersion <= 1.0 + 1e-9:
            return poisson.pmf(goals, lam)
        r = lam / (dispersion - 1.0)
        p = r / (r + lam)
        return nbinom.pmf(goals, r, p)

    def _apply_tau(self, matrix: np.ndarray, lh: float, la: float) -> np.ndarray:
        """Aplica la correccion Dixon-Coles a las celdas bajas.

        Args:
            matrix (np.ndarray): Matriz conjunta de marcadores.
            lh (float): Goles esperados del local.
            la (float): Goles esperados del visitante.

        Returns:
            np.ndarray: Matriz corregida sin valores negativos.
        """
        rho = self._rho
        if rho == 0.0:
            return matrix
        m = matrix.copy()
        m[0, 0] *= 1 - lh * la * rho
        m[0, 1] *= 1 + lh * rho
        m[1, 0] *= 1 + la * rho
        m[1, 1] *= 1 - rho
        return np.clip(m, 0.0, None)


def to_prediction(score_matrix: ScoreMatrix) -> MatchPrediction:
    """Agrega la matriz en las tres probabilidades 1X2.

    Args:
        score_matrix (ScoreMatrix): Matriz conjunta de marcadores.

    Returns:
        MatchPrediction: Probabilidades 1X2 del partido.
    """
    m = score_matrix.matrix
    prob_home = float(np.tril(m, -1).sum())
    prob_draw = float(np.trace(m))
    prob_away = float(np.triu(m, 1).sum())
    return MatchPrediction(
        home_team=score_matrix.home_team,
        away_team=score_matrix.away_team,
        prob_home_win=prob_home,
        prob_draw=prob_draw,
        prob_away_win=prob_away,
    )


def top_scorelines(
    score_matrix: ScoreMatrix, top_n: int = 10
) -> list[tuple[str, float]]:
    """Extrae los marcadores mas probables de la matriz.

    Args:
        score_matrix (ScoreMatrix): Matriz conjunta de marcadores.
        top_n (int): Cantidad de marcadores a devolver.

    Returns:
        list[tuple[str, float]]: Marcadores y su probabilidad.
    """
    m = score_matrix.matrix
    flat = [
        (f"{i}-{j}", float(m[i, j]))
        for i in range(m.shape[0])
        for j in range(m.shape[1])
    ]
    flat.sort(key=lambda kv: kv[1], reverse=True)
    return flat[:top_n]
