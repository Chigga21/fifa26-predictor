"""Modelo logistico del ganador de la tanda de penales.
Autor Chigga21
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from fifa26.data import clean_matches
from fifa26.domain import ShootoutPrediction, TeamStrength
from fifa26.models.dixon_coles import DixonColesEstimator


def _net_strength(strength: TeamStrength) -> float:
    """Resume ataque y defensa en una sola fuerza neta.

    Args:
        strength (TeamStrength): Ratings Dixon-Coles del equipo.

    Returns:
        float: Fuerza neta del equipo.
    """
    return strength.attack + strength.defense


class ShootoutModel:
    """Estima el ganador de la tanda con una logistica sobre la brecha de fuerza."""

    def __init__(
        self,
        dixon_coles_factory: Callable[[], DixonColesEstimator],
        min_year: int = 2000,
        window_years: int = 8,
    ) -> None:
        self._dixon_coles_factory = dixon_coles_factory
        self._min_year = min_year
        self._window_years = window_years
        self.alpha: float = 0.0
        self.beta: float = 0.0
        self.alpha_se: float = float("nan")
        self.beta_se: float = float("nan")
        self.n_samples: int = 0

    def fit(self, shootouts: pd.DataFrame, matches: pd.DataFrame) -> "ShootoutModel":
        """Calibra la logistica con las tandas historicas y ratings moviles.

        Args:
            shootouts (pd.DataFrame): Tandas crudas del dataset.
            matches (pd.DataFrame): Partidos crudos con marcador de 90 minutos.

        Returns:
            ShootoutModel: El propio modelo calibrado.
        """
        home_adv, diff, won = self._calibration_samples(shootouts, matches)
        self.n_samples = len(won)
        if self.n_samples == 0:
            warnings.warn(
                "Sin tandas de penales utilizables, el modelo queda en 50/50",
                RuntimeWarning,
                stacklevel=2,
            )
            return self
        self._fit_logistic(home_adv, diff, won)
        return self

    def predict(
        self, home: TeamStrength, away: TeamStrength, neutral: bool
    ) -> ShootoutPrediction:
        """Predice quien gana la tanda si el partido llega a penales.

        Args:
            home (TeamStrength): Ratings del equipo local.
            away (TeamStrength): Ratings del equipo visitante.
            neutral (bool): Si la sede es neutral.

        Returns:
            ShootoutPrediction: Probabilidad de cada equipo en la tanda.
        """
        diff = _net_strength(home) - _net_strength(away)
        z = self.alpha * (0.0 if neutral else 1.0) + self.beta * diff
        prob_home = float(1.0 / (1.0 + np.exp(-z)))
        return ShootoutPrediction(
            home_team=home.team,
            away_team=away.team,
            prob_home=prob_home,
            prob_away=1.0 - prob_home,
        )

    def summary(self) -> str:
        """Resume los coeficientes calibrados para el feedback de entrenamiento.

        Returns:
            str: Coeficientes con su error estandar y el tamano de muestra.
        """
        return (
            f"Shootout logistic: home={self.alpha:+.3f}+/-{self.alpha_se:.3f}  "
            f"strength gap={self.beta:+.3f}+/-{self.beta_se:.3f}  (n={self.n_samples})"
        )

    def _calibration_samples(
        self, shootouts: pd.DataFrame, matches: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Empareja cada tanda con la brecha de fuerza vigente en su fecha.

        Args:
            shootouts (pd.DataFrame): Tandas crudas del dataset.
            matches (pd.DataFrame): Partidos crudos con marcador de 90 minutos.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Sede local, brecha y triunfo local.
        """
        history = clean_matches(
            matches, min_year=self._min_year - self._window_years
        )
        s = shootouts.copy()
        s["date"] = pd.to_datetime(s["date"], errors="coerce")
        s = s.dropna(subset=["date", "winner"])
        s = s[s["date"].dt.year >= self._min_year]
        s = s.merge(
            history[["date", "home_team", "away_team", "neutral", "year"]],
            on=["date", "home_team", "away_team"],
            how="inner",
        )

        home_adv: list[float] = []
        diff: list[float] = []
        won: list[float] = []
        for year, group in s.groupby("year"):
            window = history[
                (history["year"] >= year - self._window_years)
                & (history["year"] < year)
            ]
            if window.empty:
                continue
            strengths = self._dixon_coles_factory().fit(window).strengths
            for row in group.itertuples():
                home = strengths.get(row.home_team)
                away = strengths.get(row.away_team)
                if home is None or away is None:
                    continue
                if row.winner not in (row.home_team, row.away_team):
                    continue
                home_adv.append(0.0 if row.neutral else 1.0)
                diff.append(_net_strength(home) - _net_strength(away))
                won.append(1.0 if row.winner == row.home_team else 0.0)
        return np.array(home_adv), np.array(diff), np.array(won)

    def _fit_logistic(
        self, home_adv: np.ndarray, diff: np.ndarray, won: np.ndarray
    ) -> None:
        """Ajusta los dos coeficientes por maxima verosimilitud.

        Args:
            home_adv (np.ndarray): Uno si hubo sede local, cero si neutral.
            diff (np.ndarray): Brecha de fuerza neta local menos visitante.
            won (np.ndarray): Uno si el local gano la tanda.
        """
        features = np.column_stack([home_adv, diff])

        def neg_log_likelihood(params: np.ndarray) -> float:
            z = features @ params
            return float(np.sum(np.logaddexp(0.0, -z) + (1.0 - won) * z))

        result = minimize(neg_log_likelihood, np.zeros(2), method="BFGS")
        self.alpha, self.beta = (float(v) for v in result.x)
        # La inversa aproximada del hessiano de BFGS da errores estandar orientativos.
        errors = np.sqrt(np.clip(np.diag(result.hess_inv), 0.0, None))
        self.alpha_se, self.beta_se = (float(v) for v in errors)
