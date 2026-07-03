"""Cabeza de dispersion condicionada a la brecha de fuerza.

Estima cuanta sobre-dispersion permitir en el marcador segun que tan favorito es un
equipo. Calibra un factor de dispersion D igual a varianza sobre media a partir de los
residuos de entrenamiento, agrupando los partidos por magnitud de la brecha de fuerza.
A mayor brecha mayor D, lo que la matriz Negative Binomial traduce en una cola mas ancha
del marcador, no solo en una media mayor.

La interfaz DispersionModel deja preparado el enganche para sustituir este mapeo
parametrico por un segundo regresor entrenado, por ejemplo un XGBoost sobre el residuo
al cuadrado, sin tocar el resto del pipeline.

Autor Chigga21
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fifa26.domain.entities import TeamStrength
from fifa26.domain.interfaces import DispersionModel
from fifa26.models.features import side_strength_gap


class CalibratedDispersionEstimator(DispersionModel):
    """Mapeo parametrico D igual a uno mas pendiente por la magnitud de la brecha."""

    name = "Calibrated-Gap"

    def __init__(
        self,
        n_buckets: int = 6,
        min_bucket: int = 20,
        max_dispersion: float = 1.5,
    ) -> None:
        self._n_buckets = n_buckets
        self._min_bucket = min_bucket
        self._max_dispersion = max_dispersion
        self._slope: float = 0.0
        self._strengths: dict[str, TeamStrength] = {}

    def fit(
        self,
        matches: pd.DataFrame,
        strengths: dict[str, TeamStrength],
    ) -> "CalibratedDispersionEstimator":
        self._strengths = strengths
        gaps: list[np.ndarray] = []
        goals: list[np.ndarray] = []
        for scoring in ("home", "away"):
            gaps.append(np.abs(side_strength_gap(matches, strengths, scoring)))
            column = "home_score" if scoring == "home" else "away_score"
            goals.append(matches[column].to_numpy(dtype=float))
        self._slope = self._fit_slope(np.concatenate(gaps), np.concatenate(goals))
        return self

    def predict_dispersion(
        self, fixtures: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._dispersion(fixtures, "home"), self._dispersion(fixtures, "away")

    def _dispersion(self, fixtures: pd.DataFrame, scoring: str) -> np.ndarray:
        gap = np.abs(side_strength_gap(fixtures, self._strengths, scoring))
        return np.clip(1.0 + self._slope * gap, 1.0, self._max_dispersion)

    def _fit_slope(self, gaps: np.ndarray, goals: np.ndarray) -> float:
        """Ajusta D menos uno igual a pendiente por brecha sobre buckets de brecha.

        Cada bucket aporta su D empirico igual a varianza sobre media, la pendiente sale
        de una regresion sin intercepto ponderada por el tamano del bucket y se fuerza a
        no ser negativa para que la dispersion nunca baje de Poisson.
        """
        if gaps.size == 0:
            return 0.0
        edges = np.quantile(gaps, np.linspace(0.0, 1.0, self._n_buckets + 1))
        centers: list[float] = []
        factors: list[float] = []
        weights: list[float] = []
        for i in range(self._n_buckets):
            lo, hi = edges[i], edges[i + 1]
            if i == self._n_buckets - 1:
                mask = (gaps >= lo) & (gaps <= hi)
            else:
                mask = (gaps >= lo) & (gaps < hi)
            count = int(mask.sum())
            if count < self._min_bucket:
                continue
            mean = float(goals[mask].mean())
            if mean <= 0.0:
                continue
            variance = float(goals[mask].var())
            centers.append(float(gaps[mask].mean()))
            factors.append(max(variance / mean, 1.0))
            weights.append(float(count))
        if not centers:
            return 0.0
        c = np.asarray(centers)
        d = np.asarray(factors)
        w = np.asarray(weights)
        denom = float(np.sum(w * c * c))
        if denom <= 0.0:
            return 0.0
        slope = float(np.sum(w * c * (d - 1.0)) / denom)
        return max(slope, 0.0)
