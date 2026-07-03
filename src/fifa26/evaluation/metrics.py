"""Metricas de evaluacion 1X2.

Ademas del accuracy se calcula el Ranked Probability Score (RPS), el log-loss y el Brier
score.
@author Chigga21
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from fifa26.domain.entities import MatchPrediction, Outcome

_ORDER = (Outcome.HOME_WIN, Outcome.DRAW, Outcome.AWAY_WIN)


@dataclass(frozen=True)
class EvaluationResult:
    """Resumen del desempeño 1X2 de un modelo sobre un conjunto de prueba.
    """

    model_name: str
    accuracy: float
    rps: float
    log_loss: float
    brier: float
    n_matches: int

    def __str__(self) -> str:
        return (
            f"{self.model_name:<14} RPS={self.rps:.4f}  accuracy 1X2={self.accuracy:.3f}  "
            f"logloss={self.log_loss:.3f}  (n={self.n_matches})"
        )


@dataclass(frozen=True)
class GoalsCalibration:
    """Calibracion de la media de goles, guardrail para que la sobre-dispersion no
    infle ni deprima el total de goles esperado frente al real.
    """

    rmse: float
    mae: float
    total_expected: float
    total_actual: float
    n: int

    def __str__(self) -> str:
        bias = self.total_expected - self.total_actual
        return (
            f"goles RMSE={self.rmse:.3f}  MAE={self.mae:.3f}  "
            f"esperados={self.total_expected:.1f} vs reales={self.total_actual:.1f} "
            f"(sesgo={bias:+.1f}, n={self.n})"
        )


def goals_calibration(
    expected: Sequence[float], actual: Sequence[float]
) -> GoalsCalibration:
    """Compara los goles esperados por lado con los reales sobre un conjunto de prueba"""
    if len(expected) != len(actual):
        raise ValueError("esperados y reales deben tener igual longitud")
    n = len(expected)
    if n == 0:
        return GoalsCalibration(0.0, 0.0, 0.0, 0.0, 0)
    diffs = [e - a for e, a in zip(expected, actual)]
    rmse = math.sqrt(sum(d * d for d in diffs) / n)
    mae = sum(abs(d) for d in diffs) / n
    return GoalsCalibration(rmse, mae, float(sum(expected)), float(sum(actual)), n)


def _probs(pred: MatchPrediction) -> tuple[float, float, float]:
    return (pred.prob_home_win, pred.prob_draw, pred.prob_away_win)


def _onehot(outcome: Outcome) -> tuple[float, float, float]:
    idx = _ORDER.index(outcome)
    return tuple(1.0 if k == idx else 0.0 for k in range(3))  # type: ignore[return-value]


def ranked_probability_score(pred: MatchPrediction, actual: Outcome) -> float:
    """RPS de un pronostico para un resultado real
    """
    p = _probs(pred)
    o = _onehot(actual)
    cum_p = cum_o = 0.0
    total = 0.0
    for i in range(len(_ORDER) - 1):  # r - 1 = 2 terminos
        cum_p += p[i]
        cum_o += o[i]
        total += (cum_p - cum_o) ** 2
    return total / (len(_ORDER) - 1)


def log_loss_1x2(pred: MatchPrediction, actual: Outcome, eps: float = 1e-15) -> float:
    """Retorna el log-loss de un pronostico"""
    p = _probs(pred)
    prob_actual = p[_ORDER.index(actual)]
    return -math.log(min(max(prob_actual, eps), 1.0))


def brier_1x2(pred: MatchPrediction, actual: Outcome) -> float:
    """Calcula la suma de cuadrados entre las probabilidades y el resultado real"""
    p = _probs(pred)
    o = _onehot(actual)
    return sum((pi - oi) ** 2 for pi, oi in zip(p, o))


def evaluate_1x2(
    model_name: str,
    predictions: list[MatchPrediction],
    actual_outcomes: list[Outcome],
) -> EvaluationResult:
    """Agrega el accuracy, RPS, log-loss y Brier de un modelo sobre el conjunto de prueba"""
    if len(predictions) != len(actual_outcomes):
        raise ValueError("predicciones y resultados reales deben tener igual longitud")
    if not predictions:
        return EvaluationResult(model_name, 0.0, 0.0, 0.0, 0.0, 0)

    n = len(predictions)
    pairs = list(zip(predictions, actual_outcomes))
    accuracy = sum(pred.predicted_outcome == actual for pred, actual in pairs) / n
    rps = sum(ranked_probability_score(pred, actual) for pred, actual in pairs) / n
    log_loss = sum(log_loss_1x2(pred, actual) for pred, actual in pairs) / n
    brier = sum(brier_1x2(pred, actual) for pred, actual in pairs) / n
    return EvaluationResult(model_name, accuracy, rps, log_loss, brier, n)
