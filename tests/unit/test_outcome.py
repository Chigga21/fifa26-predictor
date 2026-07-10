"""Tests del OutcomeCalculator
"""
from __future__ import annotations

import numpy as np

from fifa26.domain.entities import ScoreMatrix
from fifa26.prediction.outcome import OutcomeCalculator


def _matrix(values) -> ScoreMatrix:
    return ScoreMatrix("A", "B", 1.0, 1.0, np.array(values, dtype=float))


def test_agregacion_1x2_por_triangulos():
    # Local gana bajo la diagonal, empate en la diagonal, visitante por encima.
    m = _matrix([[0.1, 0.2, 0.1], [0.2, 0.1, 0.05], [0.1, 0.05, 0.1]])
    pred = OutcomeCalculator().to_prediction(m)
    assert np.isclose(pred.prob_home_win, 0.2 + 0.1 + 0.05)  # triangulo inferior
    assert np.isclose(pred.prob_draw, 0.1 + 0.1 + 0.1)  # diagonal
    assert np.isclose(pred.prob_away_win, 0.2 + 0.1 + 0.05)  # triangulo superior
    total = pred.prob_home_win + pred.prob_draw + pred.prob_away_win
    assert np.isclose(total, 1.0)


def test_top_scorelines_ordena_descendente():
    m = _matrix([[0.5, 0.1], [0.3, 0.1]])
    top = OutcomeCalculator().top_scorelines(m, top_n=2)
    assert top[0] == ("0-0", 0.5)
    assert top[1] == ("1-0", 0.3)
    assert len(top) == 2
