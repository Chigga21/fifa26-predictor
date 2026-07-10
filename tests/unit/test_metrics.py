"""Tests de las metricas 1X2
"""
from __future__ import annotations

import math

from fifa26.domain.entities import MatchPrediction, Outcome
from fifa26.evaluation.metrics import (
    brier_1x2,
    evaluate_1x2,
    log_loss_1x2,
    ranked_probability_score,
)


def _pred(home: float, draw: float, away: float) -> MatchPrediction:
    return MatchPrediction("A", "B", home, draw, away)


def test_rps_certero_es_cero():
    pred = _pred(1.0, 0.0, 0.0)
    assert ranked_probability_score(pred, Outcome.HOME_WIN) == 0.0


def test_rps_peor_caso_es_uno():
    # Predice local con certeza pero gana el visitante: maximo error ordinal.
    pred = _pred(1.0, 0.0, 0.0)
    assert ranked_probability_score(pred, Outcome.AWAY_WIN) == 1.0


def test_rps_penaliza_el_orden():
    # Confundir local con empate (adyacentes) penaliza menos que con visitante (lejano).
    pred = _pred(1.0, 0.0, 0.0)
    cerca = ranked_probability_score(pred, Outcome.DRAW)
    lejos = ranked_probability_score(pred, Outcome.AWAY_WIN)
    assert cerca < lejos


def test_log_loss_certero_es_cero():
    pred = _pred(1.0, 0.0, 0.0)
    assert log_loss_1x2(pred, Outcome.HOME_WIN) == 0.0


def test_log_loss_de_probabilidad_conocida():
    pred = _pred(0.5, 0.3, 0.2)
    assert math.isclose(log_loss_1x2(pred, Outcome.DRAW), -math.log(0.3), rel_tol=1e-9)


def test_brier_certero_es_cero():
    pred = _pred(1.0, 0.0, 0.0)
    assert brier_1x2(pred, Outcome.HOME_WIN) == 0.0


def test_evaluate_agrega_y_cuenta():
    preds = [_pred(0.7, 0.2, 0.1), _pred(0.1, 0.2, 0.7)]
    actuals = [Outcome.HOME_WIN, Outcome.HOME_WIN]  # el segundo falla
    result = evaluate_1x2("modelo", preds, actuals)
    assert result.n_matches == 2
    assert result.accuracy == 0.5
    assert result.rps > 0.0


def test_modelo_mas_afilado_tiene_menor_rps():
    actuals = [Outcome.HOME_WIN]
    afilado = evaluate_1x2("afilado", [_pred(0.9, 0.07, 0.03)], actuals)
    plano = evaluate_1x2("plano", [_pred(0.34, 0.33, 0.33)], actuals)
    assert afilado.rps < plano.rps
