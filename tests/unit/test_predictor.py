"""Tests de la composicion del pronostico completo
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fifa26.domain import ShootoutPrediction, TeamStrength
from fifa26.models.score_matrix import NegativeBinomialMatrixBuilder
from fifa26.pipeline.predictor import PredictionService
from fifa26.pipeline.training import TrainedArtifacts


class _ConstantModel:
    name = "Constant"

    def fit(self, matches, strengths):
        return self

    def predict_expected_goals(self, fixtures):
        n = len(fixtures)
        return np.full(n, 1.6), np.full(n, 1.1)


class _UnitDispersion:
    def predict_dispersion(self, fixtures):
        n = len(fixtures)
        return np.ones(n), np.ones(n)


class _FixedShootout:
    def predict(self, home, away, neutral):
        return ShootoutPrediction(
            home_team=home.team, away_team=away.team, prob_home=0.6, prob_away=0.4
        )


def _artifacts():
    strengths = {
        "A": TeamStrength(team="A", attack=0.2, defense=0.1),
        "B": TeamStrength(team="B", attack=-0.1, defense=0.0),
    }
    model = _ConstantModel()
    return TrainedArtifacts(
        best_model=model,
        best_accuracy=0.0,
        best_rps=0.0,
        models=[model],
        evaluations=[],
        strengths=strengths,
        matrix_builder=NegativeBinomialMatrixBuilder(max_goals=10, rho=0.0),
        dispersion=_UnitDispersion(),
        shootout=_FixedShootout(),
        teams=["A", "B"],
        train=pd.DataFrame(),
        test=pd.DataFrame(),
    )


def test_avance_compone_90_minutos_con_la_tanda():
    service = PredictionService(_artifacts())
    _, forecast = service.predict_all("A", "B", neutral=True)[0]
    pred = forecast.prediction
    assert forecast.prob_advance_home == pred.prob_home_win + pred.prob_draw * 0.6
    assert forecast.prob_advance_away == pred.prob_away_win + pred.prob_draw * 0.4
    total = forecast.prob_advance_home + forecast.prob_advance_away
    assert abs(total - 1.0) < 1e-9


def test_la_tanda_no_altera_las_probabilidades_de_90_minutos():
    artifacts = _artifacts()
    service = PredictionService(artifacts)
    _, forecast = service.predict_all("A", "B", neutral=True)[0]
    matrix = forecast.score_matrix.matrix
    assert abs(forecast.prediction.prob_home_win - np.tril(matrix, -1).sum()) < 1e-12
    assert abs(forecast.prediction.prob_draw - np.trace(matrix)) < 1e-12
    assert abs(forecast.prediction.prob_away_win - np.triu(matrix, 1).sum()) < 1e-12
