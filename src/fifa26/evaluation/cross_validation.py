"""Validacion cruzada de origen movil para comparar modelos por temporada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fifa26.domain.entities import MatchPrediction, Outcome
from fifa26.domain.interfaces import DispersionModel, GoalModel
from fifa26.evaluation.metrics import EvaluationResult, evaluate_1x2
from fifa26.features.dixon_coles import DixonColesEstimator
from fifa26.models.dispersion import CalibratedDispersionEstimator
from fifa26.prediction.negative_binomial_matrix import NegativeBinomialMatrixBuilder
from fifa26.prediction.outcome import OutcomeCalculator


def predict_fixtures(
    model: GoalModel,
    fixtures: pd.DataFrame,
    matrix_builder: NegativeBinomialMatrixBuilder,
    outcome: OutcomeCalculator,
    dispersion: DispersionModel | None = None,
) -> list[MatchPrediction]:
    """Convierte cada fixture en una prediccion 1X2 con el modelo dado.

    Sin dispersion cae a factores unitarios, es decir a la matriz Poisson clasica.
    """
    lambda_home, lambda_away = model.predict_expected_goals(fixtures)
    if dispersion is not None:
        d_home, d_away = dispersion.predict_dispersion(fixtures)
    else:
        d_home = np.ones(len(fixtures))
        d_away = np.ones(len(fixtures))
    predictions = []
    for (_, row), lh, la, dh, da in zip(
        fixtures.iterrows(), lambda_home, lambda_away, d_home, d_away
    ):
        sm = matrix_builder.build(row["home_team"], row["away_team"], lh, la, dh, da)
        predictions.append(outcome.to_prediction(sm))
    return predictions


def _mean_results(name: str, results: list[EvaluationResult]) -> EvaluationResult:
    n = len(results)
    return EvaluationResult(
        model_name=name,
        accuracy=sum(r.accuracy for r in results) / n,
        rps=sum(r.rps for r in results) / n,
        log_loss=sum(r.log_loss for r in results) / n,
        brier=sum(r.brier for r in results) / n,
        n_matches=sum(r.n_matches for r in results),
    )


def rolling_origin_evaluate(
    matches: pd.DataFrame,
    dixon_coles: DixonColesEstimator,
    models: list[GoalModel],
    outcome: OutcomeCalculator,
    test_years: list[int],
    max_goals: int = 10,
) -> dict[str, EvaluationResult]:
    """Promedia las metricas 1X2 de cada modelo sobre varios años de origen.
    """
    per_model: dict[str, list[EvaluationResult]] = {m.name: [] for m in models}
    for year in sorted(test_years):
        train = matches[matches["year"] < year]
        test = matches[matches["year"] == year]
        if train.empty or test.empty:
            continue
        dixon_coles.fit(train)
        strengths = dixon_coles.strengths
        matrix_builder = NegativeBinomialMatrixBuilder(max_goals, rho=dixon_coles.rho)
        dispersion = CalibratedDispersionEstimator().fit(train, strengths)
        actual = [
            Outcome.from_scores(h, a)
            for h, a in zip(test["home_score"], test["away_score"])
        ]
        for model in models:
            model.fit(train, strengths)
            preds = predict_fixtures(model, test, matrix_builder, outcome, dispersion)
            per_model[model.name].append(evaluate_1x2(model.name, preds, actual))
    return {name: _mean_results(name, res) for name, res in per_model.items() if res}
