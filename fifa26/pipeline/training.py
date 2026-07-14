"""Metricas 1X2, validacion de origen movil y orquestador de entrenamiento.
Autor Chigga21
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fifa26.data import (
    apply_regulation_scores,
    clean_matches,
    load_goalscorers,
    load_matches,
    load_shootouts,
)
from fifa26.models.dixon_coles import DixonColesEstimator
from fifa26.domain import MatchPrediction, Outcome, TeamStrength
from fifa26.models.goals import CalibratedDispersionEstimator, GoalModel
from fifa26.models.score_matrix import NegativeBinomialMatrixBuilder, to_prediction
from fifa26.models.shootout import ShootoutModel

_ORDER = (Outcome.HOME_WIN, Outcome.DRAW, Outcome.AWAY_WIN)


def actual_outcomes(matches: pd.DataFrame) -> list[Outcome]:
    """Deriva el resultado 1X2 real de cada partido.

    Args:
        matches (pd.DataFrame): Partidos con marcador final.

    Returns:
        list[Outcome]: Resultado real de cada partido en orden.
    """
    return [
        Outcome.from_scores(h, a)
        for h, a in zip(matches["home_score"], matches["away_score"])
    ]


@dataclass(frozen=True)
class EvaluationResult:
    """Resumen del desempeno 1X2 de un modelo sobre un conjunto de prueba."""

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


def _probs(pred: MatchPrediction) -> tuple[float, float, float]:
    """Extrae las probabilidades 1X2 en el orden canonico.

    Args:
        pred (MatchPrediction): Pronostico del partido.

    Returns:
        tuple[float, float, float]: Probabilidades local, empate, visita.
    """
    return (pred.prob_home_win, pred.prob_draw, pred.prob_away_win)


def _onehot(outcome: Outcome) -> tuple[float, float, float]:
    """Codifica el resultado real como vector one hot.

    Args:
        outcome (Outcome): Resultado real del partido.

    Returns:
        tuple[float, float, float]: Vector con un uno en el resultado.
    """
    idx = _ORDER.index(outcome)
    return tuple(1.0 if k == idx else 0.0 for k in range(3))  # type: ignore[return-value]


def ranked_probability_score(pred: MatchPrediction, actual: Outcome) -> float:
    """Calcula el RPS de un pronostico.

    Args:
        pred (MatchPrediction): Pronostico del partido.
        actual (Outcome): Resultado real.

    Returns:
        float: RPS del pronostico, menor es mejor.
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
    """Calcula el log-loss de un pronostico.

    Args:
        pred (MatchPrediction): Pronostico del partido.
        actual (Outcome): Resultado real.
        eps (float): Piso para evitar el logaritmo de cero.

    Returns:
        float: Log-loss del pronostico.
    """
    p = _probs(pred)
    prob_actual = p[_ORDER.index(actual)]
    return -math.log(min(max(prob_actual, eps), 1.0))


def brier_1x2(pred: MatchPrediction, actual: Outcome) -> float:
    """Calcula el Brier score de un pronostico.

    Args:
        pred (MatchPrediction): Pronostico del partido.
        actual (Outcome): Resultado real.

    Returns:
        float: Suma de cuadrados contra el vector one hot.
    """
    p = _probs(pred)
    o = _onehot(actual)
    return sum((pi - oi) ** 2 for pi, oi in zip(p, o))


def evaluate_1x2(
    model_name: str,
    predictions: list[MatchPrediction],
    actual_outcomes: list[Outcome],
) -> EvaluationResult:
    """Agrega las metricas 1X2 de un modelo sobre el conjunto de prueba.

    Args:
        model_name (str): Nombre del modelo evaluado.
        predictions (list[MatchPrediction]): Pronosticos del modelo.
        actual_outcomes (list[Outcome]): Resultados reales alineados.

    Returns:
        EvaluationResult: Metricas agregadas del modelo.
    """
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


def predict_fixtures(
    model: GoalModel,
    fixtures: pd.DataFrame,
    matrix_builder: NegativeBinomialMatrixBuilder,
    dispersion: CalibratedDispersionEstimator | None = None,
) -> list[MatchPrediction]:
    """Convierte cada fixture en una prediccion 1X2 con el modelo dado.

    Args:
        model (GoalModel): Modelo de goles entrenado.
        fixtures (pd.DataFrame): Partidos a predecir.
        matrix_builder (NegativeBinomialMatrixBuilder): Constructor de matrices.
        dispersion (CalibratedDispersionEstimator | None): Sin ella usa factores unitarios.

    Returns:
        list[MatchPrediction]: Prediccion de cada fixture en orden.
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
        predictions.append(to_prediction(sm))
    return predictions


def _mean_results(name: str, results: list[EvaluationResult]) -> EvaluationResult:
    """Promedia las metricas de varios pliegues.

    Args:
        name (str): Nombre del modelo.
        results (list[EvaluationResult]): Metricas por pliegue.

    Returns:
        EvaluationResult: Metricas promedio con los partidos sumados.
    """
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
    test_years: list[int],
    max_goals: int = 10,
) -> dict[str, EvaluationResult]:
    """Promedia las metricas 1X2 de cada modelo sobre varios anios de origen.

    Args:
        matches (pd.DataFrame): Partidos limpios completos.
        dixon_coles (DixonColesEstimator): Estimador de fuerzas a reajustar.
        models (list[GoalModel]): Modelos a comparar.
        test_years (list[int]): Temporadas a evaluar como origen.
        max_goals (int): Maximo de goles por lado en la matriz.

    Returns:
        dict[str, EvaluationResult]: Metricas promedio por modelo.
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
        actual = actual_outcomes(test)
        for model in models:
            model.fit(train, strengths)
            preds = predict_fixtures(model, test, matrix_builder, dispersion)
            per_model[model.name].append(evaluate_1x2(model.name, preds, actual))
    return {name: _mean_results(name, res) for name, res in per_model.items() if res}


@dataclass
class TrainedArtifacts:
    """Todo lo que las etapas posteriores necesitan tras el entrenamiento."""

    best_model: GoalModel
    best_accuracy: float
    best_rps: float
    models: list[GoalModel]
    evaluations: list[EvaluationResult]
    strengths: dict[str, TeamStrength]
    matrix_builder: NegativeBinomialMatrixBuilder
    dispersion: CalibratedDispersionEstimator
    shootout: ShootoutModel
    teams: list[str]
    train: pd.DataFrame
    test: pd.DataFrame


class Trainer:
    """Ejecuta las etapas de entrenamiento y evaluacion con dependencias inyectadas."""

    def __init__(
        self,
        results_csv,
        shootouts_csv,
        goalscorers_csv,
        dixon_coles: DixonColesEstimator,
        models: list[GoalModel],
        shootout: ShootoutModel,
        test_year: int = 2024,
        max_goals: int = 10,
    ) -> None:
        self._results_csv = results_csv
        self._shootouts_csv = shootouts_csv
        self._goalscorers_csv = goalscorers_csv
        self._dixon_coles = dixon_coles
        self._models = models
        self._shootout = shootout
        self._test_year = test_year
        self._max_goals = max_goals

        # Estado que se va poblando paso a paso
        self.raw: pd.DataFrame | None = None
        self.full: pd.DataFrame | None = None
        self.train: pd.DataFrame | None = None
        self.test: pd.DataFrame | None = None
        self.strengths: dict[str, TeamStrength] = {}
        self.teams: list[str] = []
        self.matrix_builder: NegativeBinomialMatrixBuilder | None = None
        self.dispersion: CalibratedDispersionEstimator | None = None
        self.evaluations: list[EvaluationResult] = []
        self._actual: list[Outcome] = []

    def load_and_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Carga, corrige a 90 minutos, limpia y divide con un split temporal.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Entrenamiento y prueba.
        """
        matches = clean_matches(self._load_regulation())
        self.full = matches.copy()
        # Split temporal para comparar modelos sin fuga de datos
        self.train = matches[matches["year"] < self._test_year].copy()
        self.test = matches[matches["year"] == self._test_year].copy()
        return self.train, self.test

    def _load_regulation(self) -> pd.DataFrame:
        """Carga los partidos crudos con el marcador ajustado a 90 minutos.

        Returns:
            pd.DataFrame: Partidos crudos sin los goles de la prorroga.
        """
        if self.raw is None:
            self.raw = apply_regulation_scores(
                load_matches(self._results_csv),
                load_goalscorers(self._goalscorers_csv),
            )
        return self.raw

    def fit_shootout(self) -> str:
        """Calibra el modelo de la tanda de penales con las tandas historicas.

        Returns:
            str: Resumen de los coeficientes calibrados.
        """
        shootouts = load_shootouts(self._shootouts_csv)
        self._shootout.fit(shootouts, self._load_regulation())
        return self._shootout.summary()

    def _fit_stack(
        self, matches: pd.DataFrame
    ) -> tuple[dict[str, TeamStrength], list[str], NegativeBinomialMatrixBuilder, CalibratedDispersionEstimator]:
        """Ajusta Dixon-Coles, la matriz y la dispersion sobre un conjunto de partidos.

        Args:
            matches (pd.DataFrame): Partidos limpios para el ajuste.

        Returns:
            tuple: Fuerzas, equipos ordenados, constructor de matriz y dispersion.
        """
        self._dixon_coles.fit(matches)
        strengths = self._dixon_coles.strengths
        matrix_builder = NegativeBinomialMatrixBuilder(
            self._max_goals, rho=self._dixon_coles.rho
        )
        dispersion = CalibratedDispersionEstimator().fit(matches, strengths)
        return strengths, sorted(strengths), matrix_builder, dispersion

    def fit_features(self) -> dict[str, TeamStrength]:
        """Ajusta las fuerzas y la dispersion sobre el conjunto de entrenamiento.

        Returns:
            dict[str, TeamStrength]: Fuerzas por equipo.
        """
        self.strengths, self.teams, self.matrix_builder, self.dispersion = (
            self._fit_stack(self.train)
        )
        self._actual = actual_outcomes(self.test)
        return self.strengths

    def train_model(self, model: GoalModel) -> EvaluationResult:
        """Entrena un modelo y lo evalua sobre el conjunto de prueba.

        Args:
            model (GoalModel): Modelo de goles a entrenar.

        Returns:
            EvaluationResult: Metricas 1X2 del modelo.
        """
        model.fit(self.train, self.strengths)
        preds = predict_fixtures(
            model, self.test, self.matrix_builder, self.dispersion
        )
        result = evaluate_1x2(model.name, preds, self._actual)
        self.evaluations.append(result)
        return result

    def best_evaluation(self) -> EvaluationResult:
        """Elige la mejor evaluacion por menor RPS.

        Returns:
            EvaluationResult: Evaluacion ganadora.
        """
        return min(self.evaluations, key=lambda e: e.rps)

    def best_model(self) -> GoalModel:
        """Empareja el modelo ganador por nombre.

        Returns:
            GoalModel: Modelo con la mejor evaluacion.
        """
        by_name = {m.name: m for m in self._models}
        return by_name[self.best_evaluation().model_name]

    def fit_production_features(self) -> dict[str, TeamStrength]:
        """Reentrena las fuerzas y la dispersion con todos los datos.

        Returns:
            dict[str, TeamStrength]: Fuerzas por equipo de produccion.
        """
        self.strengths, self.teams, self.matrix_builder, self.dispersion = (
            self._fit_stack(self.full)
        )
        return self.strengths

    def fit_production_model(self, model: GoalModel) -> None:
        """Reentrena un unico modelo con todos los datos.

        Args:
            model (GoalModel): Modelo de goles a reentrenar.
        """
        model.fit(self.full, self.strengths)

    def artifacts(self) -> TrainedArtifacts:
        """Empaqueta el resultado del entrenamiento para la prediccion.

        Returns:
            TrainedArtifacts: Artefactos con el estado actual de las fuerzas.
        """
        best_result = self.best_evaluation()
        best_model = self.best_model()
        return TrainedArtifacts(
            best_model=best_model,
            best_accuracy=best_result.accuracy,
            best_rps=best_result.rps,
            models=list(self._models),
            evaluations=list(self.evaluations),
            strengths=self.strengths,
            matrix_builder=self.matrix_builder,
            dispersion=self.dispersion,
            shootout=self._shootout,
            teams=self.teams,
            train=self.train,
            test=self.test,
        )

    @property
    def models(self) -> list[GoalModel]:
        """Copia de la lista de modelos."""
        return list(self._models)

    @property
    def test_year(self) -> int:
        """Temporada usada como conjunto de prueba."""
        return self._test_year

    @property
    def model_names(self) -> list[str]:
        """Nombre de cada modelo."""
        return [m.name for m in self._models]

    def cross_validate(self, years: list[int]) -> dict[str, EvaluationResult]:
        """Corre la validacion cruzada de origen movil.

        Args:
            years (list[int]): Temporadas a evaluar como origen.

        Returns:
            dict[str, EvaluationResult]: Metricas promedio por modelo.
        """
        matches = clean_matches(self._load_regulation())
        return rolling_origin_evaluate(
            matches, self._dixon_coles, self._models, years, self._max_goals
        )
