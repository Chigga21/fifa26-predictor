"""Clase que carga, limpia, divide los datos, 
entrena los modelos de machine learning y elige al mejor.

@author Chigga21
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from fifa26.data.cleaning import MatchCleaner
from fifa26.domain.entities import MatchPrediction, Outcome, TeamStrength
from fifa26.domain.interfaces import DispersionModel, GoalModel, MatchRepository
from fifa26.evaluation.cross_validation import predict_fixtures, rolling_origin_evaluate
from fifa26.evaluation.metrics import EvaluationResult, evaluate_1x2
from fifa26.features.dixon_coles import DixonColesEstimator
from fifa26.models.dispersion import CalibratedDispersionEstimator
from fifa26.prediction.negative_binomial_matrix import NegativeBinomialMatrixBuilder
from fifa26.prediction.outcome import OutcomeCalculator


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
    dispersion: DispersionModel
    teams: list[str]
    train: pd.DataFrame
    test: pd.DataFrame


class Trainer:
    """Ejecuta las etapas de entrenamiento y evaluacion inyectando
    directamente las dependencias.
    """
    def __init__(
        self,
        repository: MatchRepository,
        cleaner: MatchCleaner,
        dixon_coles: DixonColesEstimator,
        models: list[GoalModel],
        outcome_calculator: OutcomeCalculator,
        test_year: int = 2024,
        max_goals: int = 10,
    ) -> None:
        self._repository = repository
        self._cleaner = cleaner
        self._dixon_coles = dixon_coles
        self._models = models
        self._outcome = outcome_calculator
        self._test_year = test_year
        self._max_goals = max_goals

        # Estado que se va poblando paso a paso
        self.full: pd.DataFrame | None = None
        self.train: pd.DataFrame | None = None
        self.test: pd.DataFrame | None = None
        self.strengths: dict[str, TeamStrength] = {}
        self.teams: list[str] = []
        self.matrix_builder: NegativeBinomialMatrixBuilder | None = None
        self.dispersion: DispersionModel | None = None
        self.evaluations: list[EvaluationResult] = []
        self._actual: list[Outcome] = []

        # Se reentrenan todos los modelos con todos los datos
        # para obtener la mejor fidelidad posible
        self._prod_strengths: dict[str, TeamStrength] = {}
        self._prod_teams: list[str] = []
        self._prod_matrix_builder: NegativeBinomialMatrixBuilder | None = None
        self._prod_dispersion: DispersionModel | None = None

    def load_and_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        matches = self._cleaner.clean(self._repository.load())
        self.full = matches.copy()
        # Split temporal para comparar modelos sin fuga de datos
        self.train = matches[matches["year"] < self._test_year].copy()
        self.test = matches[matches["year"] == self._test_year].copy()
        return self.train, self.test

    def fit_features(self) -> dict[str, TeamStrength]:
        self._dixon_coles.fit(self.train)
        self.strengths = self._dixon_coles.strengths
        self.teams = sorted(self.strengths)
        self.matrix_builder = NegativeBinomialMatrixBuilder(
            self._max_goals, rho=self._dixon_coles.rho
        )
        self.dispersion = CalibratedDispersionEstimator().fit(self.train, self.strengths)
        self._actual = [
            Outcome.from_scores(h, a)
            for h, a in zip(self.test["home_score"], self.test["away_score"])
        ]
        return self.strengths

    def train_model(self, model: GoalModel) -> EvaluationResult:
        model.fit(self.train, self.strengths)
        preds = self._predict_fixtures(
            model, self.test, self.matrix_builder, self.dispersion
        )
        result = evaluate_1x2(model.name, preds, self._actual)
        self.evaluations.append(result)
        return result

    def best_evaluation(self) -> EvaluationResult:
        """La mejor evaluacion por menor RPS (metrica de seleccion)"""
        return min(self.evaluations, key=lambda e: e.rps)

    def best_model(self) -> GoalModel:
        """Retorna el modelo ganador emparejado por nombre"""
        by_name = {m.name: m for m in self._models}
        return by_name[self.best_evaluation().model_name]

    def fit_production_features(self) -> dict[str, TeamStrength]:
        """Reentrena Dixon-Coles con todos los datos para la produccion."""
        self._dixon_coles.fit(self.full)
        self._prod_strengths = self._dixon_coles.strengths
        self._prod_teams = sorted(self._prod_strengths)
        self._prod_matrix_builder = NegativeBinomialMatrixBuilder(
            self._max_goals, rho=self._dixon_coles.rho
        )
        self._prod_dispersion = CalibratedDispersionEstimator().fit(
            self.full, self._prod_strengths
        )
        return self._prod_strengths

    def fit_production_model(self, model: GoalModel) -> None:
        """Reentrena un unico modelo con todos los datos de produccion."""
        model.fit(self.full, self._prod_strengths)

    def fit_production(self, on_model: Callable[[GoalModel], None] | None = None) -> None:
        """Reentrena Dixon-Coles y TODOS los modelos con todos los datos.

        Cada modelo se reentrena para que la prediccion final muestre ambos a la
        vez. El callback opcional on_model permite a la UI anunciar cada modelo
        antes de ajustarlo. Se conserva como envoltura de las dos etapas finas
        fit_production_features y fit_production_model.
        """
        self.fit_production_features()
        for model in self._models:
            if on_model is not None:
                on_model(model)
            self.fit_production_model(model)

    def artifacts(self) -> TrainedArtifacts:
        best_result = self.best_evaluation()
        best_model = self.best_model()
        # Usa los artefactos de produccion si fit_production ya corrio
        strengths = self._prod_strengths or self.strengths
        teams = self._prod_teams or self.teams
        matrix_builder = self._prod_matrix_builder or self.matrix_builder
        dispersion = self._prod_dispersion or self.dispersion
        return TrainedArtifacts(
            best_model=best_model,
            best_accuracy=best_result.accuracy,
            best_rps=best_result.rps,
            models=list(self._models),
            evaluations=list(self.evaluations),
            strengths=strengths,
            matrix_builder=matrix_builder,
            dispersion=dispersion,
            teams=teams,
            train=self.train,
            test=self.test,
        )

    @property
    def models(self) -> list[GoalModel]:
        return list(self._models)

    @property
    def test_year(self) -> int:
        return self._test_year

    @property
    def model_names(self) -> list[str]:
        return [m.name for m in self._models]

    def cross_validate(self, years: list[int]) -> dict[str, EvaluationResult]:
        """Validacion cruzada de origen movil sobre los años indicados.

        Reajusta Dixon-Coles y los modelos por pliegue, por lo que conviene
        ejecutarla antes del entrenamiento de produccion.
        """
        matches = self._cleaner.clean(self._repository.load())
        return rolling_origin_evaluate(
            matches, self._dixon_coles, self._models, self._outcome, years, self._max_goals
        )

    def _predict_fixtures(
        self,
        model: GoalModel,
        fixtures: pd.DataFrame,
        matrix_builder: NegativeBinomialMatrixBuilder,
        dispersion: DispersionModel | None = None,
    ) -> list[MatchPrediction]:
        return predict_fixtures(
            model, fixtures, matrix_builder, self._outcome, dispersion
        )
