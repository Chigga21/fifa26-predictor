"""Composition Root del predictor de partidos WC26 y FIFA 26.

Es el unico lugar que conoce todas las clases concretas. Cablea las dependencias por
inyeccion y lanza la UI interactiva en terminal (logo, seleccion de equipos y 1X2).
Cambiar la fuente de datos, anadir un modelo o el front-end es un cambio aqui, no en la
logica de negocio.

Autor Chigga21
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# Hace importable src/ sin instalacion.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Silencia solo el ruido conocido de las librerias (deprecaciones, avisos de uso) pero deja
# pasar los RuntimeWarning propios, como los de convergencia de los modelos.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from fifa26.application.training import Trainer  # noqa: E402
from fifa26.cli.app import InteractiveApp  # noqa: E402
from fifa26.data.cleaning import MatchCleaner  # noqa: E402
from fifa26.data.csv_repository import CsvMatchRepository  # noqa: E402
from fifa26.features.dixon_coles import DixonColesEstimator  # noqa: E402
from fifa26.models.factory import ModelFactory  # noqa: E402
from fifa26.prediction.calibration import CalibratedGoalModel  # noqa: E402
from fifa26.prediction.outcome import OutcomeCalculator  # noqa: E402
from fifa26.visualization.plots import Visualizer  # noqa: E402

DATA_DIR = ROOT / "international_results"
OUTPUT_DIR = ROOT / "outputs"


def build_trainer() -> tuple[Trainer, OutcomeCalculator]:
    repository = CsvMatchRepository(DATA_DIR / "results.csv")
    cleaner = MatchCleaner(min_year=2018, min_matches_per_team=8)
    dixon_coles = DixonColesEstimator(half_life_days=540)
    # El XGBoost lleva el guardrail de calibracion leakage-free; el bayesiano no, porque su
    # origen movil exigiria varias corridas MCMC completas.
    xgboost = CalibratedGoalModel(
        make_base=lambda: ModelFactory.create("xgboost"),
        dixon_coles_factory=lambda: DixonColesEstimator(half_life_days=540),
    )
    models = [
        xgboost,
        ModelFactory.create("bayesian", draws=1000, tune=1000, chains=4),
    ]
    outcome = OutcomeCalculator()
    trainer = Trainer(
        repository=repository,
        cleaner=cleaner,
        dixon_coles=dixon_coles,
        models=models,
        outcome_calculator=outcome,
        test_year=2025,
        max_goals=10,
    )
    return trainer, outcome


def run_interactive() -> int:
    trainer, outcome = build_trainer()
    visualizer = Visualizer(OUTPUT_DIR)
    return InteractiveApp(trainer, outcome, visualizer).run()


if __name__ == "__main__":
    rc = run_interactive()
    # Termina de inmediato sin el teardown lento del interprete (PyMC y PyTensor).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
