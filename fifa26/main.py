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

# Hace importable el paquete fifa26 sin instalacion.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Silencia solo el ruido conocido de las librerias (deprecaciones, avisos de uso) pero deja
# pasar los RuntimeWarning propios, como los de convergencia de los modelos.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from fifa26.training import Trainer  # noqa: E402
from fifa26.cli.app import InteractiveApp  # noqa: E402
from fifa26.dixon_coles import DixonColesEstimator  # noqa: E402
from fifa26.models import (  # noqa: E402
    BayesianPoissonModel,
    CalibratedGoalModel,
    XGBoostGoalModel,
)
from fifa26.plots import Visualizer  # noqa: E402

DATA_DIR = ROOT / "data" / "external" / "international_results"
OUTPUT_DIR = ROOT / "outputs"

# Vida media del decaimiento temporal de Dixon-Coles, en dias. Se usa dos veces (ratings de
# evaluacion y los del guardrail de calibracion) y por eso vive como una sola constante.
DIXON_COLES_HALF_LIFE_DAYS = 540


def build_trainer() -> Trainer:
    dixon_coles = DixonColesEstimator(half_life_days=DIXON_COLES_HALF_LIFE_DAYS)
    xgboost = CalibratedGoalModel(
        make_base=XGBoostGoalModel,
        dixon_coles_factory=lambda: DixonColesEstimator(
            half_life_days=DIXON_COLES_HALF_LIFE_DAYS
        ),
    )
    models = [
        xgboost,
        BayesianPoissonModel(draws=1000, tune=1000, chains=4),
    ]
    return Trainer(
        results_csv=DATA_DIR / "results.csv",
        dixon_coles=dixon_coles,
        models=models,
        test_year=2025,
        max_goals=10,
    )


def run_interactive() -> int:
    trainer = build_trainer()
    visualizer = Visualizer(OUTPUT_DIR)
    return InteractiveApp(trainer, visualizer).run()


if __name__ == "__main__":
    rc = run_interactive()
    # Termina de inmediato sin el teardown lento del interprete (PyMC y PyTensor).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
