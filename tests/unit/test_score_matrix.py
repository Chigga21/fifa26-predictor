"""Tests de la matriz de marcadores con dispersion unitaria, la Poisson clasica.
Autor Chigga21
"""
from __future__ import annotations

import numpy as np

from fifa26.models.score_matrix import NegativeBinomialMatrixBuilder


def test_matriz_suma_uno_y_forma_correcta():
    builder = NegativeBinomialMatrixBuilder(max_goals=8, rho=0.0)
    sm = builder.build("A", "B", 1.4, 1.1)
    assert sm.matrix.shape == (9, 9)
    assert np.isclose(sm.matrix.sum(), 1.0)
    assert (sm.matrix >= 0).all()


def test_tau_modifica_las_celdas_bajas():
    sin_tau = NegativeBinomialMatrixBuilder(max_goals=8, rho=0.0).build("A", "B", 1.2, 1.0)
    con_tau = NegativeBinomialMatrixBuilder(max_goals=8, rho=0.1).build("A", "B", 1.2, 1.0)
    # La correccion cambia la matriz pero conserva la normalizacion.
    assert not np.allclose(sin_tau.matrix, con_tau.matrix)
    assert np.isclose(con_tau.matrix.sum(), 1.0)


def test_lambdas_se_guardan():
    sm = NegativeBinomialMatrixBuilder(max_goals=5).build("A", "B", 2.0, 0.5)
    assert sm.lambda_home == 2.0
    assert sm.lambda_away == 0.5
