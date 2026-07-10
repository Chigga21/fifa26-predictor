"""Tests de la ModelFactory
"""
from __future__ import annotations

import pytest

from fifa26.domain.interfaces import GoalModel
from fifa26.models.factory import ModelFactory


def test_crea_modelos_registrados():
    for name in ModelFactory.available():
        model = ModelFactory.create(name)
        assert isinstance(model, GoalModel)


def test_nombre_es_insensible_a_mayusculas():
    assert isinstance(ModelFactory.create("XGBoost"), GoalModel)


def test_nombre_desconocido_lanza_error():
    with pytest.raises(ValueError):
        ModelFactory.create("inexistente")
