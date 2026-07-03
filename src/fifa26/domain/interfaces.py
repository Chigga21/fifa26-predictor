"""Puertos abstractos del dominio, definen los contratos
que los demas modulos deben cumplir
@author Chigga21
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from fifa26.domain.entities import TeamStrength


class MatchRepository(ABC):
    """Puerto que entrega los partidos crudos, sea cual sea el almacenamiento"""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Devuelve los partidos como DataFrame con las columnas del dataset crudo"""
        raise NotImplementedError


class GoalModel(ABC):
    """Modelo que sabe predecir los goles esperados de un partido.

    El bayesiano y el XGBoost implementan este contrato, asi se entrenan y comparan de
    forma intercambiable.
    """

    name: str

    @abstractmethod
    def fit(
        self,
        matches: pd.DataFrame,
        strengths: dict[str, TeamStrength],
    ) -> "GoalModel":
        """Entrena con los partidos y las fuerzas Dixon-Coles como variables"""
        raise NotImplementedError

    @abstractmethod
    def predict_expected_goals(
        self, fixtures: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve los arrays local y visitante alineados con los fixtures"""
        raise NotImplementedError


class DispersionModel(ABC):
    """Cabeza que estima la sobre-dispersion del marcador segun la brecha de fuerza.
    """

    name: str

    @abstractmethod
    def fit(
        self,
        matches: pd.DataFrame,
        strengths: dict[str, TeamStrength],
    ) -> "DispersionModel":
        """Calibra la relacion entre la brecha de fuerza y la dispersion"""
        raise NotImplementedError

    @abstractmethod
    def predict_dispersion(
        self, fixtures: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Factores de dispersion local y visitante alineados con los fixtures"""
        raise NotImplementedError
