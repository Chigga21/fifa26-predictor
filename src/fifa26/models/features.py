"""Feature engineering compartido de los modelos de goles.

Centraliza como se arma el vector de variables por lado a partir de las fuerzas
Dixon-Coles, para que el modelo de la media y la cabeza de dispersion usen
exactamente la misma definicion de la brecha de fuerza.

Autor Chigga21
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fifa26.domain.entities import TeamStrength

# Variables que consume el modelo de goles. La brecha de fuerza aparece con signo
# en strength_diff y con magnitud en abs_strength_diff, que marca el favoritismo.
# is_home marca la ventaja de local con sede propia e is_neutral marca la cancha
# neutral, que lo complementa para que un partido neutral no se confunda con el
# lado visitante de un partido normal, donde el modelo antes lo aplastaba.
SIDE_FEATURES = [
    "attack",
    "opp_defense",
    "strength_diff",
    "abs_strength_diff",
    "is_home",
    "is_neutral",
    "is_competitive",
]


def strength_of(strengths: dict[str, TeamStrength], team: str) -> TeamStrength:
    """Fuerza del equipo, neutra si no aparece en el diccionario"""
    return strengths.get(team, TeamStrength(team, 0.0, 0.0))


def side_features(
    fixtures: pd.DataFrame,
    strengths: dict[str, TeamStrength],
    scoring: str,
) -> pd.DataFrame:
    """Arma las variables de un lado, local o visitante, para cada fixture"""
    if scoring == "home":
        team, opp = fixtures["home_team"], fixtures["away_team"]
        is_home = (~fixtures["neutral"]).astype(float).to_numpy()
    else:
        team, opp = fixtures["away_team"], fixtures["home_team"]
        is_home = np.zeros(len(fixtures))

    attack = team.map(lambda t: strength_of(strengths, t).attack).to_numpy()
    opp_defense = opp.map(lambda t: strength_of(strengths, t).defense).to_numpy()
    strength_diff = attack - opp_defense
    # is_neutral es igual para ambos lados, marca la cancha neutral del partido.
    is_neutral = fixtures["neutral"].astype(float).to_numpy()
    competitive = (fixtures["tournament"] != "Friendly").astype(float).to_numpy()
    return pd.DataFrame(
        {
            "attack": attack,
            "opp_defense": opp_defense,
            "strength_diff": strength_diff,
            "abs_strength_diff": np.abs(strength_diff),
            "is_home": np.asarray(is_home, float),
            "is_neutral": is_neutral,
            "is_competitive": competitive,
        }
    )


def side_strength_gap(
    fixtures: pd.DataFrame,
    strengths: dict[str, TeamStrength],
    scoring: str,
) -> np.ndarray:
    """Brecha de fuerza con signo de un lado, ataque menos defensa rival"""
    return side_features(fixtures, strengths, scoring)["strength_diff"].to_numpy()
