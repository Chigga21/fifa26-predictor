"""Variables compartidas por lado para los modelos de goles.
Autor Chigga21
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fifa26.domain import TeamStrength

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
    """Obtiene la fuerza de un equipo con respaldo neutro.

    Args:
        strengths (dict[str, TeamStrength]): Fuerzas por equipo.
        team (str): Nombre del equipo.

    Returns:
        TeamStrength: Fuerza del equipo, neutra si no aparece.
    """
    return strengths.get(team, TeamStrength(team, 0.0, 0.0))


def side_features(
    fixtures: pd.DataFrame,
    strengths: dict[str, TeamStrength],
    scoring: str,
) -> pd.DataFrame:
    """Arma las variables de un lado para cada fixture.

    Args:
        fixtures (pd.DataFrame): Partidos a describir.
        strengths (dict[str, TeamStrength]): Fuerzas por equipo.
        scoring (str): Lado que anota, home o away.

    Returns:
        pd.DataFrame: Variables del lado indicado.
    """
    if scoring == "home":
        team, opp = fixtures["home_team"], fixtures["away_team"]
        is_home = (~fixtures["neutral"]).astype(float).to_numpy()
    else:
        team, opp = fixtures["away_team"], fixtures["home_team"]
        is_home = np.zeros(len(fixtures))

    attack = team.map(lambda t: strength_of(strengths, t).attack).to_numpy()
    opp_defense = opp.map(lambda t: strength_of(strengths, t).defense).to_numpy()
    strength_diff = attack - opp_defense
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


def strength_gap(
    fixtures: pd.DataFrame,
    strengths: dict[str, TeamStrength],
    scoring: str,
) -> np.ndarray:
    """Calcula la brecha de fuerza con signo de un lado.

    Args:
        fixtures (pd.DataFrame): Partidos a describir.
        strengths (dict[str, TeamStrength]): Fuerzas por equipo.
        scoring (str): Lado que anota, home o away.

    Returns:
        np.ndarray: Ataque propio menos defensa rival por partido.
    """
    if scoring == "home":
        team, opp = fixtures["home_team"], fixtures["away_team"]
    else:
        team, opp = fixtures["away_team"], fixtures["home_team"]
    attack = team.map(lambda t: strength_of(strengths, t).attack).to_numpy()
    opp_defense = opp.map(lambda t: strength_of(strengths, t).defense).to_numpy()
    return attack - opp_defense
