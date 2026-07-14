"""Tests del ajuste del marcador a 90 minutos
"""
from __future__ import annotations

import pandas as pd

from fifa26.data import apply_regulation_scores

_MATCH_COLS = [
    "date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral",
]
_GOAL_COLS = ["date", "home_team", "away_team", "team", "minute"]


def _matches(rows):
    return pd.DataFrame(rows, columns=_MATCH_COLS)


def _goals(rows):
    return pd.DataFrame(rows, columns=_GOAL_COLS)


def test_resta_los_goles_de_la_prorroga():
    matches = _matches([["2022-12-01", "A", "B", 2, 1, "FIFA World Cup", "TRUE"]])
    goals = _goals([
        ["2022-12-01", "A", "B", "A", 30],
        ["2022-12-01", "A", "B", "B", 60],
        ["2022-12-01", "A", "B", "A", 105],
    ])
    out = apply_regulation_scores(matches, goals)
    assert out["home_score"].iloc[0] == 1
    assert out["away_score"].iloc[0] == 1


def test_no_toca_partidos_sin_prorroga():
    matches = _matches([
        ["2022-12-01", "A", "B", 2, 0, "Friendly", "FALSE"],
        ["2022-12-02", "C", "D", 1, 1, "Friendly", "FALSE"],
    ])
    goals = _goals([
        ["2022-12-01", "A", "B", "A", 30],
        ["2022-12-01", "A", "B", "A", 90],
    ])
    out = apply_regulation_scores(matches, goals)
    assert out["home_score"].tolist() == [2, 1]
    assert out["away_score"].tolist() == [0, 1]


def test_tanda_tras_prorroga_con_goles_queda_empate_a_los_90():
    # 2-2 tras la prorroga con un gol de cada lado en ella queda 1-1
    matches = _matches([["2022-12-01", "A", "B", 2, 2, "FIFA World Cup", "TRUE"]])
    goals = _goals([
        ["2022-12-01", "A", "B", "A", 100],
        ["2022-12-01", "A", "B", "B", 118],
    ])
    out = apply_regulation_scores(matches, goals)
    assert out["home_score"].iloc[0] == 1
    assert out["away_score"].iloc[0] == 1


def test_minutos_nulos_y_marcadores_inconsistentes_no_rompen():
    matches = _matches([["2022-12-01", "A", "B", 0, 0, "Friendly", "TRUE"]])
    goals = _goals([
        ["2022-12-01", "A", "B", "A", None],
        ["2022-12-01", "A", "B", "A", 95],
    ])
    out = apply_regulation_scores(matches, goals)
    # El gol de prorroga sobre un 0-0 inconsistente no deja marcadores negativos
    assert out["home_score"].iloc[0] == 0
    assert out["away_score"].iloc[0] == 0
