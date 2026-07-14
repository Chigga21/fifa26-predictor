"""Tests del modelo logistico de la tanda de penales
"""
from __future__ import annotations

import pandas as pd

from fifa26.domain import TeamStrength
from fifa26.models.shootout import ShootoutModel

_STRONG = TeamStrength(team="Strong", attack=0.5, defense=0.5)
_WEAK = TeamStrength(team="Weak", attack=-0.5, defense=-0.5)


class _StubDixonColes:
    """Estimador trivial con fuerzas fijas, barato para los tests."""

    def __init__(self, strengths):
        self.strengths = strengths

    def fit(self, matches):
        return self


def _history(years, neutral="TRUE"):
    rows = []
    for year in years:
        for month in range(1, 11):
            rows.append([
                f"{year}-{month:02d}-01", "Strong", "Weak", 2, 0, "Friendly", neutral,
            ])
    cols = [
        "date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral",
    ]
    return pd.DataFrame(rows, columns=cols)


def _shootouts(years, winner_pattern, home="Strong", away="Weak"):
    rows = []
    for year in years:
        for i in range(10):
            winner = winner_pattern(i, home, away)
            rows.append([f"{year}-{i + 1:02d}-01", home, away, winner])
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team", "winner"])


def _model(strengths):
    return ShootoutModel(
        dixon_coles_factory=lambda: _StubDixonColes(strengths),
        min_year=2000,
        window_years=8,
    )


def test_recupera_pendiente_positiva_con_favorito_dominante():
    strengths = {"Strong": _STRONG, "Weak": _WEAK}
    history = _history(range(1992, 2020))
    # El fuerte gana 8 de cada 10 tandas en sede neutral
    shootouts = _shootouts(
        range(2000, 2020), lambda i, h, a: h if i < 8 else a
    )
    model = _model(strengths).fit(shootouts, history)
    assert model.n_samples == 200
    assert model.beta > 0
    pred = model.predict(_STRONG, _WEAK, neutral=True)
    assert 0.5 < pred.prob_home < 1.0
    assert abs(pred.prob_home + pred.prob_away - 1.0) < 1e-12


def test_prediccion_neutral_es_simetrica():
    strengths = {"Strong": _STRONG, "Weak": _WEAK}
    history = _history(range(1992, 2020))
    shootouts = _shootouts(range(2000, 2020), lambda i, h, a: h if i < 7 else a)
    model = _model(strengths).fit(shootouts, history)
    direct = model.predict(_STRONG, _WEAK, neutral=True)
    reverse = model.predict(_WEAK, _STRONG, neutral=True)
    assert abs(direct.prob_home - reverse.prob_away) < 1e-12


def test_sede_local_solo_afecta_partidos_no_neutrales():
    # Equipos parejos donde el local gana 7 de 10 en sede propia
    even = {
        "Strong": TeamStrength(team="Strong", attack=0.0, defense=0.0),
        "Weak": TeamStrength(team="Weak", attack=0.0, defense=0.0),
    }
    history = _history(range(1992, 2020), neutral="FALSE")
    shootouts = _shootouts(range(2000, 2020), lambda i, h, a: h if i < 7 else a)
    model = _model(even).fit(shootouts, history)
    assert model.alpha > 0
    home_pred = model.predict(even["Strong"], even["Weak"], neutral=False)
    neutral_pred = model.predict(even["Strong"], even["Weak"], neutral=True)
    assert home_pred.prob_home > 0.5
    assert abs(neutral_pred.prob_home - 0.5) < 1e-12
