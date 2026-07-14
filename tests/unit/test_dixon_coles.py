"""Tests del estimador Dixon-Coles
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from scipy.special import gammaln

from fifa26.models.dixon_coles import DixonColesEstimator, _match_importance


def test_match_importance_asigna_los_niveles_esperados():
    tournaments = pd.Series([
        "Friendly",
        "FIFA World Cup qualification",
        "UEFA Nations League",
        "CONCACAF Nations League qualification",
        "UEFA Euro",
        "FIFA World Cup",
        "Some Regional Cup",
    ])
    weights = _match_importance(tournaments)
    assert list(weights) == [1.0, 2.0, 2.5, 2.0, 3.0, 4.0, 1.5]


def _synthetic_args(n=10, m=300, seed=0):
    rng = np.random.default_rng(seed)
    hi = rng.integers(0, n, m)
    ai = rng.integers(0, n, m)
    same = hi == ai
    ai[same] = (ai[same] + 1) % n
    hs = rng.poisson(1.3, m)
    as_ = rng.poisson(1.1, m)
    home_adv = rng.integers(0, 2, m).astype(float)
    weights = rng.uniform(0.3, 1.0, m)
    log_fact = gammaln(hs + 1) + gammaln(as_ + 1)
    return hi, ai, hs, as_, home_adv, weights, log_fact, n


def test_gradiente_analitico_coincide_con_numerico():
    args = _synthetic_args()
    n = args[-1]
    rng = np.random.default_rng(1)
    p0 = np.concatenate([[0.0, 0.3, -0.05], rng.normal(0, 0.3, n), rng.normal(0, 0.3, n)])
    err = check_grad(
        lambda p: DixonColesEstimator._neg_log_likelihood(p, *args),
        lambda p: DixonColesEstimator._neg_log_likelihood_grad(p, *args),
        p0,
    )
    assert err < 1e-3


def test_fit_converge_y_centra_los_ratings():
    rng = np.random.default_rng(2)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    for _ in range(400):
        h, a = rng.choice(teams, size=2, replace=False)
        rows.append([
            "2022-01-01", h, a, int(rng.poisson(1.4)), int(rng.poisson(1.0)),
            "Friendly", False,
        ])
    df = pd.DataFrame(rows, columns=[
        "date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral",
    ])
    df["date"] = pd.to_datetime(df["date"])
    est = DixonColesEstimator(half_life_days=540).fit(df)
    attacks = np.array([s.attack for s in est.strengths.values()])
    defenses = np.array([s.defense for s in est.strengths.values()])
    # El centrado para identificabilidad deja media cero en ataque y defensa.
    assert abs(attacks.mean()) < 1e-6
    assert abs(defenses.mean()) < 1e-6
