"""Compara la matriz de marcadores Poisson frente a la Negative Binomial.

Ilustra, para tres niveles de favoritismo, como un factor de dispersion mayor que uno
ensancha la cola del marcador del favorito sin mover la media, y confirma la equivalencia
con un muestreo Monte Carlo compound Poisson-Gamma. Ver docs/METHODOLOGY.md, seccion
Dispersion condicionada al favoritismo.

Autor Chigga21
"""


from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fifa26.models.score_matrix import NegativeBinomialMatrixBuilder  # noqa: E402

# Un escenario por nivel de favoritismo. El factor de dispersion crece con la brecha de
# fuerza, acotado por el maximo 1.5 de CalibratedDispersionEstimator (models/goals.py).
SCENARIOS = [
    ("Partido parejo", 1.3, 1.2, 1.00, 1.00),
    ("Favorito moderado", 2.2, 0.9, 1.25, 1.10),
    ("Favorito claro", 3.0, 0.6, 1.50, 1.05),
]

MAX_GOALS = 10


def _side_stats(matrix: np.ndarray, axis: int) -> tuple[float, float, float]:
    """Media, varianza y P(>=4 goles) de un lado, a partir de la marginal de la matriz."""
    pmf = matrix.sum(axis=axis)
    goals = np.arange(pmf.size)
    mean = float((pmf * goals).sum())
    variance = float((pmf * goals**2).sum() - mean**2)
    tail = float(pmf[4:].sum())
    return mean, variance, tail


def compare_scenario(name: str, lh: float, la: float, dh: float, da: float) -> dict:
    # Con dispersion unitaria el builder produce exactamente la matriz Poisson clasica.
    poisson = NegativeBinomialMatrixBuilder(MAX_GOALS).build("Home", "Away", lh, la)
    neg_binom = NegativeBinomialMatrixBuilder(MAX_GOALS).build("Home", "Away", lh, la, dh, da)

    p_mean, p_var, p_tail = _side_stats(poisson.matrix, axis=1)
    nb_mean, nb_var, nb_tail = _side_stats(neg_binom.matrix, axis=1)

    return {
        "name": name,
        "poisson": (p_mean, p_var, p_tail),
        "neg_binom": (nb_mean, nb_var, nb_tail),
    }


def monte_carlo_dispersion(lam: float, dispersion: float, n: int = 200_000) -> float:
    """Muestrea el compound Poisson-Gamma y devuelve varianza sobre media empirica.

    Con forma r = lam / (dispersion - 1) y escala (dispersion - 1), el lambda de cada
    partido es Gamma(r, escala) y los goles son Poisson(ese lambda). El resultado marginal
    es exactamente la Negative Binomial que usa NegativeBinomialMatrixBuilder.
    """
    rng = np.random.default_rng(0)
    r = lam / (dispersion - 1.0)
    sampled_lambda = rng.gamma(shape=r, scale=dispersion - 1.0, size=n)
    goals = rng.poisson(sampled_lambda)
    return float(goals.var() / goals.mean())


def main() -> None:
    print("Poisson (factor 1) vs Negative Binomial, por nivel de favoritismo")
    print("-" * 70)
    results = []
    for name, lh, la, dh, da in SCENARIOS:
        result = compare_scenario(name, lh, la, dh, da)
        results.append(result)
        p_mean, p_var, p_tail = result["poisson"]
        nb_mean, nb_var, nb_tail = result["neg_binom"]
        print(f"{name}")
        print(f"  lambda local={lh:.2f}  dispersion local={dh:.2f}")
        print(f"  Poisson       media={p_mean:.3f}  var={p_var:.3f}  P(>=4)={p_tail:.3f}")
        print(f"  Neg. Binomial media={nb_mean:.3f}  var={nb_var:.3f}  P(>=4)={nb_tail:.3f}")
        print()

    print("Muestreo Monte Carlo compound Poisson-Gamma (confirma la forma generativa)")
    print("-" * 70)
    for name, lh, la, dh, da in SCENARIOS:
        if dh <= 1.0:
            continue
        empirical = monte_carlo_dispersion(lh, dh)
        print(f"{name}: dispersion objetivo={dh:.2f}  empirica={empirical:.2f}")

    _self_check(results)
    print("\nOK: la cola del favorito se ensancha con la brecha, media identica al Poisson.")


def _self_check(results: list[dict]) -> None:
    """Comprobacion runnable: la media casi no se mueve, la varianza si crece.

    La tolerancia de la media no es exacta porque build() trunca en max_goals y
    renormaliza, lo que recorta una pizca de la cola extra de la Negative Binomial.
    La varianza es la metrica robusta del ensanchamiento, P(>=4) puede no ser monotona
    por el mismo efecto de truncamiento.
    """
    variances = []
    for result in results:
        p_mean, p_var, _ = result["poisson"]
        nb_mean, nb_var, _ = result["neg_binom"]
        assert abs(p_mean - nb_mean) < 0.05, f"la media cambio en {result['name']}"
        assert nb_var >= p_var - 1e-9, f"la varianza no crecio en {result['name']}"
        variances.append(nb_var)

    assert variances[0] < variances[1] < variances[2], "la varianza no crece con el favoritismo"


if __name__ == "__main__":
    main()
