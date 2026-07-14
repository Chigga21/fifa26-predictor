"""Variables por lado, modelos de goles esperados y la cabeza de dispersion.
Autor Chigga21
"""
from __future__ import annotations

import contextlib
import io
import logging
import time
import warnings
from abc import ABC, abstractmethod
from typing import Callable

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from xgboost import XGBRegressor

from fifa26.domain import TeamStrength
from fifa26.models.dixon_coles import DixonColesEstimator

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
    return side_features(fixtures, strengths, scoring)["strength_diff"].to_numpy()


class GoalModel(ABC):
    """Modelo que predice los goles esperados de un partido."""

    name: str

    @abstractmethod
    def fit(
        self,
        matches: pd.DataFrame,
        strengths: dict[str, TeamStrength],
    ) -> "GoalModel":
        """Entrena el modelo.

        Args:
            matches (pd.DataFrame): Partidos de entrenamiento.
            strengths (dict[str, TeamStrength]): Fuerzas Dixon-Coles por equipo.

        Returns:
            GoalModel: El propio modelo entrenado.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_expected_goals(
        self, fixtures: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predice los goles esperados por lado.

        Args:
            fixtures (pd.DataFrame): Partidos a predecir.

        Returns:
            tuple[np.ndarray, np.ndarray]: Goles esperados local y visitante.
        """
        raise NotImplementedError


class XGBoostGoalModel(GoalModel):
    """Regresion Tweedie de goles con gradient boosting."""

    name = "XGBoost"

    def __init__(self, **xgb_kwargs) -> None:
        params = dict(
            objective="reg:tweedie",
            tweedie_variance_power=1.4,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        params.update(xgb_kwargs)
        self._model = XGBRegressor(**params)
        self._strengths: dict[str, TeamStrength] = {}

    def fit(self, matches: pd.DataFrame, strengths: dict[str, TeamStrength]) -> "XGBoostGoalModel":
        """Entrena el regresor con los dos lados de cada partido.

        Args:
            matches (pd.DataFrame): Partidos de entrenamiento.
            strengths (dict[str, TeamStrength]): Fuerzas Dixon-Coles por equipo.

        Returns:
            XGBoostGoalModel: El propio modelo entrenado.
        """
        self._strengths = strengths
        home = self._side_frame(matches, scoring="home")
        away = self._side_frame(matches, scoring="away")
        long_df = pd.concat([home, away], ignore_index=True)
        self._model.fit(long_df[SIDE_FEATURES], long_df["goals"])
        return self

    def predict_expected_goals(self, fixtures: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predice los goles esperados por lado.

        Args:
            fixtures (pd.DataFrame): Partidos a predecir.

        Returns:
            tuple[np.ndarray, np.ndarray]: Goles esperados local y visitante.
        """
        home_feats = side_features(fixtures, self._strengths, scoring="home")
        away_feats = side_features(fixtures, self._strengths, scoring="away")
        lambda_home = self._model.predict(home_feats[SIDE_FEATURES])
        lambda_away = self._model.predict(away_feats[SIDE_FEATURES])
        return np.asarray(lambda_home, float), np.asarray(lambda_away, float)

    def _side_frame(self, matches: pd.DataFrame, scoring: str) -> pd.DataFrame:
        """Arma las variables de un lado junto a sus goles reales.

        Args:
            matches (pd.DataFrame): Partidos de entrenamiento.
            scoring (str): Lado que anota, home o away.

        Returns:
            pd.DataFrame: Variables y goles del lado indicado.
        """
        df = side_features(matches, self._strengths, scoring=scoring)
        df["goals"] = matches["home_score" if scoring == "home" else "away_score"].to_numpy()
        return df


class BayesianPoissonModel(GoalModel):
    """Reestima ataque y defensa con incertidumbre completa via MCMC."""

    name = "Bayesian-MCMC"
    verbose_training = True

    def __init__(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.95,
        cores: int = 1,
    ) -> None:
        self._draws = draws
        self._tune = tune
        self._chains = chains
        self._target_accept = target_accept
        self._cores = cores
        self.mu: float = 0.0
        self.gamma: float = 0.0
        self._attack: dict[str, float] = {}
        self._defense: dict[str, float] = {}
        self.rhat_max: float = float("nan")
        self.ess_bulk_min: float = float("nan")
        self.divergences: int = 0
        self.on_progress: Callable[[str], None] | None = None

    def _emit(self, message: str) -> None:
        """Envia un mensaje de progreso si hay callback registrado.

        Args:
            message (str): Mensaje de fase del muestreo.
        """
        if self.on_progress is not None:
            self.on_progress(message)

    def fit(self, matches: pd.DataFrame, strengths: dict[str, TeamStrength]) -> "BayesianPoissonModel":
        """Muestrea el posterior con las fuerzas Dixon-Coles como priors.

        Args:
            matches (pd.DataFrame): Partidos de entrenamiento.
            strengths (dict[str, TeamStrength]): Fuerzas Dixon-Coles por equipo.

        Returns:
            BayesianPoissonModel: El propio modelo entrenado.
        """
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        index = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hi = matches["home_team"].map(index).to_numpy()
        ai = matches["away_team"].map(index).to_numpy()
        hs = matches["home_score"].to_numpy()
        as_ = matches["away_score"].to_numpy()
        home_adv = (~matches["neutral"].to_numpy()).astype(float)

        # Estimaciones Dixon-Coles como priors
        att_prior = np.array([strength_of(strengths, t).attack for t in teams])
        def_prior = np.array([strength_of(strengths, t).defense for t in teams])

        with pm.Model():
            mu = pm.Normal("mu", mu=0.0, sigma=1.0)
            gamma = pm.Normal("gamma", mu=0.3, sigma=0.5)
            sigma_att = pm.HalfNormal("sigma_att", sigma=1.0)
            sigma_def = pm.HalfNormal("sigma_def", sigma=1.0)

            # Desviaciones no centradas y de suma cero sobre los priors de Dixon-Coles
            attack_raw = pm.ZeroSumNormal("attack_raw", sigma=1.0, shape=n)
            defense_raw = pm.ZeroSumNormal("defense_raw", sigma=1.0, shape=n)
            attack = pm.Deterministic("attack", att_prior + attack_raw * sigma_att)
            defense = pm.Deterministic("defense", def_prior + defense_raw * sigma_def)

            log_lh = mu + gamma * home_adv + attack[hi] - defense[ai]
            log_la = mu + attack[ai] - defense[hi]
            pm.Poisson("home_goals", mu=pm.math.exp(log_lh), observed=hs)
            pm.Poisson("away_goals", mu=pm.math.exp(log_la), observed=as_)

            self._emit("Initializing NUTS...")
            self._emit(f"Sampling ({self._chains} chains in {self._cores} job(s))")
            start = time.perf_counter()
            # Silencia la salida cruda de PyMC y PyTensor para no romper la pantalla.
            with _quiet():
                idata = pm.sample(
                    draws=self._draws,
                    tune=self._tune,
                    chains=self._chains,
                    cores=self._cores,
                    target_accept=self._target_accept,
                    progressbar=False,
                    random_seed=42,
                )
            took = time.perf_counter() - start
            self._emit(
                f"Sampling {self._chains} chains for {self._tune:,} tune and "
                f"{self._draws:,} draw iterations, took {took:.0f} seconds"
            )

        post = idata.posterior
        self.mu = float(post["mu"].mean())
        self.gamma = float(post["gamma"].mean())
        att_mean = post["attack"].mean(dim=("chain", "draw")).to_numpy()
        def_mean = post["defense"].mean(dim=("chain", "draw")).to_numpy()
        self._attack = {t: float(att_mean[i]) for t, i in index.items()}
        self._defense = {t: float(def_mean[i]) for t, i in index.items()}

        self._report_diagnostics(idata)
        return self

    def _report_diagnostics(self, idata) -> None:
        """Reporta r-hat, ESS y divergencias del muestreo NUTS.

        Args:
            idata: Datos de inferencia del muestreo.
        """
        with _quiet():
            rhat = az.rhat(idata)
            ess = az.ess(idata)
        self.rhat_max = max(float(rhat[v].max()) for v in rhat.data_vars)
        self.ess_bulk_min = min(float(ess[v].min()) for v in ess.data_vars)
        if "diverging" in idata.sample_stats:
            self.divergences = int(idata.sample_stats["diverging"].sum())
        self._emit(
            f"Diagnostics: r-hat max={self.rhat_max:.3f}, "
            f"ESS min={self.ess_bulk_min:.0f}, divergences={self.divergences}"
        )
        if self.rhat_max > 1.01 or self.divergences > 0:
            warnings.warn(
                f"Convergencia bayesiana dudosa: r-hat max={self.rhat_max:.3f}, "
                f"divergencias={self.divergences}",
                RuntimeWarning,
                stacklevel=2,
            )

    def predict_expected_goals(self, fixtures: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predice los goles esperados con las medias del posterior.

        Args:
            fixtures (pd.DataFrame): Partidos a predecir.

        Returns:
            tuple[np.ndarray, np.ndarray]: Goles esperados local y visitante.
        """
        att_h = fixtures["home_team"].map(lambda t: self._attack.get(t, 0.0)).to_numpy()
        def_h = fixtures["home_team"].map(lambda t: self._defense.get(t, 0.0)).to_numpy()
        att_a = fixtures["away_team"].map(lambda t: self._attack.get(t, 0.0)).to_numpy()
        def_a = fixtures["away_team"].map(lambda t: self._defense.get(t, 0.0)).to_numpy()
        home_adv = (~fixtures["neutral"].to_numpy()).astype(float)

        lambda_home = np.exp(self.mu + self.gamma * home_adv + att_h - def_a)
        lambda_away = np.exp(self.mu + att_a - def_h)
        return lambda_home, lambda_away


@contextlib.contextmanager
def _quiet():
    """Redirige stdout y stderr y silencia los loggers de PyMC y PyTensor."""
    loggers = [logging.getLogger("pymc"), logging.getLogger("pytensor")]
    previous = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        for lg, level in zip(loggers, previous):
            lg.setLevel(level)


class CalibratedDispersionEstimator:
    """Calibra un factor de dispersion lineal en la brecha de fuerza."""

    name = "Calibrated-Gap"

    def __init__(
        self,
        n_buckets: int = 6,
        min_bucket: int = 20,
        max_dispersion: float = 1.5,
    ) -> None:
        self._n_buckets = n_buckets
        self._min_bucket = min_bucket
        self._max_dispersion = max_dispersion
        self._slope: float = 0.0
        self._strengths: dict[str, TeamStrength] = {}

    def fit(
        self,
        matches: pd.DataFrame,
        strengths: dict[str, TeamStrength],
    ) -> "CalibratedDispersionEstimator":
        """Calibra la pendiente de dispersion sobre los partidos dados.

        Args:
            matches (pd.DataFrame): Partidos de entrenamiento.
            strengths (dict[str, TeamStrength]): Fuerzas Dixon-Coles por equipo.

        Returns:
            CalibratedDispersionEstimator: El propio estimador calibrado.
        """
        self._strengths = strengths
        gaps: list[np.ndarray] = []
        goals: list[np.ndarray] = []
        for scoring in ("home", "away"):
            gaps.append(np.abs(strength_gap(matches, strengths, scoring)))
            column = "home_score" if scoring == "home" else "away_score"
            goals.append(matches[column].to_numpy(dtype=float))
        self._slope = self._fit_slope(np.concatenate(gaps), np.concatenate(goals))
        return self

    def predict_dispersion(
        self, fixtures: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predice los factores de dispersion por lado.

        Args:
            fixtures (pd.DataFrame): Partidos a predecir.

        Returns:
            tuple[np.ndarray, np.ndarray]: Factores local y visitante.
        """
        return self._dispersion(fixtures, "home"), self._dispersion(fixtures, "away")

    def _dispersion(self, fixtures: pd.DataFrame, scoring: str) -> np.ndarray:
        """Calcula el factor de dispersion de un lado.

        Args:
            fixtures (pd.DataFrame): Partidos a predecir.
            scoring (str): Lado que anota, home o away.

        Returns:
            np.ndarray: Factor de dispersion acotado por partido.
        """
        gap = np.abs(strength_gap(fixtures, self._strengths, scoring))
        return np.clip(1.0 + self._slope * gap, 1.0, self._max_dispersion)

    def _fit_slope(self, gaps: np.ndarray, goals: np.ndarray) -> float:
        """Ajusta la pendiente de dispersion por buckets de brecha.

        Args:
            gaps (np.ndarray): Brecha absoluta de fuerza por observacion.
            goals (np.ndarray): Goles anotados por observacion.

        Returns:
            float: Pendiente no negativa de la dispersion.
        """
        if gaps.size == 0:
            return 0.0
        edges = np.quantile(gaps, np.linspace(0.0, 1.0, self._n_buckets + 1))
        centers: list[float] = []
        factors: list[float] = []
        weights: list[float] = []
        for i in range(self._n_buckets):
            lo, hi = edges[i], edges[i + 1]
            if i == self._n_buckets - 1:
                mask = (gaps >= lo) & (gaps <= hi)
            else:
                mask = (gaps >= lo) & (gaps < hi)
            count = int(mask.sum())
            if count < self._min_bucket:
                continue
            mean = float(goals[mask].mean())
            if mean <= 0.0:
                continue
            variance = float(goals[mask].var())
            centers.append(float(gaps[mask].mean()))
            factors.append(max(variance / mean, 1.0))
            weights.append(float(count))
        if not centers:
            return 0.0
        c = np.asarray(centers)
        d = np.asarray(factors)
        w = np.asarray(weights)
        denom = float(np.sum(w * c * c))
        if denom <= 0.0:
            return 0.0
        slope = float(np.sum(w * c * (d - 1.0)) / denom)
        return max(slope, 0.0)


class CalibratedGoalModel(GoalModel):
    """Escala las lambdas de un modelo base con un factor validado."""

    def __init__(
        self,
        make_base: Callable[[], GoalModel],
        dixon_coles_factory: Callable[[], DixonColesEstimator],
        k_folds: int = 4,
        clip: tuple[float, float] = (0.8, 1.25),
        min_train_years: int = 2,
    ) -> None:
        self._make_base = make_base
        self._dixon_coles_factory = dixon_coles_factory
        self._k_folds = k_folds
        self._clip = clip
        self._min_train_years = min_train_years
        self._base = make_base()
        self.factor: float = 1.0
        # Conserva el nombre del base para no romper la seleccion por nombre ni la UI.
        self.name = self._base.name

    def fit(
        self,
        matches: pd.DataFrame,
        strengths: dict[str, TeamStrength],
    ) -> "CalibratedGoalModel":
        """Calibra el factor y entrena el modelo base.

        Args:
            matches (pd.DataFrame): Partidos de entrenamiento.
            strengths (dict[str, TeamStrength]): Fuerzas Dixon-Coles por equipo.

        Returns:
            CalibratedGoalModel: El propio decorador entrenado.
        """
        self.factor = self._rolling_factor(matches)
        self._base.fit(matches, strengths)
        return self

    def predict_expected_goals(
        self, fixtures: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predice los goles esperados del base escalados por el factor.

        Args:
            fixtures (pd.DataFrame): Partidos a predecir.

        Returns:
            tuple[np.ndarray, np.ndarray]: Goles esperados local y visitante.
        """
        lambda_home, lambda_away = self._base.predict_expected_goals(fixtures)
        return lambda_home * self.factor, lambda_away * self.factor

    def _rolling_factor(self, matches: pd.DataFrame) -> float:
        """Calcula el factor de goles reales sobre esperados acumulados.

        Args:
            matches (pd.DataFrame): Partidos de entrenamiento con columna year.

        Returns:
            float: Factor de calibracion acotado por el clip.
        """
        years = sorted(matches["year"].unique())
        min_year = years[0]
        folds = [
            year
            for year in years
            if year - min_year >= self._min_train_years
        ][-self._k_folds:]
        total_expected = 0.0
        total_actual = 0.0
        for year in folds:
            train = matches[matches["year"] < year]
            holdout = matches[matches["year"] == year]
            if train.empty or holdout.empty:
                continue
            dixon_coles = self._dixon_coles_factory()
            dixon_coles.fit(train)
            model = self._make_base()
            model.fit(train, dixon_coles.strengths)
            lambda_home, lambda_away = model.predict_expected_goals(holdout)
            total_expected += float(lambda_home.sum() + lambda_away.sum())
            total_actual += float(
                holdout["home_score"].sum() + holdout["away_score"].sum()
            )
        if total_expected <= 0.0:
            return 1.0
        return float(np.clip(total_actual / total_expected, *self._clip))
