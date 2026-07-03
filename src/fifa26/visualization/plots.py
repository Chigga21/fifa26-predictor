"""Visualizaciones de las predicciones con matplotlib y seaborn.
@author Chigga21
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fifa26.domain.entities import MatchPrediction, ScoreMatrix
from fifa26.evaluation.metrics import EvaluationResult

if TYPE_CHECKING:  
    from fifa26.application.prediction_service import MatchForecast

sns.set_theme(style="whitegrid")


class Visualizer:
    """Dibuja las graficas y las guarda en el directorio de salida.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def score_matrix_heatmap(self, sm: ScoreMatrix) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        sns.heatmap(
            sm.matrix * 100,
            annot=True,
            fmt=".1f",
            cmap="viridis",
            cbar_kws={"label": "Probability (%)"},
            ax=ax,
        )
        ax.set_xlabel(f"{sm.away_team} goals (away)")
        ax.set_ylabel(f"{sm.home_team} goals (home)")
        ax.set_title(
            f"Scoreline matrix  {sm.home_team} vs {sm.away_team}\n"
            f"λ_home={sm.lambda_home:.2f}  λ_away={sm.lambda_away:.2f}"
        )
        return self._save(fig, "01_scoreline_matrix.png")

    def top_scorelines(self, scorelines: list[tuple[str, float]], sm: ScoreMatrix) -> Path:
        labels = [s for s, _ in scorelines]
        values = [p * 100 for _, p in scorelines]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.barplot(x=values, y=labels, hue=labels, palette="rocket", legend=False, ax=ax)
        ax.set_xlabel("Probability (%)")
        ax.set_ylabel("Scoreline (home-away)")
        ax.set_title(f"Top {len(scorelines)} most likely scorelines  {sm.home_team} vs {sm.away_team}")
        for i, v in enumerate(values):
            ax.text(v + 0.1, i, f"{v:.1f}%", va="center")
        return self._save(fig, "02_top_scorelines.png")

    def outcome_1x2(self, prediction: MatchPrediction) -> Path:
        labels = [
            f"{prediction.home_team} win",
            "Draw",
            f"{prediction.away_team} win",
        ]
        values = [
            prediction.prob_home_win * 100,
            prediction.prob_draw * 100,
            prediction.prob_away_win * 100,
        ]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylabel("Probability (%)")
        ax.set_title(f"1X2 probabilities  {prediction.home_team} vs {prediction.away_team}")
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set_ylim(0, max(values) * 1.2)
        return self._save(fig, "03_outcome_1x2.png")

    def dispersion_comparison(self, scenarios: list[dict]) -> Path:
        """Compara la distribucion de goles del favorito antes y despues.
        """
        n = len(scenarios)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, sc in zip(axes, scenarios):
            goals = sc["goals"]
            width = 0.4
            ax.bar(goals - width / 2, sc["poisson"], width=width,
                   color="#8d99ae", label="Poisson (antes)")
            ax.bar(goals + width / 2, sc["nbinom"], width=width,
                   color="#e63946", label="Negative Binomial (despues)")
            edges = np.arange(goals[0] - 0.5, goals[-1] + 1.5, 1.0)
            sns.histplot(sc["mc_samples"], bins=edges,
                         stat="probability", element="step", fill=False,
                         color="#1d3557", ax=ax, label="Muestreo compound")
            ax.set_title(
                f"{sc['title']}\nλ={sc['lam']:.2f}  D={sc['dispersion']:.2f}"
            )
            ax.set_xlabel(f"{sc['team']} goals")
            ax.set_xticks(goals)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Probability")
        fig.suptitle("Overdispersion by favoritism · favorite goal distribution")
        return self._save(fig, "05_dispersion_comparison.png")

    def model_comparison(self, results: list[EvaluationResult]) -> Path:
        names = [r.model_name for r in results]
        values = [r.accuracy * 100 for r in results]
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(names, values, color=sns.color_palette("mako", len(names)))
        ax.set_ylabel("Accuracy 1X2 (%)")
        ax.set_title("Model comparison · 1X2 accuracy on test (higher = better)")
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set_ylim(0, max(values) * 1.25)
        return self._save(fig, "04_model_comparison.png")

    def _save(self, fig, filename: str) -> Path:
        path = self._dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path


def render_match_figures(visualizer: Visualizer, forecast: "MatchForecast") -> list[Path]:
    """Dibuja las figuras por partido de un pronostico y devuelve sus rutas
    """
    return [
        visualizer.score_matrix_heatmap(forecast.score_matrix),
        visualizer.top_scorelines(forecast.top_scorelines, forecast.score_matrix),
        visualizer.outcome_1x2(forecast.prediction),
    ]


def open_figures(paths: list[Path]) -> None:
    """Abre los PNG guardados con el visor del sistema sin fallar si no puede."""
    if not sys.stdout.isatty():
        return
    opener = _viewer_command()
    if opener is None:
        return
    for path in paths:
        try:
            subprocess.Popen(
                [*opener, str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return


def _viewer_command() -> list[str] | None:
    if sys.platform == "darwin" and shutil.which("open"):
        return ["open"]
    if sys.platform.startswith("win"):
        return ["cmd", "/c", "start", ""]
    if shutil.which("xdg-open"):
        return ["xdg-open"]
    return None
