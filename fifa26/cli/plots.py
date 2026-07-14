"""Visualizaciones de las predicciones con matplotlib y seaborn.
Autor Chigga21
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
import seaborn as sns

from fifa26.domain import MatchPrediction, ScoreMatrix

if TYPE_CHECKING:
    from fifa26.pipeline.predictor import MatchForecast

sns.set_theme(style="whitegrid")


class Visualizer:
    """Dibuja las graficas y las guarda en el directorio de salida."""

    def __init__(self, output_dir: str | Path) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def score_matrix_heatmap(self, sm: ScoreMatrix) -> Path:
        """Dibuja el mapa de calor de la matriz de marcadores.

        Args:
            sm (ScoreMatrix): Matriz conjunta de marcadores.

        Returns:
            Path: Ruta del PNG guardado.
        """
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
            f"Scoreline matrix (90 minutes)  {sm.home_team} vs {sm.away_team}\n"
            f"λ_home={sm.lambda_home:.2f}  λ_away={sm.lambda_away:.2f}"
        )
        return self._save(fig, "01_scoreline_matrix.png")

    def top_scorelines(self, scorelines: list[tuple[str, float]], sm: ScoreMatrix) -> Path:
        """Dibuja las barras de los marcadores mas probables.

        Args:
            scorelines (list[tuple[str, float]]): Marcadores y probabilidad.
            sm (ScoreMatrix): Matriz del partido para el titulo.

        Returns:
            Path: Ruta del PNG guardado.
        """
        labels = [s for s, _ in scorelines]
        values = [p * 100 for _, p in scorelines]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.barplot(x=values, y=labels, hue=labels, palette="rocket", legend=False, ax=ax)
        ax.set_xlabel("Probability (%)")
        ax.set_ylabel("Scoreline (home-away)")
        ax.set_title(f"Top {len(scorelines)} most likely scorelines (90')  {sm.home_team} vs {sm.away_team}")
        for i, v in enumerate(values):
            ax.text(v + 0.1, i, f"{v:.1f}%", va="center")
        return self._save(fig, "02_top_scorelines.png")

    def outcome_1x2(self, prediction: MatchPrediction) -> Path:
        """Dibuja las barras de probabilidad 1X2.

        Args:
            prediction (MatchPrediction): Probabilidades del partido.

        Returns:
            Path: Ruta del PNG guardado.
        """
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
        ax.set_title(f"1X2 probabilities (90 minutes)  {prediction.home_team} vs {prediction.away_team}")
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set_ylim(0, max(values) * 1.2)
        return self._save(fig, "03_outcome_1x2.png")

    def _save(self, fig, filename: str) -> Path:
        """Guarda la figura y la cierra.

        Args:
            fig: Figura de matplotlib.
            filename (str): Nombre del archivo de salida.

        Returns:
            Path: Ruta del PNG guardado.
        """
        path = self._dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path


def render_match_figures(visualizer: Visualizer, forecast: "MatchForecast") -> list[Path]:
    """Dibuja las figuras por partido de un pronostico.

    Args:
        visualizer (Visualizer): Dibujante de graficas.
        forecast (MatchForecast): Pronostico completo del partido.

    Returns:
        list[Path]: Rutas de los PNG generados.
    """
    return [
        visualizer.score_matrix_heatmap(forecast.score_matrix),
        visualizer.top_scorelines(forecast.top_scorelines, forecast.score_matrix),
        visualizer.outcome_1x2(forecast.prediction),
    ]


def open_figures(paths: list[Path]) -> None:
    """Abre los PNG guardados con el visor del sistema sin fallar si no puede.

    Args:
        paths (list[Path]): Rutas de las figuras a abrir.
    """
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
    """Elige el comando visor de imagenes de la plataforma.

    Returns:
        list[str] | None: Comando base o None si no hay visor.
    """
    if sys.platform == "darwin" and shutil.which("open"):
        return ["open"]
    if sys.platform.startswith("win"):
        return ["cmd", "/c", "start", ""]
    if shutil.which("xdg-open"):
        return ["xdg-open"]
    return None
