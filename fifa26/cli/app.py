"""Experiencia de UI de terminal impresa como stream continuo.
Autor Chigga21
"""
from __future__ import annotations

from pathlib import Path

from fifa26.pipeline.predictor import MatchForecast, PredictionService
from fifa26.pipeline.training import Trainer, TrainedArtifacts
from fifa26.cli import ansi
from fifa26.cli.menu import read_line, select_team
from fifa26.cli.indicator import ProgressIndicator, run_with_dots, run_with_spinner
from fifa26.domain import Outcome
from fifa26.cli.plots import Visualizer, open_figures, render_match_figures

AUTHOR = "Said Apis (Chigga21)"
SUBTITLE = "World Cup 2026 Match Predictor"

_TITLE_PATH = Path(__file__).resolve().parents[2] / "static/TITLE.txt"

_FALLBACK = r"""
 _    _  ___ ___  __    ___  ___ ___ ___ ___ ___ _____ ___  ___
| |  | |/ __|_  )/ /   | _ \| _ \ __|   \_ _/ __|_   _/ _ \| _ \
| |/\| | (__ / // _ \  |  _/|   / _|| |) | | (__  | || (_) |   /
|__/\__|\___/___\___/  |_|  |_|_\___|___/___\___| |_| \___/|_|_\
""".strip("\n")


def _load_logo() -> str:
    """Carga el logo desde TITLE.txt con respaldo integrado.

    Returns:
        str: Texto ASCII del logo.
    """
    try:
        text = _TITLE_PATH.read_text(encoding="utf-8")
    except OSError:
        return _FALLBACK
    lines = text.rstrip("\n").split("\n")
    return "\n".join(lines) if any(line.strip() for line in lines) else _FALLBACK


def render_header() -> None:
    """Imprime el logo en color, el autor centrado y el subtitulo."""
    logo = _load_logo()
    width = max((len(line) for line in logo.splitlines()), default=0)
    author = f"By {AUTHOR}".center(width)
    print(ansi.title(logo))
    print()
    print(ansi.title(author))
    print()
    print(ansi.heading(SUBTITLE))
    print()


class InteractiveApp:
    """Conduce el flujo interactivo de entrenamiento y pronostico."""

    def __init__(
        self,
        trainer: Trainer,
        visualizer: Visualizer,
    ) -> None:
        self._trainer = trainer
        self._visualizer = visualizer
        self._service: PredictionService | None = None
        self._generate_graphs = False
        self._run_cv = False

    def run(self) -> int:
        """Corre la aplicacion completa.

        Returns:
            int: Codigo de salida del proceso.
        """
        try:
            if not self._main_menu():
                return 0
            if self._run_cv:
                self._cross_validate()
            self._prepare_service()
            if self._service is not None:
                self._predict_loop()
        except KeyboardInterrupt:
            print()
            print("  " + ansi.hint("Interrupted"))
        except Exception as exc:  # noqa BLE001 
            print()
            print("  " + ansi.error(f"Error: {exc}"))
            return 1
        finally:
            ansi.show_cursor()
        return 0

    def _main_menu(self) -> bool:
        """Muestra el banner y la configuracion de arranque.

        Returns:
            bool: Verdadero si el usuario decide continuar.
        """
        render_header()
        self._generate_graphs = self._ask_yes_no(
            "Generate graphs",
            "when on, figures are saved to outputs/ and opened after each prediction",
            "graphs",
        )
        self._run_cv = self._ask_yes_no(
            "Rolling cross-validation",
            "compares models over several seasons before training, slower",
            "cross-validation",
        )

        print("  The following ML models will be trained:")
        for name in self._trainer.model_names:
            print("    " + ansi.active("[x] ") + name)
        print()
        print("  " + ansi.hint("[ Enter ] continue    [ Q ] quit"))
        return self._ask({"enter": "go", "q": "quit"}) == "go"

    def _ask_yes_no(self, title: str, help_text: str, state_label: str) -> bool:
        """Pregunta si o no con Enter como no y muestra el estado elegido.

        Args:
            title (str): Titulo de la pregunta.
            help_text (str): Ayuda breve bajo la pregunta.
            state_label (str): Etiqueta del estado impreso al confirmar.

        Returns:
            bool: Verdadero si el usuario eligio si.
        """
        print(
            f"{title}:   "
            + " [ Y ] yes"
            + "    [ N ] no   "
            + ansi.hint("( Enter = no )")
        )
        print(ansi.hint(help_text))
        enabled = self._ask({"y": "yes", "n": "no", "enter": "no"}) == "yes"
        print("  " + ansi.hint(f"{state_label}: {'ON' if enabled else 'OFF'}"))
        print()
        return enabled

    def _prepare_service(self) -> None:
        """Entrena los modelos y construye el servicio de prediccion."""
        artifacts = self._train()
        self._service = PredictionService(artifacts)

    def _cross_validate(self) -> None:
        """Ejecuta la validacion cruzada de origen movil y muestra una tabla."""
        years = [self._trainer.test_year - 2, self._trainer.test_year - 1, self._trainer.test_year]
        print()
        results = run_with_spinner(
            f"Cross-validating over seasons {years}",
            lambda: self._trainer.cross_validate(years),
        )
        print()
        print(ansi.heading("[ * ] Rolling-origin cross-validation"))
        print()
        for result in sorted(results.values(), key=lambda r: r.rps):
            print("  " + ansi.hint(str(result)))
        print()

    def _train(self) -> TrainedArtifacts:
        """Conduce el entrenamiento completo con feedback en pantalla.

        Returns:
            TrainedArtifacts: Artefactos listos para predecir.
        """
        print()
        train, test = run_with_spinner(
            "Loading and cleaning data", self._trainer.load_and_split
        )
        print(
            "  "
            + ansi.hint(f"Training: {len(train)} matches | test: {len(test)} matches")
        )
        run_with_dots(
            "Estimating offensive and defensive strengths with Dixon–Coles",
            self._trainer.fit_features,
        )

        for model in self._trainer.models:
            label = f"Training model {model.name}"
            if getattr(model, "verbose_training", False):
                # Puntos animados en hilo aparte: el MCMC ocupa el hilo principal
                # pero el indicador sigue vivo y refleja cada fase del muestreo.
                result = self._run_verbose(
                    model, label, lambda m=model: self._trainer.train_model(m)
                )
            else:
                result = run_with_spinner(
                    label, lambda m=model: self._trainer.train_model(m)
                )
            print("    " + ansi.hint(str(result)))

        # Reentrena Dixon-Coles y TODOS los modelos con todos los datos para
        # poder mostrar ambos pronosticos a la vez.
        print()
        print("  " + ansi.focused("[*]") + " Refitting all models on the full dataset")
        run_with_dots(
            "Refitting offensive and defensive strengths with Dixon–Coles",
            self._trainer.fit_production_features,
        )
        summary = run_with_dots(
            "Calibrating penalty shootout model on historical shootouts",
            self._trainer.fit_shootout,
        )
        print("    " + ansi.hint(summary))
        for model in self._trainer.models:
            label = f"Refitting model {model.name}"
            if getattr(model, "verbose_training", False):
                self._run_verbose(
                    model, label, lambda m=model: self._trainer.fit_production_model(m)
                )
            else:
                run_with_spinner(
                    label, lambda m=model: self._trainer.fit_production_model(m)
                )

        artifacts = self._trainer.artifacts()
        print()
        print(
            "  "
            + ansi.confirm(
                f"[done] Best on {self._trainer.test_year}: {artifacts.best_model.name} "
                f"(RPS {artifacts.best_rps:.4f} | accuracy {artifacts.best_accuracy:.3f})"
            )
        )
        return artifacts

    def _run_verbose(self, model, label, fn):
        """Ajusta un modelo verboso refrescando la etiqueta con su progreso.

        Args:
            model: Modelo con callback on_progress.
            label: Etiqueta inicial del indicador.
            fn: Tarea de ajuste a ejecutar.

        Returns:
            El resultado de la tarea.
        """
        indicator = ProgressIndicator(label, style="dots", dim=True, indent="    ").start()
        model.on_progress = indicator.update
        try:
            result = fn()
        except BaseException:
            model.on_progress = None
            indicator.stop(done_message="")
            raise
        model.on_progress = None
        indicator.stop(done_message=indicator.label)
        return result

    def _predict_loop(self) -> None:
        """Repite el ciclo de eleccion de partido y pronostico."""
        assert self._service is not None
        teams = self._service.teams
        while True:
            matchup = self._choose_matchup(teams)
            if matchup is None:
                if self._ask_quit():
                    return
                continue

            home, away, neutral = matchup
            decision = self._confirm(home, away, neutral)
            if decision == "quit":
                return
            if decision != "ok": 
                continue

            forecasts = self._service.predict_all(home, away, neutral=neutral)
            self._show_results(home, away, forecasts)
            self._maybe_visualise(forecasts[0][1])

            print()
            print("  " + ansi.hint("[ N ] new match    [ Q ] quit"))
            if self._ask({"n": "again", "enter": "again", "q": "quit"}) == "quit":
                return

    def _choose_matchup(self, teams: list[str]):
        """Pide los dos equipos y la sede.

        Args:
            teams (list[str]): Equipos disponibles.

        Returns:
            tuple | None: Local, visitante y sede neutral, o None si cancela.
        """
        home = select_team(teams, "[ 2 ] Select Team A (local)")
        if home is None:
            return None
        away = select_team(
            teams, f"[ 3 ] Select Team B (away)   -   opponent of {home}", exclude=home
        )
        if away is None:
            return None
        neutral = self._choose_venue(home)
        return home, away, neutral

    def _choose_venue(self, home: str) -> bool:
        """Pregunta si la sede es neutral o local.

        Args:
            home (str): Equipo que jugaria en casa.

        Returns:
            bool: Verdadero si la sede es neutral.
        """
        print()
        print(ansi.heading("[ 4 ] Match venue"))
        print()
        print("  " + ansi.active("[N] neutral venue  ") + ansi.hint("(World Cup context)"))
        print(f"  [H] {home} plays at home")
        print()
        print("  " + ansi.hint("[ N ] neutral    [ H ] Team A local    ( Enter = neutral )"))
        return self._ask({"n": "neutral", "enter": "neutral", "h": "home"}) == "neutral"

    def _confirm(self, home: str, away: str, neutral: bool) -> str:
        """Pide confirmar el partido armado.

        Args:
            home (str): Equipo local.
            away (str): Equipo visitante.
            neutral (bool): Si la sede es neutral.

        Returns:
            str: Accion elegida, ok, edit o quit.
        """
        print()
        print(ansi.heading("[ 5 ] Confirm match "))
        print()
        venue = "neutral venue" if neutral else f"{home} plays at home"
        print("    " + ansi.active(home) + ansi.bold("   vs   ") + ansi.active(away))
        print("    " + ansi.hint(venue))
        print()
        print("  " + ansi.hint("[ Enter ] confirm   [ E ] edit    [ Q ] quit"))
        return self._ask({"enter": "ok", "e": "edit", "q": "quit"})

    def _show_results(
        self, home: str, away: str, forecasts: list[tuple[str, MatchForecast]]
    ) -> None:
        """Muestra el pronostico de cada modelo en columnas comparables."""
        print()
        print(ansi.heading(f"[ 6 ] Prediction  {home}  vs  {away}"))
        print()

        names = [name for name, _ in forecasts]

        expected = [
            f"{f.score_matrix.lambda_home:.2f} - {f.score_matrix.lambda_away:.2f}"
            for _, f in forecasts
        ]
        results = [
            self._outcome_label(f.prediction.predicted_outcome, home, away)
            for _, f in forecasts
        ]

        # Filas principales de la tabla.
        rows = [
            ("Expected goals", expected),
            (f"[1] {home} wins (90')", [f"{f.prediction.prob_home_win:5.1%}" for _, f in forecasts]),
            ("[X] Draw (90')", [f"{f.prediction.prob_draw:5.1%}" for _, f in forecasts]),
            (f"[2] {away} wins (90')", [f"{f.prediction.prob_away_win:5.1%}" for _, f in forecasts]),
            ("Most likely result (90')", results),
        ]

        depth = min(10, *(len(f.top_scorelines) for _, f in forecasts))
        scoreline_rows = [
            (
                f"{rank + 1:>2}.",
                [
                    f"{f.top_scorelines[rank][0]} {f.top_scorelines[rank][1]:4.0%}"
                    for _, f in forecasts
                ],
            )
            for rank in range(depth)
        ]

        # La tanda es condicional al empate a los 90 y no toca las filas de arriba.
        shootout_rows = [
            (f"{home} wins shootout", [f"{f.shootout.prob_home:5.1%}" for _, f in forecasts]),
            (f"{away} wins shootout", [f"{f.shootout.prob_away:5.1%}" for _, f in forecasts]),
            (f"{home} advances", [f"{f.prob_advance_home:5.1%}" for _, f in forecasts]),
            (f"{away} advances", [f"{f.prob_advance_away:5.1%}" for _, f in forecasts]),
        ]

        # Ancho de la etiqueta y de cada columna medidos sobre todo el contenido,
        # con un minimo para que las tablas cortas conserven su forma habitual.
        all_rows = [("", names), *rows, *scoreline_rows, *shootout_rows]
        label_w = max(22, *(len(label) for label, _ in all_rows)) + 2
        col_w = max(16, *(len(cell) for _, cells in all_rows for cell in cells)) + 2

        def row(label: str, cells: list[str]) -> str:
            body = "".join(f"{c:<{col_w}}" for c in cells)
            return f"  {label:<{label_w}}{body}"

        # Cabecera con el nombre de cada modelo. 
        print(ansi.bold(row("", names)))
        print()
        for label, cells in rows[:-1]:
            print(row(label, cells))
        print(ansi.confirm(row(*rows[-1])))

        print()
        print("  " + ansi.bold(f"Top {depth} scorelines (90')"))
        for label, cells in scoreline_rows:
            print(row(label, cells))

        print()
        print("  " + ansi.bold("Penalty shootout (if tied after 90 minutes)"))
        for label, cells in shootout_rows:
            print(row(label, cells))

    def _maybe_visualise(self, forecast: MatchForecast) -> None:
        """Dibuja las graficas si estan activas."""
        if not self._generate_graphs:
            return
        print()
        paths = run_with_dots(
            "Generating graphs",
            lambda: render_match_figures(self._visualizer, forecast),
        )
        open_figures(paths)
        for path in paths:
            print("    " + ansi.hint(f"- {path}"))

    @staticmethod
    def _outcome_label(outcome: Outcome, home: str, away: str) -> str:
        if outcome is Outcome.HOME_WIN:
            return f"{home} wins"
        if outcome is Outcome.AWAY_WIN:
            return f"{away} wins"
        return "Draw"

    def _ask_quit(self) -> bool:
        print()
        print("  " + ansi.hint("[ Q ] quit    [ Enter ] main menu"))
        return self._ask({"q": "quit", "enter": "back"}) == "quit"

    def _ask(self, actions: dict[str, str]) -> str:
        """Lee una linea confirmada con Enter y la resuelve a una accion de "actions"
        """
        while True:
            raw = read_line().lower()
            if raw == "":
                if "enter" in actions:
                    return actions["enter"]
                continue
            char = raw[0]
            if char in actions:
                return actions[char]
            print("  " + ansi.hint("Not an option. Try again."))
