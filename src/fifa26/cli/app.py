"""Clase que implementa la experiencia de UI de terminal
impresa de manera continua.

@author Chigga21
"""
from __future__ import annotations

from fifa26.application.prediction_service import (
    MatchForecast,
    PredictionService,
)
from fifa26.application.training import Trainer, TrainedArtifacts
from fifa26.cli import ansi
from fifa26.cli.banner import render_header
from fifa26.cli.prompt import read_line
from fifa26.cli.selector import TeamSelector
from fifa26.cli.indicator import ProgressIndicator
from fifa26.cli.spinner import run_with_dots, run_with_spinner
from fifa26.domain.entities import Outcome
from fifa26.prediction.outcome import OutcomeCalculator
from fifa26.visualization.plots import Visualizer, open_figures, render_match_figures


class InteractiveApp:
    def __init__(
        self,
        trainer: Trainer,
        outcome_calculator: OutcomeCalculator,
        visualizer: Visualizer,
    ) -> None:
        self._trainer = trainer
        self._outcome = outcome_calculator
        self._visualizer = visualizer
        self._service: PredictionService | None = None
        self._generate_graphs = False
        self._run_cv = False

    def run(self) -> int:
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
        """Banner de bienvenida y configuracion de arranque
        Pregunta al usuario si se generarán gráficos o no de los resultados.
        """
        render_header()
        print(
            "Generate graphs:   "
            + " [ Y ] yes"
            + "    [ N ] no   "
            + ansi.hint("( Enter = no )")
        )
        print(ansi.hint("when on, figures are saved to outputs/ and opened after each prediction"))
        self._generate_graphs = self._ask({"y": "yes", "n": "no", "enter": "no"}) == "yes"
        print("  " + ansi.hint(f"graphs: {'ON' if self._generate_graphs else 'OFF'}"))
        print()

        print(
            "Rolling cross-validation:   "
            + " [ Y ] yes"
            + "    [ N ] no   "
            + ansi.hint("( Enter = no )")
        )
        print(ansi.hint("compares models over several seasons before training, slower"))
        self._run_cv = self._ask({"y": "yes", "n": "no", "enter": "no"}) == "yes"
        print("  " + ansi.hint(f"cross-validation: {'ON' if self._run_cv else 'OFF'}"))
        print()

        print("  The following ML models will be trained:")
        for name in self._trainer.model_names:
            print("    " + ansi.active("[x] ") + name)
        print()
        print("  " + ansi.hint("[ Enter ] continue    [ Q ] quit"))
        return self._ask({"enter": "go", "q": "quit"}) == "go"

    def _prepare_service(self) -> None:
        """Entrena los modelos y construye el servicio de prediccion con el resultado"""
        artifacts = self._train()
        self._service = PredictionService.from_artifacts(artifacts, self._outcome)

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
        """Ajusta un modelo verboso mostrando puntos animados que se refrescan
        con cada mensaje de progreso del propio modelo.

        El indicador anima en su hilo y model.on_progress solo cambia la
        etiqueta, asi el mensaje de fase queda vivo aunque el muestreo acapare
        el hilo principal. Al terminar se fija la ultima linea de diagnostico.
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
        assert self._service is not None
        selector = TeamSelector(self._service.teams)
        while True:
            matchup = self._choose_matchup(selector)
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

    def _choose_matchup(self, selector: TeamSelector):
        home = selector.select("[ 2 ] Select Team A (local)")
        if home is None:
            return None
        away = selector.select(
            f"[ 3 ] Select Team B (away)   -   opponent of {home}", exclude=home
        )
        if away is None:
            return None
        neutral = self._choose_venue(home)
        return home, away, neutral

    def _choose_venue(self, home: str) -> bool:
        print()
        print(ansi.heading("[ 4 ] Match venue"))
        print()
        print("  " + ansi.active("[N] neutral venue  ") + ansi.hint("(World Cup context)"))
        print(f"  [H] {home} plays at home")
        print()
        print("  " + ansi.hint("[ N ] neutral    [ H ] Team A local    ( Enter = neutral )"))
        return self._ask({"n": "neutral", "enter": "neutral", "h": "home"}) == "neutral"

    def _confirm(self, home: str, away: str, neutral: bool) -> str:
        """Devuelve la accion elegida: 'ok', 'edit' o 'quit' (sin usar excepciones)."""
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

        # Filas principales de la tabla. Se recogen antes de imprimir para medir
        # el ancho real del contenido y evitar que un nombre de pais largo, en la
        # etiqueta o en la celda de resultado, descuadre las columnas.
        rows = [
            ("Expected goals", expected),
            (f"[1] {home} wins", [f"{f.prediction.prob_home_win:5.1%}" for _, f in forecasts]),
            ("[X] Draw", [f"{f.prediction.prob_draw:5.1%}" for _, f in forecasts]),
            (f"[2] {away} wins", [f"{f.prediction.prob_away_win:5.1%}" for _, f in forecasts]),
            ("Most likely result", results),
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

        # Ancho de la etiqueta y de cada columna medidos sobre todo el contenido,
        # con un minimo para que las tablas cortas conserven su forma habitual.
        all_rows = [("", names), *rows, *scoreline_rows]
        label_w = max(22, *(len(label) for label, _ in all_rows)) + 2
        col_w = max(16, *(len(cell) for _, cells in all_rows for cell in cells)) + 2

        def row(label: str, cells: list[str]) -> str:
            body = "".join(f"{c:<{col_w}}" for c in cells)
            return f"  {label:<{label_w}}{body}"

        # Cabecera con el nombre de cada modelo. El color envuelve toda la linea
        # para no romper el ancho de las columnas.
        print(ansi.bold(row("", names)))
        print()

        for label, cells in rows[:-1]:
            print(row(label, cells))
        print(ansi.confirm(row(*rows[-1])))

        print()
        print("  " + ansi.bold(f"Top {depth} scorelines"))
        for label, cells in scoreline_rows:
            print(row(label, cells))

    # ------------------------------------------------------------- visualizacion
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
