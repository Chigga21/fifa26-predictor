"""Tests del indicador de progreso animado.

Verifican que la animacion sigue llegando a la terminal capturada aunque otro
paso redirija sys.stdout, que es justo lo que hace el MCMC para silenciar a
PyMC, y que sin terminal el indicador no anima.
"""
from __future__ import annotations

import contextlib
import io
import sys
import time

from fifa26.cli.indicator import ProgressIndicator


class _FakeTty(io.StringIO):
    """Stream en memoria que se hace pasar por terminal y cuenta fotogramas."""

    def __init__(self) -> None:
        super().__init__()
        self.frames = 0

    def isatty(self) -> bool:
        return True

    def write(self, s: str) -> int:
        if "\r" in s:
            self.frames += 1
        return super().write(s)


def test_animation_survives_redirect_stdout(monkeypatch):
    """La animacion sigue dibujandose aunque sys.stdout este redirigido."""
    terminal = _FakeTty()
    monkeypatch.setattr(sys, "stdout", terminal)

    indicator = ProgressIndicator("Sampling", style="dots", interval=0.02).start()
    before = terminal.frames
    # Imita _quiet() del modelo bayesiano: sustituye sys.stdout durante el computo.
    with contextlib.redirect_stdout(io.StringIO()):
        time.sleep(0.2)
    during = terminal.frames - before
    indicator.stop(done_message="")

    assert during > 0


def test_long_label_is_truncated_to_terminal_width():
    """Una etiqueta larga se recorta para que el fotograma no se envuelva.

    Sin recorte la linea envuelta haria que el \\r\\033[K solo limpie una fila y
    el texto se repita concatenado en cada refresco.
    """
    long_label = "Sampling 4 chains for 1,000 tune and 1,000 draw iterations, took 61 seconds"
    for style in ("dots", "spinner"):
        indicator = ProgressIndicator(long_label, style=style)
        indicator._width = 30  # terminal angosta
        frame = indicator._frame(indicator.label, tick=1)
        assert len(frame) < indicator._width


def test_non_tty_does_not_animate(monkeypatch):
    """Sin terminal se imprime la etiqueta una vez y no se lanza el hilo."""
    plain = io.StringIO()
    monkeypatch.setattr(sys, "stdout", plain)

    indicator = ProgressIndicator("Loading", style="spinner").start()
    indicator.stop()

    assert indicator._thread is None
    assert "Loading" in plain.getvalue()
