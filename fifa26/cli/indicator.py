"""Indicador de progreso con animacion en hilo propio, desacoplada del computo.
Autor Chigga21
"""
from __future__ import annotations

import shutil
import sys
import threading
import time
from typing import Callable, TypeVar

from fifa26.cli import ansi

T = TypeVar("T")

_SPINNER_FRAMES = ("|", "/", "-", "\\")
_DOT_FRAMES = ("", ".", "..", "...")


class ProgressIndicator:
    """Anima una linea de estado en un hilo aparte mientras corre el computo."""

    def __init__(
        self,
        label: str,
        *,
        style: str = "spinner",
        dim: bool = False,
        indent: str = "  ",
        interval: float = 0.1,
    ) -> None:
        self._label = label
        self._style = style
        self._dim = dim
        self._indent = indent
        self._interval = interval
        # Se captura el stream real de la terminal al construir, antes de que
        # cualquier redireccion lo cambie. Asi la animacion sigue visible aunque
        # un paso como el MCMC envuelva sys.stdout con redirect_stdout para
        # silenciar a PyMC. La decision de color tambien se fija aqui, porque si
        # se consultara durante la redireccion veria un stream sin terminal.
        self._stream = sys.stdout
        self._tty = self._stream.isatty()
        self._color = ansi.color_enabled()
        # El ancho se usa para truncar el fotograma y que la linea nunca se
        # envuelva, porque al envolverse el \r\033[K solo limpia la fila visible
        # y el texto se repetiria concatenado en cada refresco.
        self._width = shutil.get_terminal_size(fallback=(80, 24)).columns
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def label(self) -> str:
        """Etiqueta mostrada en este momento, leida de forma segura."""
        with self._lock:
            return self._label

    def update(self, label: str) -> None:
        """Cambia la etiqueta mostrada sin cortar la animacion.

        Args:
            label (str): Nueva etiqueta a mostrar.
        """
        with self._lock:
            self._label = label

    def start(self) -> "ProgressIndicator":
        """Arranca la animacion si hay terminal.

        Returns:
            ProgressIndicator: El propio indicador.
        """
        if not self._tty:
            # Sin terminal no se anima: se imprime una vez y se sigue.
            self._write(f"{self._indent}* {self._label}\n")
            return self
        self._cursor("\033[?25l")
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def stop(self, done_message: str | None = None) -> None:
        """Detiene la animacion y deja la linea final.

        Args:
            done_message (str | None): Texto final, vacio omite la linea y
                None cierra con el marcador done.
        """
        if not self._tty:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._write("\r\033[K")
        self._cursor("\033[?25h")
        final = self._final_line(done_message)
        if final:
            self._write(final + "\n")
        self._stream.flush()

    def __enter__(self) -> "ProgressIndicator":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        # Si el bloque fallo no se imprime done, solo se limpia la linea.
        self.stop(done_message=None if exc_type else "")

    # ----------------------------------------------------------------- internos
    def _animate(self) -> None:
        """Refresca el fotograma a intervalo fijo hasta que se detenga."""
        tick = 0
        while not self._stop.is_set():
            with self._lock:
                label = self._label
            self._write("\r\033[K" + self._frame(label, tick))
            self._stream.flush()
            tick += 1
            time.sleep(self._interval)

    def _frame(self, label: str, tick: int) -> str:
        """Compone el fotograma actual segun el estilo.

        Args:
            label (str): Etiqueta a mostrar.
            tick (int): Numero de fotograma.

        Returns:
            str: Linea lista para escribir.
        """
        if self._style == "dots":
            dots = _DOT_FRAMES[tick % len(_DOT_FRAMES)]
            label = self._truncate(label, len(self._indent) + len(_DOT_FRAMES[-1]))
            text = f"{label}{dots}"
            body = self._paint(text, ansi.DIM) if self._dim else text
            return f"{self._indent}{body}"
        frame = _SPINNER_FRAMES[tick % len(_SPINNER_FRAMES)]
        label = self._truncate(label, len(self._indent) + len("[x] "))
        marker = self._paint("[" + frame + "]", ansi.BOLD, ansi._FG["bright_yellow"])
        return f"{self._indent}{marker} {label}"

    def _truncate(self, label: str, reserved: int) -> str:
        """Recorta la etiqueta para que el fotograma quepa en una fila.

        Args:
            label (str): Etiqueta a recortar.
            reserved (int): Ancho ocupado por sangrado y marcador.

        Returns:
            str: Etiqueta que cabe en la terminal.
        """
        room = self._width - reserved - 1
        if 0 < room < len(label):
            return label[:room]
        return label

    def _final_line(self, done_message: str | None) -> str:
        """Compone la linea final tras detener la animacion.

        Args:
            done_message (str | None): Texto final solicitado.

        Returns:
            str: Linea final o cadena vacia.
        """
        if done_message == "":
            return ""
        if done_message is not None:
            text = done_message
            body = self._paint(text, ansi.DIM) if self._dim else text
            return f"{self._indent}{body}"
        # Por defecto se cierra con el marcador done sobre la etiqueta actual.
        with self._lock:
            label = self._label
        done = self._paint("[done]", ansi.BOLD, ansi._FG["bright_green"])
        return f"{self._indent}{done} {label}"

    def _paint(self, text: str, *codes: str) -> str:
        """Aplica codigos ANSI segun el color fijado al construir.

        Args:
            text (str): Texto a estilizar.
            *codes (str): Codigos SGR a aplicar.

        Returns:
            str: Texto estilizado o intacto sin color.
        """
        if not codes or not self._color:
            return text
        return "".join(codes) + text + ansi.RESET

    def _write(self, text: str) -> None:
        self._stream.write(text)

    def _cursor(self, code: str) -> None:
        if self._color:
            self._stream.write(code)
            self._stream.flush()


def run_with_indicator(
    label: str,
    fn: Callable[[], T],
    *,
    style: str = "spinner",
    dim: bool = False,
    indent: str = "  ",
) -> T:
    """Corre la funcion mostrando el indicador animado.

    Args:
        label (str): Etiqueta mostrada durante la tarea.
        fn (Callable): Tarea a ejecutar en el hilo que llama.
        style (str): Estilo de animacion, spinner o dots.
        dim (bool): Si la etiqueta va con bajo enfasis.
        indent (str): Sangrado de la linea animada.

    Returns:
        T: Resultado de la tarea.
    """
    indicator = ProgressIndicator(label, style=style, dim=dim, indent=indent).start()
    try:
        result = fn()
    except BaseException:
        indicator.stop(done_message="")
        raise
    indicator.stop()
    return result


def run_with_spinner(label: str, fn: Callable[[], T]) -> T:
    """Corre la funcion con el spinner animado y cierre en done.

    Args:
        label (str): Etiqueta mostrada junto al spinner.
        fn (Callable): Tarea a ejecutar.

    Returns:
        T: Resultado de la tarea.
    """
    return run_with_indicator(label, fn, style="spinner")


def run_with_dots(label: str, fn: Callable[[], T], *, dim: bool = True) -> T:
    """Corre la funcion con puntos suspensivos animados.

    Args:
        label (str): Etiqueta mostrada junto a los puntos.
        fn (Callable): Tarea a ejecutar.
        dim (bool): Si la etiqueta va con bajo enfasis.

    Returns:
        T: Resultado de la tarea.
    """
    return run_with_indicator(label, fn, style="dots", dim=dim)
