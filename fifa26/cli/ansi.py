"""Utilidades de estilo ANSI y estados semanticos de color para la terminal.
Autor Chigga21
"""
from __future__ import annotations

import os
import sys

# ----------------------------------------------------------------- codigos SGR crudos
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

_FG = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
}


def color_enabled() -> bool:
    """Indica si el color esta activo.

    Returns:
        bool: Falso con NO_COLOR, FIFA26_NO_COLOR o sin terminal.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FIFA26_NO_COLOR") is not None:
        return False
    return sys.stdout.isatty()


def _wrap(text: str, *codes: str) -> str:
    """Envuelve el texto con codigos SGR si el color esta activo.

    Args:
        text (str): Texto a estilizar.
        *codes (str): Codigos SGR a aplicar.

    Returns:
        str: Texto estilizado o intacto sin color.
    """
    if not codes or not color_enabled():
        return text
    return "".join(codes) + text + RESET


def bold(text: str) -> str:
    """Aplica negrita al texto."""
    return _wrap(text, BOLD)


def dim(text: str) -> str:
    """Aplica bajo enfasis al texto."""
    return _wrap(text, DIM)


def color(text: str, name: str, *, bold_: bool = False) -> str:
    """Aplica un color con nombre al texto.

    Args:
        text (str): Texto a estilizar.
        name (str): Nombre del color en la paleta.
        bold_ (bool): Si ademas aplica negrita.

    Returns:
        str: Texto estilizado.
    """
    codes = (_FG.get(name, ""),)
    if bold_:
        codes = (BOLD,) + codes
    return _wrap(text, *codes)


def title(text: str) -> str:
    """Estiliza el logo y nombre del producto."""
    return color(text, "bright_white", bold_=True)


def heading(text: str) -> str:
    """Estiliza un encabezado de pantalla o seccion."""
    return color(text, "cyan", bold_=True)


def focused(text: str) -> str:
    """Estiliza la opcion bajo el cursor en un menu."""
    return color(text, "bright_yellow", bold_=True)


def active(text: str) -> str:
    """Estiliza una eleccion que el usuario ya fijo."""
    return color(text, "bright_green", bold_=True)


# Un resultado final o pronostico que conviene destacar, mismo estilo que active.
confirm = active


def hint(text: str) -> str:
    """Estiliza el texto de ayuda de bajo enfasis."""
    return dim(text)


def error(text: str) -> str:
    """Estiliza un mensaje de error."""
    return color(text, "bright_red", bold_=True)


def hide_cursor() -> None:
    """Oculta el cursor de la terminal."""
    if color_enabled():
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()


def show_cursor() -> None:
    """Muestra el cursor de la terminal."""
    if color_enabled():
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


def move_up(lines: int) -> None:
    """Sube el cursor para redibujar un bloque en sitio.

    Args:
        lines (int): Numero de filas a subir.
    """
    if lines > 0 and sys.stdout.isatty():
        sys.stdout.write(f"\033[{lines}A")
        sys.stdout.flush()
