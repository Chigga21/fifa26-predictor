"""Menu navegable con teclas de flecha para la UI interactiva de terminal.
Autor Chigga21
"""
from __future__ import annotations

import os
import sys
from collections.abc import Iterable

from fifa26.cli import ansi

try:  # Solo POSIX, si usas windows olvidate xd
    import select
    import termios
    import tty

    _HAS_TERMIOS = True
except ImportError:  
    _HAS_TERMIOS = False

WINDOW = 12  # opciones visibles a la vez
SYMBOL = "> "

_UP = "up"
_DOWN = "down"
_ENTER = "enter"
_CANCEL = "cancel"
_BACKSPACE = "backspace"


def read_line(symbol: str = SYMBOL) -> str:
    """Lee una linea confirmada con Enter mostrando el prompt fijo.

    Args:
        symbol (str): Simbolo mostrado antes del cursor.

    Returns:
        str: Linea leida sin espacios en los extremos.
    """
    try:
        return input("  " + ansi.bold(symbol)).strip()
    except EOFError as exc:  # Ctrl-D o pipe agotado
        raise KeyboardInterrupt from exc


def supported() -> bool:
    """Indica si puede correr el menu de flechas.

    Returns:
        bool: Verdadero con termios disponible y un TTY en ambos extremos.
    """
    return _HAS_TERMIOS and sys.stdin.isatty() and sys.stdout.isatty()


def filter_options(options: list[str], text: str) -> list[str]:
    """Filtra las opciones por texto, prefijos primero.

    Args:
        options (list[str]): Opciones disponibles.
        text (str): Texto del filtro.

    Returns:
        list[str]: Opciones que coinciden con el filtro.
    """
    if not text:
        return options
    needle = text.lower()
    starts = [o for o in options if o.lower().startswith(needle)]
    contains = [o for o in options if needle in o.lower() and not o.lower().startswith(needle)]
    return starts + contains


def arrow_select(
    title: str,
    options: Iterable[str],
    *,
    exclude: str | None = None,
    window: int = WINDOW,
) -> str | None:
    """Elige una opcion con las flechas, con filtro por texto.

    Args:
        title (str): Titulo mostrado sobre la lista.
        options (Iterable[str]): Opciones disponibles.
        exclude (str | None): Opcion a ocultar de la lista.
        window (int): Opciones visibles a la vez.

    Returns:
        str | None: Opcion elegida o None si se cancela.
    """
    pool = [o for o in options if o != exclude]
    print()
    print(ansi.heading(title))
    print(
        "  "
        + ansi.hint(
            "use up/down arrows to move, type to filter, Enter to select, Esc to cancel"
        )
    )

    menu = _Menu(pool, window)
    ansi.hide_cursor()
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        menu.render()
        while True:
            key = _read_key()
            if key == _ENTER:
                chosen = menu.current()
                if chosen is not None:
                    menu.finish()
                    print("  " + ansi.active(f"[x] {chosen}"))
                    return chosen
            elif key == _UP:
                menu.move(-1)
            elif key == _DOWN:
                menu.move(1)
            elif key == _CANCEL:
                menu.finish()
                print("  " + ansi.hint("cancelled"))
                return None
            elif key == _BACKSPACE:
                menu.backspace()
            elif isinstance(key, str) and len(key) == 1 and key.isprintable():
                menu.type(key)
            else:
                continue
            menu.render()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        ansi.show_cursor()


def _read_key() -> str:
    """Lee una tecla logica de stdin decodificando las secuencias de flecha.

    Returns:
        str: Tecla logica o caracter imprimible.
    """
    fd = sys.stdin.fileno()
    ch = os.read(fd, 1)
    if not ch:  # EOF
        return _CANCEL
    if ch in (b"\r", b"\n"):
        return _ENTER
    if ch in (b"\x7f", b"\b"):
        return _BACKSPACE
    if ch == b"\x03":  # Ctrl-C
        raise KeyboardInterrupt
    if ch == b"\x1b":  
        if not _pending(fd):
            return _CANCEL
        seq = os.read(fd, 2)  
        if seq[:1] == b"[":
            code = seq[1:2]
            if code == b"A":
                return _UP
            if code == b"B":
                return _DOWN
        return ""  # secuencia lateral o no manejada se ignora
    try:
        return ch.decode()
    except UnicodeDecodeError:  
        return ""


def _pending(fd: int) -> bool:
    """Indica si hay mas entrada pendiente, distingue Esc de las flechas.

    Args:
        fd (int): Descriptor de la entrada estandar.

    Returns:
        bool: Verdadero si hay mas bytes por leer.
    """
    ready, _, _ = select.select([fd], [], [], 0.05)
    return bool(ready)


class _Menu:
    """Guarda el estado de filtro, cursor y scroll del bloque redibujable."""

    def __init__(self, pool: list[str], window: int) -> None:
        self._pool = pool
        self._window = window
        self._filter = ""
        self._matches = list(pool)
        self._index = 0
        self._offset = 0
        self._lines = 0  

    def current(self) -> str | None:
        """Opcion bajo el cursor o None sin coincidencias."""
        return self._matches[self._index] if self._matches else None

    def move(self, delta: int) -> None:
        """Mueve el cursor dentro de las coincidencias.

        Args:
            delta (int): Desplazamiento del cursor.
        """
        if not self._matches:
            return
        self._index = max(0, min(len(self._matches) - 1, self._index + delta))
        self._scroll()

    def type(self, ch: str) -> None:
        """Agrega un caracter al filtro.

        Args:
            ch (str): Caracter tecleado.
        """
        self._filter += ch
        self._refilter()

    def backspace(self) -> None:
        """Borra el ultimo caracter del filtro."""
        if self._filter:
            self._filter = self._filter[:-1]
            self._refilter()

    def render(self) -> None:
        """Redibuja el bloque de opciones sobre su posicion anterior."""
        block = self._compose()
        if self._lines:
            ansi.move_up(self._lines)
        sys.stdout.write("".join("\r\033[2K" + line + "\n" for line in block))
        sys.stdout.flush()
        self._lines = len(block)

    def finish(self) -> None:
        """Borra el bloque para imprimir limpia la linea elegida."""
        if self._lines:
            ansi.move_up(self._lines)
            sys.stdout.write("\r\033[J") 
            sys.stdout.flush()
        self._lines = 0

    def _refilter(self) -> None:
        """Recalcula las coincidencias y reinicia cursor y scroll."""
        self._matches = filter_options(self._pool, self._filter)
        self._index = 0
        self._offset = 0

    def _scroll(self) -> None:
        """Desplaza la ventana para mantener visible el cursor."""
        if self._index < self._offset:
            self._offset = self._index
        elif self._index >= self._offset + self._window:
            self._offset = self._index - self._window + 1

    def _compose(self) -> list[str]:
        """Construye las lineas del bloque con altura fija.

        Returns:
            list[str]: Filas de opciones mas el pie de estado.
        """
        rows: list[str] = []
        window = self._matches[self._offset : self._offset + self._window]
        for pos, team in enumerate(window):
            absolute = self._offset + pos
            if absolute == self._index:
                rows.append("  " + ansi.focused(f"[*] {team}"))
            else:
                rows.append("  " + f"[ ] {team}")
        rows += [""] * (self._window - len(rows))  

        if not self._matches:
            footer = ansi.error(f"no matches for '{self._filter}'")
        else:
            shown = f"{self._index + 1}/{len(self._matches)}"
            filt = f"   filter: '{self._filter}'" if self._filter else ""
            footer = ansi.hint(f"{shown}{filt}")
        rows.append("  " + footer)
        return rows


def select_team(
    teams: Iterable[str],
    title: str,
    exclude: str | None = None,
    window: int = WINDOW,
) -> str | None:
    """Pide al usuario elegir un equipo de la lista.

    Args:
        teams (Iterable[str]): Equipos disponibles.
        title (str): Titulo mostrado sobre la lista.
        exclude (str | None): Equipo a ocultar de las opciones.
        window (int): Opciones visibles a la vez.

    Returns:
        str | None: Equipo elegido o None si se cancela.
    """
    teams = list(teams)
    if supported():
        return arrow_select(title, teams, exclude=exclude, window=window)
    return _select_line_based(teams, title, exclude, window)


def _select_line_based(
    teams: list[str], title: str, exclude: str | None, window: int
) -> str | None:
    """Selecciona por filtro y numero cuando no hay TTY.

    Args:
        teams (list[str]): Equipos disponibles.
        title (str): Titulo mostrado sobre la lista.
        exclude (str | None): Equipo a ocultar de las opciones.
        window (int): Opciones visibles a la vez.

    Returns:
        str | None: Equipo elegido o None si se cancela.
    """
    pool = [t for t in teams if t != exclude]
    print()
    print(ansi.heading(title))
    print(
        "  "
        + ansi.hint(
            "type part of the name and press Enter to filter; "
            "then type the number to choose. (empty Enter cancels)"
        )
    )

    shown: list[str] = []
    while True:
        raw = read_line()
        if raw == "":
            return None  # cancelar
        if raw.isdigit():
            chosen = _pick(shown, int(raw))
            if chosen is not None:
                print("  " + ansi.active(f"[x] {chosen}"))
                return chosen
            continue
        shown = _show_matches(pool, raw, window)


def _show_matches(pool: list[str], text: str, window: int) -> list[str]:
    """Imprime las coincidencias numeradas del filtro.

    Args:
        pool (list[str]): Equipos disponibles.
        text (str): Texto del filtro.
        window (int): Coincidencias visibles a la vez.

    Returns:
        list[str]: Coincidencias mostradas en pantalla.
    """
    matches = filter_options(pool, text)
    if not matches:
        print("  " + ansi.error(f"no matches for '{text}'"))
        return []
    shown = matches[:window]
    for i, team in enumerate(shown, start=1):
        print(f"  [{i:>2}] {team}")
    if len(matches) > window:
        print("  " + ansi.hint(f"... {len(matches) - window} more; refine the filter"))
    print("  " + ansi.hint("type the number to choose, or filter again"))
    return shown


def _pick(shown: list[str], number: int) -> str | None:
    """Resuelve el numero tecleado a un equipo mostrado.

    Args:
        shown (list[str]): Coincidencias visibles.
        number (int): Numero elegido por el usuario.

    Returns:
        str | None: Equipo elegido o None si el numero no es valido.
    """
    index = number - 1
    if 0 <= index < len(shown):
        return shown[index]
    if not shown:
        print("  " + ansi.error("filter first to see options"))
    else:
        print("  " + ansi.error(f"number out of range (1-{len(shown)})"))
    return None
