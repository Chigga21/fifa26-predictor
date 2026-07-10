"""Carga y limpieza de los partidos del dataset international results.
Autor Chigga21
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = (
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
)


def load_matches(path: str | Path) -> pd.DataFrame:
    """Lee el CSV del dataset y valida sus columnas.

    Args:
        path (str | Path): Ruta del archivo results.csv.

    Returns:
        pd.DataFrame: Partidos crudos del dataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {path}")
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {sorted(missing)}")
    return df


def clean_matches(
    df: pd.DataFrame,
    min_year: int = 2018,
    min_matches_per_team: int = 8,
) -> pd.DataFrame:
    """Limpia los partidos crudos y los deja listos para entrenar.

    Args:
        df (pd.DataFrame): Partidos crudos del dataset.
        min_year (int): Ano minimo a conservar.
        min_matches_per_team (int): Partidos minimos por equipo.

    Returns:
        pd.DataFrame: Partidos limpios ordenados por fecha.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df = df[df["date"].dt.year >= min_year].copy()

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = _as_bool(df["neutral"])
    df["year"] = df["date"].dt.year

    df = _filter_rare_teams(df, min_matches_per_team)
    return df.sort_values("date").reset_index(drop=True)


def _as_bool(series: pd.Series) -> pd.Series:
    """Normaliza una columna de texto o numerica a booleanos.

    Args:
        series (pd.Series): Columna con valores de verdad heterogeneos.

    Returns:
        pd.Series: Columna booleana normalizada.
    """
    if series.dtype == bool:
        return series
    truthy = {"TRUE", "1", "1.0", "YES", "Y", "T"}
    falsy = {"FALSE", "0", "0.0", "NO", "N", "F", "NAN", ""}
    normalized = series.astype(str).str.strip().str.upper()
    mapping = {v: True for v in truthy}
    mapping.update({v: False for v in falsy})
    return normalized.map(mapping).fillna(False).astype(bool)


def _filter_rare_teams(df: pd.DataFrame, min_matches: int) -> pd.DataFrame:
    """Descarta los equipos con pocos partidos jugados.

    Args:
        df (pd.DataFrame): Partidos limpios.
        min_matches (int): Partidos minimos por equipo.

    Returns:
        pd.DataFrame: Partidos entre equipos con suficiente historial.
    """
    appearances = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    valid = set(appearances[appearances >= min_matches].index)
    return df[df["home_team"].isin(valid) & df["away_team"].isin(valid)]
