from __future__ import annotations

import pandas as pd

from fifa26.data.cleaning import MatchCleaner


def _raw(rows):
    cols = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]
    return pd.DataFrame(rows, columns=cols)


def test_filtra_anos_y_castea_marcadores():
    raw = _raw([
        ["2017-01-01", "A", "B", "1", "0", "Friendly", "FALSE"],  # antes de min_year
        ["2020-06-01", "A", "B", "2", "1", "Friendly", "TRUE"],
    ])
    out = MatchCleaner(min_year=2018, min_matches_per_team=1).clean(raw)
    assert len(out) == 1
    assert out["year"].iloc[0] == 2020
    assert out["home_score"].dtype.kind == "i"
    assert out["neutral"].dtype == bool


def test_descarta_marcadores_nulos():
    raw = _raw([
        ["2020-01-01", "A", "B", None, "1", "Friendly", "FALSE"],
        ["2020-02-01", "A", "B", "1", "1", "Friendly", "FALSE"],
    ])
    out = MatchCleaner(min_year=2018, min_matches_per_team=1).clean(raw)
    assert len(out) == 1


def test_neutral_acepta_varias_codificaciones():
    raw = _raw([
        ["2020-01-01", "A", "B", "1", "0", "Friendly", "TRUE"],
        ["2020-02-01", "A", "B", "1", "0", "Friendly", "1"],
        ["2020-03-01", "A", "B", "1", "0", "Friendly", "false"],
        ["2020-04-01", "A", "B", "1", "0", "Friendly", "0"],
    ])
    out = MatchCleaner(min_year=2018, min_matches_per_team=1).clean(raw)
    assert out["neutral"].tolist() == [True, True, False, False]


def test_descarta_equipos_raros():
    rows = [["2020-01-0%d" % (i + 1), "A", "B", "1", "0", "Friendly", "FALSE"] for i in range(3)]
    rows.append(["2020-02-01", "A", "C", "1", "0", "Friendly", "FALSE"])  # C aparece una vez
    out = MatchCleaner(min_year=2018, min_matches_per_team=3).clean(_raw(rows))
    teams = set(out["home_team"]) | set(out["away_team"])
    assert "C" not in teams
    assert {"A", "B"} <= teams
