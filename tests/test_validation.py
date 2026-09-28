from __future__ import annotations

import pandas as pd
import pytest

from baloto_ml.data.schema import canonicalize
from baloto_ml.data.validation import (
    DataValidationError,
    find_conflicts,
    validate_draws,
)


def checks(report, level: str | None = None) -> set[str]:
    return {i.check for i in report.issues if level is None or i.level == level}


def first_row(df: pd.DataFrame, juego: str = "baloto") -> int:
    return int(df.index[df["juego"] == juego][0])


def test_valid_history_passes(history: pd.DataFrame) -> None:
    report = validate_draws(history)
    assert report.ok
    assert report.issues == []
    cruce = report.summary["cruce_baloto_revancha"]
    assert cruce["sin_desfases"] and cruce["sorteos_en_ambos"] == 60
    resumen = report.summary["juegos"]["baloto"]
    assert resumen["n_sorteos"] == 60
    assert set(resumen["sorteos_por_dia"]) == {"lunes", "miércoles", "sábado"}


@pytest.mark.parametrize(
    ("column", "value", "check"),
    [
        ("b5", 44, "rango_balotas"),
        ("b1", 0, "rango_balotas"),
        ("superbalota", 17, "rango_superbalota"),
        ("superbalota", 0, "rango_superbalota"),
    ],
)
def test_out_of_range(history: pd.DataFrame, column: str, value: int, check: str) -> None:
    df = history.copy()
    df.loc[first_row(df), column] = value
    report = validate_draws(df)
    assert check in checks(report, "error")
    with pytest.raises(DataValidationError):
        report.raise_for_errors()


def test_repeated_balls(history: pd.DataFrame) -> None:
    df = history.copy()
    i = first_row(df)
    df.loc[i, "b2"] = df.loc[i, "b1"]
    assert "repetidas" in checks(validate_draws(df), "error")


def test_duplicates(history: pd.DataFrame) -> None:
    exact = pd.concat([history, history.iloc[[0]]], ignore_index=True)
    assert "filas_duplicadas" in checks(validate_draws(exact), "error")

    other = history.iloc[[0]].copy()
    other["superbalota"] = other["superbalota"] % 16 + 1  # mismo sorteo, otro resultado
    report = validate_draws(pd.concat([history, other], ignore_index=True))
    assert {"n_sorteo_duplicado", "fecha_duplicada"} <= checks(report, "error")


def test_gap_is_error_and_cross_game_warning(history: pd.DataFrame) -> None:
    df = history.drop(index=first_row(history) + 10)  # un sorteo de Baloto en la mitad
    report = validate_draws(df)
    assert "saltos" in checks(report, "error")
    assert "cruce_solo_revancha" in checks(report, "warning")
    assert report.summary["cruce_baloto_revancha"]["solo_revancha"] == [1010]


def test_dates_must_increase(history: pd.DataFrame) -> None:
    df = history.copy()
    i = first_row(df)
    df.loc[[i, i + 1], "fecha"] = df.loc[[i + 1, i], "fecha"].to_numpy()
    assert "orden_fechas" in checks(validate_draws(df), "error")


def test_off_schedule_is_only_a_warning(history: pd.DataFrame) -> None:
    df = history.copy()
    monday = df[(df["juego"] == "baloto") & (df["fecha"].dt.dayofweek == 0)].index[0]
    df.loc[monday, "fecha"] += pd.Timedelta(days=1)  # martes: sigue en orden
    report = validate_draws(df)
    assert report.ok
    assert {"fuera_de_calendario", "cruce_fecha_distinta"} <= checks(report, "warning")


def test_unknown_game(history: pd.DataFrame) -> None:
    df = history.copy()
    df.loc[0, "juego"] = "miloto"
    assert "juego_desconocido" in checks(validate_draws(df), "error")


def test_find_conflicts(history: pd.DataFrame) -> None:
    new = history.iloc[[3]].copy()
    assert find_conflicts(history, new) == []  # llega idéntico: no es conflicto
    new["b5"] = 43 if new["b5"].iloc[0] != 43 else 42
    issues = find_conflicts(history, canonicalize(new))
    assert len(issues) == 1 and issues[0].check == "conflicto"
    assert issues[0].n_sorteos == (int(new["n_sorteo"].iloc[0]),)
