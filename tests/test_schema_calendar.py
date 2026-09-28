from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from baloto_ml.data.calendar import DEFAULT_SCHEDULE, DrawSchedule, ScheduleSegment
from baloto_ml.data.schema import (
    COLUMNS,
    canonicalize,
    parse_fecha_es,
    read_canonical_csv,
    to_csv_frame,
    weekday_from_name,
)
from baloto_ml.data.store import draws_hash, write_draws


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("23 de Mayo de 2026", date(2026, 5, 23)),
        ("01 de mayo de 2021", date(2021, 5, 1)),
        ("1 de Setiembre de 2024", date(2024, 9, 1)),
        ("  26 de SEPTIEMBRE de 2026 ", date(2026, 9, 26)),
    ],
)
def test_parse_fecha_es(text: str, expected: date) -> None:
    assert parse_fecha_es(text) == expected


@pytest.mark.parametrize("text", ["2026-05-23", "23 de Mayoo de 2026", "31 de Febrero de 2026", ""])
def test_parse_fecha_es_rejects_invalid(text: str) -> None:
    with pytest.raises(ValueError):
        parse_fecha_es(text)


def test_weekday_from_name() -> None:
    assert weekday_from_name("Sábado") == 5
    assert weekday_from_name("MIERCOLES") == 2
    assert weekday_from_name("feriado") is None


def test_canonicalize_sorts_balls_and_rows() -> None:
    df = pd.DataFrame(
        {
            "fecha": ["2026-05-23", "2026-05-20"],
            "n_sorteo": ["2660", "2659"],
            "juego": ["Baloto ", "baloto"],
            "b1": [38, 1],
            "b2": [23, 28],
            "b3": [30, 32],
            "b4": [27, 33],
            "b5": [29, 35],
            "superbalota": [8, 13],
        }
    )
    out = canonicalize(df)
    assert list(out.columns) == COLUMNS
    assert out["n_sorteo"].tolist() == [2659, 2660]
    assert out["juego"].tolist() == ["baloto", "baloto"]
    assert out.loc[1, ["b1", "b2", "b3", "b4", "b5"]].tolist() == [23, 27, 29, 30, 38]


def test_canonicalize_requires_columns() -> None:
    with pytest.raises(ValueError, match="Faltan columnas"):
        canonicalize(pd.DataFrame({"fecha": ["2026-01-01"]}))


def test_csv_round_trip_is_deterministic(history: pd.DataFrame, tmp_path) -> None:
    path = tmp_path / "draws.csv"
    write_draws(history, path)
    back = read_canonical_csv(path)
    pd.testing.assert_frame_equal(back, history)
    assert b"\r\n" not in path.read_bytes()
    assert to_csv_frame(history)["fecha"].iloc[0] == "2025-05-03"
    # El hash no depende del orden de las filas de entrada.
    assert draws_hash(history.sample(frac=1, random_state=0)) == draws_hash(history)


def test_schedule_before_and_after_mondays() -> None:
    s = DEFAULT_SCHEDULE
    assert s.is_draw_day(date(2025, 5, 31))  # sábado
    assert not s.is_draw_day(date(2025, 5, 26))  # lunes, antes del cambio
    assert s.is_draw_day(date(2025, 6, 2))  # primer lunes con sorteo (2508)
    assert not s.is_draw_day(date(2026, 9, 29))  # martes
    assert s.next_draw_date(date(2025, 5, 28)) == date(2025, 5, 31)
    assert s.next_draw_date(date(2025, 5, 31)) == date(2025, 6, 2)
    assert s.next_draw_date(date(2026, 9, 26)) == date(2026, 9, 28)


def test_schedule_requires_chronological_segments() -> None:
    with pytest.raises(ValueError):
        DrawSchedule(
            (
                ScheduleSegment(date(2025, 1, 1), frozenset({2})),
                ScheduleSegment(date(2024, 1, 1), frozenset({5})),
            )
        )
