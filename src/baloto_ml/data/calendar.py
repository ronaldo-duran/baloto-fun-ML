"""Calendario de sorteos: qué días de la semana hay sorteo según la época."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta, timezone

# Colombia no tiene horario de verano desde 1993: UTC-5 fijo (evita depender de tzdata en Windows).
COT = timezone(timedelta(hours=-5), "COT")

WEEKDAY_NAMES_ES = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")


@dataclass(frozen=True)
class ScheduleSegment:
    """Desde `start` (inclusive) hay sorteo en `weekdays` (0 = lunes ... 6 = domingo)."""

    start: date
    weekdays: frozenset[int]


@dataclass(frozen=True)
class DrawSchedule:
    """Calendario por tramos; cada tramo rige hasta que empieza el siguiente."""

    segments: tuple[ScheduleSegment, ...]

    def __post_init__(self) -> None:
        starts = [s.start for s in self.segments]
        if not starts or starts != sorted(starts):
            raise ValueError("Los tramos del calendario deben existir y estar en orden cronológico")

    def weekdays_on(self, d: date) -> frozenset[int]:
        current: frozenset[int] = frozenset()
        for seg in self.segments:
            if seg.start <= d:
                current = seg.weekdays
        return current

    def is_draw_day(self, d: date) -> bool:
        return d.weekday() in self.weekdays_on(d)

    def next_draw_date(self, after: date) -> date:
        """Primer día de sorteo estrictamente posterior a `after`."""
        d = after
        for _ in range(14):
            d += timedelta(days=1)
            if self.is_draw_day(d):
                return d
        raise ValueError(f"No hay sorteos en las dos semanas siguientes a {after}")


# Miércoles y sábado; desde el lunes 2025-06-02 (sorteo 2508) también lunes.
DEFAULT_SCHEDULE = DrawSchedule(
    (
        ScheduleSegment(date(2017, 1, 1), frozenset({2, 5})),
        ScheduleSegment(date(2025, 6, 2), frozenset({0, 2, 5})),
    )
)
