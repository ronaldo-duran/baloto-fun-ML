"""Historiales sintéticos de sorteos uniformes: para tests y simulaciones Monte Carlo."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd

from baloto_ml.config import BALLS_PER_DRAW, JUEGOS, N_BALLS, N_SUPER
from baloto_ml.data.calendar import DEFAULT_SCHEDULE, DrawSchedule
from baloto_ml.data.schema import BALL_COLS, SUPER_COL, canonicalize


def random_draws(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """`n` sorteos uniformes: (n, 5) balotas distintas y ordenadas en 1..43, (n,) superbalotas."""
    balls = np.argsort(rng.random((n, N_BALLS)), axis=1)[:, :BALLS_PER_DRAW] + 1
    return np.sort(balls, axis=1), rng.integers(1, N_SUPER + 1, size=n)


def draw_dates(n: int, start: date, schedule: DrawSchedule = DEFAULT_SCHEDULE) -> list[date]:
    """Las primeras `n` fechas de sorteo desde `start` (inclusive) según el calendario."""
    d = start if schedule.is_draw_day(start) else schedule.next_draw_date(start)
    out = []
    for _ in range(n):
        out.append(d)
        d = schedule.next_draw_date(d)
    return out


def sticky_draws(n: int, rng: np.random.Generator, effect: float) -> tuple[np.ndarray, np.ndarray]:
    """Lotería con señal PLANTADA: lo que salió en el sorteo anterior pesa (1 + effect).

    Muestreo sin reemplazo proporcional a los pesos (truco Gumbel-top-k). Sirve de control
    positivo: si el pipeline no detectara esta señal, su resultado negativo no valdría nada.
    """
    balls = np.empty((n, BALLS_PER_DRAW), dtype=np.int64)
    sb = np.empty(n, dtype=np.int64)
    prev_b = np.zeros(N_BALLS)
    prev_s = np.zeros(N_SUPER)
    for t in range(n):
        keys = np.log1p(effect * prev_b) + rng.gumbel(size=N_BALLS)
        picked = np.argpartition(-keys, BALLS_PER_DRAW)[:BALLS_PER_DRAW]
        balls[t] = np.sort(picked + 1)
        s = int(np.argmax(np.log1p(effect * prev_s) + rng.gumbel(size=N_SUPER)))
        sb[t] = s + 1
        prev_b = np.zeros(N_BALLS)
        prev_b[picked] = 1
        prev_s = np.zeros(N_SUPER)
        prev_s[s] = 1
    return balls, sb


def synthetic_history(
    n_draws: int,
    rng: np.random.Generator,
    juegos: Sequence[str] = JUEGOS,
    start: date = date(2024, 1, 3),
    start_n: int = 1,
    schedule: DrawSchedule = DEFAULT_SCHEDULE,
) -> pd.DataFrame:
    """Historial uniforme en el esquema común; todos los juegos comparten fechas y numeración."""
    dates = draw_dates(n_draws, start, schedule)
    frames = []
    for juego in juegos:
        balls, sb = random_draws(n_draws, rng)
        df = pd.DataFrame(balls, columns=BALL_COLS)
        df.insert(0, "juego", juego)
        df.insert(0, "n_sorteo", np.arange(start_n, start_n + n_draws))
        df.insert(0, "fecha", pd.to_datetime(dates))
        df[SUPER_COL] = sb
        frames.append(df)
    return canonicalize(pd.concat(frames, ignore_index=True))
