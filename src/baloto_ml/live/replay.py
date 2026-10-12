"""replay: el registro en vivo, repetido sobre los sorteos pasados.

¿Dónde habría caído cada combinación ganadora según el modelo? Para cada sorteo desde
`REPLAY_START` se reentrena el modelo de producción con todos los sorteos anteriores (lo mismo
que hace el pipeline antes de cada predicción real) y se ubica la combinación que salió: su
puntaje y su percentil exacto entre las 15.401.568 jugadas posibles. En los sorteos que también
están en el registro en vivo da los mismos números que reconciliation_log.csv.

Es incremental: un sorteo pasado no cambia, así que las filas ya calculadas no se recalculan y
cada corrida del pipeline solo agrega los sorteos nuevos.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats as sps

from baloto_ml.config import EVAL, JUEGOS, REPLAY_START, EvalConfig, Paths
from baloto_ml.data.store import read_draws
from baloto_ml.evaluation.significance import play_score_percentile
from baloto_ml.live.log import append_rows, format_combination, read_log
from baloto_ml.models.base import Forecaster, GameData
from baloto_ml.models.catalog import MODELS, PRODUCTION_MODEL

logger = logging.getLogger(__name__)

REPLAY_COLUMNS = [
    "juego",
    "n_sorteo",
    "fecha",
    "resultado",
    "superbalota",
    "sorteos_entrenamiento",
    "puntaje_resultado",
    "percentil_resultado",
]


def replay_game(
    data: GameData,
    start: date = REPLAY_START,
    done: set[int] | frozenset[int] = frozenset(),
    cfg: EvalConfig = EVAL,
    factory: Callable[[], Forecaster] = MODELS[PRODUCTION_MODEL],
) -> list[dict]:
    """Una fila por sorteo desde `start` que no esté en `done` (números de sorteo)."""
    first = max(int(np.searchsorted(data.fechas, np.datetime64(start))), cfg.first_test_index)
    rows = []
    for t in range(first, len(data)):
        if int(data.n_sorteo[t]) in done:
            continue
        train_idx = np.arange(cfg.warmup, t)  # estrictamente antes del sorteo, como en producción
        probs = factory().fit(data, train_idx).predict(data, np.array([t]))
        balls = (np.flatnonzero(data.y_balls[t]) + 1).tolist()
        sb = int(np.flatnonzero(data.y_sb[t])[0]) + 1
        score, pct = play_score_percentile(probs.balls[0], probs.sb[0], balls, sb)
        rows.append(
            {
                "juego": data.juego,
                "n_sorteo": int(data.n_sorteo[t]),
                "fecha": pd.Timestamp(data.fechas[t]).date().isoformat(),
                "resultado": format_combination(balls),
                "superbalota": sb,
                "sorteos_entrenamiento": len(train_idx),
                "puntaje_resultado": round(score, 6),
                "percentil_resultado": round(pct, 2),
            }
        )
    return rows


def run_replay(paths: Paths, start: date = REPLAY_START, cfg: EvalConfig = EVAL) -> int:
    """Agrega a reports/simulacion_ganadoras.csv los sorteos que falten; devuelve cuántos."""
    draws = read_draws(paths.processed_draws)
    existing = read_log(paths.replay_report, REPLAY_COLUMNS)
    rows = []
    for juego in JUEGOS:
        if not (draws["juego"] == juego).any():
            continue
        done = set(existing.loc[existing["juego"] == juego, "n_sorteo"].astype(int))
        rows += replay_game(GameData.from_draws(draws, juego), start, done, cfg)
    append_rows(paths.replay_report, rows, REPLAY_COLUMNS)
    if rows:
        logger.info("Simulación: %d sorteos nuevos en %s", len(rows), paths.replay_report)
    return len(rows)


def summarize_percentiles(percentiles: np.ndarray) -> dict:
    """Qué tan lejos del azar quedan los percentiles (0-100) de las combinaciones ganadoras.

    Sin señal son uniformes: media 50, la mitad entre 25 y 75, uno de cada diez por encima de 90.
    """
    u = np.asarray(percentiles, dtype=float) / 100
    n = len(u)
    middle = int(np.sum((u >= 0.25) & (u < 0.75)))
    return {
        "n": n,
        "media": 100 * float(u.mean()) if n else None,
        "entre_25_y_75": middle / n if n else None,
        "sobre_90": float(np.mean(u >= 0.9)) if n else None,
        "deciles": np.histogram(u, bins=10, range=(0, 1))[0].tolist(),
        "p_valor_ks_uniforme": float(sps.kstest(u, "uniform").pvalue) if n else None,
        "p_valor_binomial_mitad": float(sps.binomtest(middle, n, 0.5).pvalue) if n else None,
    }
