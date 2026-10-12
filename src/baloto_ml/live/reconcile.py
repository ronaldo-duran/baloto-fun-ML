"""log (reconcile): compara cada predicción registrada con el resultado real y acumula los
aciertos frente a lo esperado por azar. También es append-only.

Además de los aciertos de la combinación sugerida, guarda dónde cayó la combinación GANADORA
según el modelo antes del sorteo: su puntaje y su percentil exacto entre las 15.401.568 jugadas
posibles. Si el modelo no sabe nada, ese percentil es un número al azar entre 0 y 100."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from baloto_ml.config import EXPECTED_HITS, JUEGOS, N_SUPER, P_BALL, P_SUPER, Paths
from baloto_ml.data.schema import BALL_COLS, SUPER_COL
from baloto_ml.data.store import read_draws, write_json
from baloto_ml.evaluation.metrics import EPS, LOG_LOSS_BALL_CONST, LOG_LOSS_SB_CONST
from baloto_ml.evaluation.significance import play_score_percentile
from baloto_ml.evaluation.stats import hits_variance
from baloto_ml.live.log import (
    BALL_PROB_COLS,
    PREDICTION_COLUMNS,
    RECONCILIATION_COLUMNS,
    RESULT_SCORE_COLUMNS,
    SUPER_PROB_COLS,
    add_columns,
    append_rows,
    format_combination,
    parse_combination,
    read_log,
)

logger = logging.getLogger(__name__)


def result_score(pred: pd.Series, balls: list[int], sb: int) -> dict:
    """Puntaje y percentil de la combinación ganadora con las probabilidades registradas."""
    score, pct = play_score_percentile(
        pred[BALL_PROB_COLS].to_numpy(dtype=float),
        pred[SUPER_PROB_COLS].to_numpy(dtype=float),
        balls,
        sb,
    )
    return {"puntaje_resultado": round(score, 6), "percentil_resultado": round(pct, 2)}


def _reconcile_row(pred: pd.Series, actual: pd.Series, now: datetime) -> dict:
    balls = [int(actual[c]) for c in BALL_COLS]
    sb_real = int(actual[SUPER_COL])
    suggested = parse_combination(pred["combinacion"])
    p = np.clip(pred[BALL_PROB_COLS].to_numpy(dtype=float), EPS, 1 - EPS)
    y = np.zeros(len(p))
    y[np.array(balls) - 1] = 1
    q = np.clip(pred[SUPER_PROB_COLS].to_numpy(dtype=float), EPS, 1)
    return {
        "prediccion_id": pred["prediccion_id"],
        "conciliada_utc": now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "juego": pred["juego"],
        "n_sorteo": int(pred["n_sorteo"]),
        "fecha_sorteo": pred["fecha_sorteo"],
        "fecha_real": pd.Timestamp(actual["fecha"]).date().isoformat(),
        "modelo_version": pred["modelo_version"],
        "combinacion": pred["combinacion"],
        "resultado": format_combination(balls),
        "superbalota_sugerida": int(pred["superbalota_sugerida"]),
        "superbalota_real": sb_real,
        "aciertos": len(set(suggested) & set(balls)),
        "acierto_superbalota": int(int(pred["superbalota_sugerida"]) == sb_real),
        "aciertos_esperados": round(EXPECTED_HITS, 6),
        "log_loss_balotas": round(float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()), 6),
        "log_loss_constante": round(LOG_LOSS_BALL_CONST, 6),
        "log_loss_superbalota": round(float(-np.log(q[sb_real - 1] / q.sum())), 6),
        "log_loss_superbalota_constante": round(LOG_LOSS_SB_CONST, 6),
        **result_score(pred, balls, sb_real),
    }


def upgrade_reconciliation_log(paths: Paths) -> int:
    """Completa el puntaje y el percentil del resultado en un registro anterior a esas columnas.

    Se calculan con las probabilidades que ya estaban en predictions_log.csv antes de cada
    sorteo, y solo se agregan celdas al final de cada fila: ninguna existente cambia.
    Devuelve cuántas filas se completaron (0 si el registro ya estaba al día).
    """
    path = paths.reconciliation_log
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    previous = [c for c in RECONCILIATION_COLUMNS if c not in RESULT_SCORE_COLUMNS]
    if header != previous:
        return 0  # al día (o desconocido: read_log lo reportará)
    old = read_log(path, previous)
    preds = read_log(paths.predictions_log, PREDICTION_COLUMNS).set_index("prediccion_id")
    values = []
    for row in old.itertuples():
        scored = result_score(
            preds.loc[row.prediccion_id],
            parse_combination(row.resultado),
            int(row.superbalota_real),
        )
        values.append([scored[c] for c in RESULT_SCORE_COLUMNS])
    add_columns(path, RESULT_SCORE_COLUMNS, values)
    logger.info("%s: puntaje y percentil del resultado completados en %d filas", path, len(old))
    return len(old)


def reconcile(paths: Paths, now: datetime | None = None) -> list[dict]:
    """Agrega una fila por cada predicción cuyo sorteo ya tiene resultado validado."""
    now = now or datetime.now(UTC)
    upgrade_reconciliation_log(paths)
    preds = read_log(paths.predictions_log, PREDICTION_COLUMNS)
    done = set(read_log(paths.reconciliation_log, RECONCILIATION_COLUMNS)["prediccion_id"])
    draws = read_draws(paths.processed_draws).set_index(["juego", "n_sorteo"])
    rows = []
    for _, pred in preds.sort_values(["juego", "n_sorteo"]).iterrows():
        key = (pred["juego"], int(pred["n_sorteo"]))
        if pred["prediccion_id"] in done or key not in draws.index:
            continue
        actual = draws.loc[key]
        row = _reconcile_row(pred, actual, now)
        if row["fecha_real"] != row["fecha_sorteo"]:
            logger.warning(
                "%s: el sorteo %d se jugó el %s y no el %s previsto",
                key[0],
                key[1],
                row["fecha_real"],
                row["fecha_sorteo"],
            )
        rows.append(row)
        logger.info(
            "%s %d: sugerida %s + %d | resultado %s + %d | %d aciertos (azar: %.3f) | "
            "el resultado quedó en el percentil %.0f del modelo",
            key[0],
            key[1],
            row["combinacion"],
            row["superbalota_sugerida"],
            row["resultado"],
            row["superbalota_real"],
            row["aciertos"],
            EXPECTED_HITS,
            row["percentil_resultado"],
        )
    append_rows(paths.reconciliation_log, rows, RECONCILIATION_COLUMNS)
    return rows


def live_summary(paths: Paths) -> dict:
    """Aciertos acumulados del modelo frente al azar, por juego (sin marcas de tiempo)."""
    preds = read_log(paths.predictions_log, PREDICTION_COLUMNS)
    rec = read_log(paths.reconciliation_log, RECONCILIATION_COLUMNS)
    out: dict = {"juegos": {}}
    for juego in JUEGOS:
        p = preds[preds["juego"] == juego]
        r = rec[rec["juego"] == juego].sort_values("n_sorteo")
        n = len(r)
        hits = int(r["aciertos"].sum()) if n else 0
        sb_hits = int(r["acierto_superbalota"].sum()) if n else 0
        sd = math.sqrt(n * hits_variance())
        sd_sb = math.sqrt(n * P_SUPER * (1 - P_SUPER))
        pending = p[~p["prediccion_id"].isin(r["prediccion_id"])].sort_values("n_sorteo")
        out["juegos"][juego] = {
            "predicciones": int(len(p)),
            "conciliadas": n,
            "aciertos": hits,
            "aciertos_esperados_azar": n * EXPECTED_HITS,
            "banda95_aciertos": [n * EXPECTED_HITS - 1.96 * sd, n * EXPECTED_HITS + 1.96 * sd],
            "z_aciertos": (hits - n * EXPECTED_HITS) / sd if n else None,
            "aciertos_superbalota": sb_hits,
            "aciertos_superbalota_esperados": n * P_SUPER,
            "banda95_superbalota": [n * P_SUPER - 1.96 * sd_sb, n * P_SUPER + 1.96 * sd_sb],
            "log_loss_medio": float(r["log_loss_balotas"].mean()) if n else None,
            "log_loss_constante": LOG_LOSS_BALL_CONST,
            "percentil_resultado_medio": float(r["percentil_resultado"].mean()) if n else None,
            "pendientes": [
                {
                    "n_sorteo": int(row.n_sorteo),
                    "fecha_sorteo": row.fecha_sorteo,
                    "combinacion": row.combinacion,
                    "superbalota_sugerida": int(row.superbalota_sugerida),
                    "modelo_version": row.modelo_version,
                }
                for row in pending.itertuples()
            ],
        }
    out["referencias"] = {
        "aciertos_por_sorteo": EXPECTED_HITS,
        "p_superbalota": P_SUPER,
        "p_balota": P_BALL,
        "n_superbalota": N_SUPER,
        "percentil_resultado_medio": 50.0,
    }
    return out


def write_live_summary(paths: Paths) -> dict:
    summary = live_summary(paths)
    write_json(summary, paths.live_summary)
    return summary
