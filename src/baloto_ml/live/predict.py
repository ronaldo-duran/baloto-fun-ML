"""predict_next: registra la predicción del próximo sorteo ANTES de que ocurra.

Solo se registra si (a) hay un modelo vigente entrenado con los datos actuales, (b) todavía no
pasó la hora límite del sorteo (20:00 hora de Colombia del día del sorteo; el sorteo es más
tarde en la noche) y (c) no existe ya una predicción para ese sorteo. Si los datos están
desactualizados, el "próximo" sorteo según el último resultado conocido ya ocurrió: predecirlo
sería predecir el pasado, así que no se registra nada.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, time

import numpy as np
import pandas as pd

from baloto_ml.config import BALLS_PER_DRAW, Juego, Paths
from baloto_ml.data.calendar import COT
from baloto_ml.data.store import draws_hash, read_draws
from baloto_ml.evaluation.metrics import top_k
from baloto_ml.evaluation.stats import rng_for
from baloto_ml.live.log import (
    BALL_PROB_COLS,
    PREDICTION_COLUMNS,
    RECONCILIATION_COLUMNS,
    SUPER_PROB_COLS,
    append_rows,
    format_combination,
    prediction_id,
    read_log,
)
from baloto_ml.models.base import GameData, Probabilities
from baloto_ml.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

PREDICTION_CUTOFF = time(20, 0)  # hora de Colombia del día del sorteo


@dataclass(frozen=True)
class NextDraw:
    juego: str
    n_sorteo: int
    fecha: date
    deadline_utc: datetime
    version: str
    model: str
    probs: Probabilities  # una fila
    up_to_date: bool  # el modelo se entrenó con los datos actuales


def deadline_for(fecha: date) -> datetime:
    return datetime.combine(fecha, PREDICTION_CUTOFF, tzinfo=COT).astimezone(UTC)


def next_draw_forecast(paths: Paths, juego: Juego) -> NextDraw:
    """Probabilidades del modelo vigente para el sorteo siguiente al último conocido."""
    registry = ModelRegistry(paths.models)
    bundle, meta = registry.load(juego)
    draws = read_draws(paths.processed_draws)
    game = draws[draws["juego"] == juego]
    data = GameData.from_draws(draws, juego)
    probs = bundle.predict(data, np.array([len(data)]))  # fila n: el sorteo siguiente
    return NextDraw(
        juego=juego,
        n_sorteo=int(data.n_sorteo[-1]) + 1,
        fecha=data.next_date,
        deadline_utc=deadline_for(data.next_date),
        version=meta["version"],
        model=bundle.production,
        probs=probs,
        up_to_date=meta["datos"]["hash"] == draws_hash(game),
    )


def suggested_combination(nd: NextDraw) -> tuple[list[int], int]:
    """Las 5 balotas y la superbalota más probables según el modelo (empates con semilla)."""
    rng = rng_for("combinacion", nd.juego, nd.n_sorteo)
    balls = top_k(nd.probs.balls, BALLS_PER_DRAW, rng)[0] + 1
    sb = int(top_k(nd.probs.sb, 1, rng)[0, 0]) + 1
    return sorted(int(b) for b in balls), sb


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def predict_next(paths: Paths, juego: Juego, now: datetime | None = None) -> dict | None:
    """Agrega al registro la predicción del próximo sorteo; None si no corresponde."""
    now = now or datetime.now(UTC)
    try:
        nd = next_draw_forecast(paths, juego)
    except FileNotFoundError:
        logger.warning("%s: no hay modelo registrado; no se predice", juego)
        return None
    if not nd.up_to_date:
        logger.warning(
            "%s: el modelo vigente no está al día con los datos; corre el pipeline", juego
        )
        return None
    if now >= nd.deadline_utc:
        logger.warning(
            "%s: el sorteo %d (%s) ya pasó o está por jugarse: datos desactualizados, "
            "no se registra predicción",
            juego,
            nd.n_sorteo,
            nd.fecha,
        )
        return None
    pid = prediction_id(juego, nd.n_sorteo)
    existing = read_log(paths.predictions_log, PREDICTION_COLUMNS)
    if (existing["prediccion_id"] == pid).any():
        logger.info(
            "%s: ya existe la predicción del sorteo %d (no se modifica)", juego, nd.n_sorteo
        )
        return None
    balls, sb = suggested_combination(nd)
    row = {
        "prediccion_id": pid,
        "creada_utc": _utc_iso(now),
        "juego": juego,
        "n_sorteo": nd.n_sorteo,
        "fecha_sorteo": nd.fecha.isoformat(),
        "hora_limite_utc": _utc_iso(nd.deadline_utc),
        "modelo_version": nd.version,
        "modelo": nd.model,
        "combinacion": format_combination(balls),
        "superbalota_sugerida": sb,
        **{c: round(float(p), 6) for c, p in zip(BALL_PROB_COLS, nd.probs.balls[0], strict=True)},
        **{c: round(float(p), 6) for c, p in zip(SUPER_PROB_COLS, nd.probs.sb[0], strict=True)},
    }
    append_rows(paths.predictions_log, [row], PREDICTION_COLUMNS)
    logger.info(
        "%s: predicción registrada para el sorteo %d (%s): %s + %d",
        juego,
        nd.n_sorteo,
        nd.fecha,
        row["combinacion"],
        sb,
    )
    return row


def pending_predictions(paths: Paths) -> pd.DataFrame:
    """Predicciones registradas que aún no tienen resultado conciliado."""
    preds = read_log(paths.predictions_log, PREDICTION_COLUMNS)
    done = read_log(paths.reconciliation_log, RECONCILIATION_COLUMNS)
    return preds[~preds["prediccion_id"].isin(done["prediccion_id"])]
