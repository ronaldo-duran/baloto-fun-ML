"""Registro en vivo: predicciones antes del sorteo, append-only, y conciliación honesta."""

from __future__ import annotations

import shutil
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from helpers import write_raw_csv

from baloto_ml.config import EXPECTED_HITS, EvalConfig, Paths
from baloto_ml.data.store import read_draws, read_json, write_draws
from baloto_ml.data.synthetic import synthetic_history
from baloto_ml.live.log import (
    BALL_PROB_COLS,
    PREDICTION_COLUMNS,
    RECONCILIATION_COLUMNS,
    is_append_only,
    parse_combination,
    read_log,
)
from baloto_ml.live.predict import deadline_for, next_draw_forecast, predict_next
from baloto_ml.models.base import GameData
from baloto_ml.pipeline import stages
from baloto_ml.pipeline.orchestrator import run_pipeline
from baloto_ml.pipeline.stages import PipelineConfig

CFG = PipelineConfig(eval=EvalConfig(warmup=10, min_train=20, refit_every=10), n_sims=200)
N0 = 60


def history(n: int) -> pd.DataFrame:
    return synthetic_history(n, np.random.default_rng(33), start=date(2025, 5, 3), start_n=1000)


def next_date(n_known: int) -> date:
    return GameData.from_draws(history(n_known), "baloto").next_date


def before_deadline(n_known: int) -> datetime:
    """Un momento seguro para predecir el sorteo siguiente a los primeros `n_known`."""
    return deadline_for(next_date(n_known)) - timedelta(hours=12)


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Paths:
    paths = Paths(tmp_path_factory.mktemp("live"))
    h = history(N0)
    for juego in ("baloto", "revancha"):
        write_raw_csv(h[h["juego"] == juego], paths.raw_file(juego))
    result = run_pipeline(paths, CFG, now=before_deadline(N0))
    assert sorted(result.predicciones) == ["baloto-1060", "revancha-1060"]
    return paths


@pytest.fixture
def live(template: Paths, tmp_path) -> Paths:
    shutil.copytree(template.root, tmp_path / "p")
    return Paths(tmp_path / "p")


def add_next_draw(paths: Paths, k: int = 1) -> pd.DataFrame:
    h = history(N0 + k)
    new = h[h["n_sorteo"] >= 1000 + N0]
    write_draws(new, paths.incoming / f"nuevos_{k}.csv")
    return new


def test_prediction_is_logged_before_the_draw(live: Paths) -> None:
    preds = read_log(live.predictions_log, PREDICTION_COLUMNS)
    assert list(preds.columns) == PREDICTION_COLUMNS
    row = preds[preds["juego"] == "baloto"].iloc[0]
    assert row["n_sorteo"] == 1060
    assert row["fecha_sorteo"] == next_date(N0).isoformat()
    assert row["creada_utc"] < row["hora_limite_utc"]  # ISO: comparable como texto
    probs = row[BALL_PROB_COLS].to_numpy(dtype=float)
    top5 = sorted((np.argsort(-probs)[:5] + 1).tolist())
    assert parse_combination(row["combinacion"]) == top5
    assert 1 <= row["superbalota_sugerida"] <= 16


def test_no_duplicate_and_no_prediction_after_deadline(live: Paths) -> None:
    before = live.predictions_log.read_text(encoding="utf-8")
    assert predict_next(live, "baloto", now=before_deadline(N0)) is None  # ya existe
    deadline = deadline_for(next_date(N0))
    assert predict_next(live, "baloto", now=deadline) is None  # llegó la hora límite
    assert live.predictions_log.read_text(encoding="utf-8") == before


def test_stale_model_is_refused(live: Paths) -> None:
    add_next_draw(live)
    stages.ingest(live, CFG)
    stages.validate(live)  # hay un sorteo más, pero el modelo no se reentrenó
    assert not next_draw_forecast(live, "baloto").up_to_date
    assert predict_next(live, "baloto", now=before_deadline(N0 + 1)) is None


def test_new_result_is_reconciled_and_next_draw_predicted(live: Paths) -> None:
    log_before = live.predictions_log.read_text(encoding="utf-8")
    new = add_next_draw(live)
    result = run_pipeline(live, CFG, now=before_deadline(N0 + 1))
    assert sorted(result.conciliadas) == ["baloto-1060", "revancha-1060"]
    assert sorted(result.predicciones) == ["baloto-1061", "revancha-1061"]

    log_after = live.predictions_log.read_text(encoding="utf-8")
    assert is_append_only(log_before, log_after)  # las predicciones previas no cambiaron

    rec = read_log(live.reconciliation_log, RECONCILIATION_COLUMNS)
    pred = read_log(live.predictions_log, PREDICTION_COLUMNS)
    r = rec[rec["juego"] == "baloto"].iloc[0]
    actual = new[new["juego"] == "baloto"].iloc[0]
    real = [int(actual[c]) for c in ("b1", "b2", "b3", "b4", "b5")]
    suggested = parse_combination(
        pred[pred["prediccion_id"] == "baloto-1060"].iloc[0]["combinacion"]
    )
    assert r["aciertos"] == len(set(real) & set(suggested))
    assert parse_combination(r["resultado"]) == real
    assert r["aciertos_esperados"] == pytest.approx(EXPECTED_HITS, abs=1e-6)

    summary = read_json(live.live_summary)["juegos"]["baloto"]
    assert summary["conciliadas"] == 1 and summary["aciertos"] == r["aciertos"]
    assert summary["aciertos_esperados_azar"] == pytest.approx(EXPECTED_HITS, abs=1e-4)
    assert [p["n_sorteo"] for p in summary["pendientes"]] == [1061]


def test_reconcile_is_idempotent(live: Paths) -> None:
    add_next_draw(live)
    run_pipeline(live, CFG, now=before_deadline(N0 + 1))
    rec_before = live.reconciliation_log.read_text(encoding="utf-8")
    assert stages.log(live, now=datetime.now(UTC)) == []
    assert live.reconciliation_log.read_text(encoding="utf-8") == rec_before


def test_stale_data_never_predicts_the_past(live: Paths) -> None:
    """Si llega un sorteo con retraso (ya se jugó el siguiente), no se predice el pasado."""
    add_next_draw(live)
    late = deadline_for(next_date(N0 + 1)) + timedelta(days=1)
    result = run_pipeline(live, CFG, now=late)
    assert result.predicciones == []  # el 1061 ya pasó: no se registra
    assert sorted(result.conciliadas) == ["baloto-1060", "revancha-1060"]
    assert len(read_draws(live.processed_draws)) == 2 * (N0 + 1)


def test_is_append_only() -> None:
    assert is_append_only("a,b\n1,2\n", "a,b\n1,2\n3,4\n")
    assert not is_append_only("a,b\n1,2\n", "a,b\n1,9\n3,4\n")
    assert not is_append_only("a,b\n1,2\n", "a,b\n")
