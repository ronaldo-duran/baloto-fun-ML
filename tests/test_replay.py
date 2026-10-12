"""Simulación del registro en vivo: mismo cálculo que la conciliación, sin mirar el futuro."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from helpers import write_raw_csv

from baloto_ml.config import EvalConfig, Paths
from baloto_ml.data.store import write_draws
from baloto_ml.data.synthetic import synthetic_history
from baloto_ml.live.log import RECONCILIATION_COLUMNS, read_log
from baloto_ml.live.predict import deadline_for
from baloto_ml.live.replay import REPLAY_COLUMNS, replay_game, summarize_percentiles
from baloto_ml.models.base import GameData
from baloto_ml.pipeline import stages
from baloto_ml.pipeline.orchestrator import run_pipeline
from baloto_ml.pipeline.stages import PipelineConfig

EVAL_CFG = EvalConfig(warmup=10, min_train=20, refit_every=10)
N0 = 60


def history(n: int) -> pd.DataFrame:
    return synthetic_history(n, np.random.default_rng(44), start=date(2025, 5, 3), start_n=1000)


def before_deadline(n_known: int) -> datetime:
    nxt = GameData.from_draws(history(n_known), "baloto").next_date
    return deadline_for(nxt) - timedelta(hours=12)


# La simulación cubre los últimos 5 sorteos conocidos al empezar (y los que lleguen después).
START = pd.Timestamp(GameData.from_draws(history(N0), "baloto").fechas[N0 - 5]).date()
CFG = PipelineConfig(eval=EVAL_CFG, n_sims=100, replay_start=START)


@pytest.fixture
def project(tmp_path) -> Paths:
    paths = Paths(tmp_path)
    h = history(N0)
    for juego in ("baloto", "revancha"):
        write_raw_csv(h[h["juego"] == juego], paths.raw_file(juego))
    result = run_pipeline(paths, CFG, now=before_deadline(N0))
    assert result.simulados == 2 * 5
    return paths


def add_next_draw(paths: Paths) -> None:
    h = history(N0 + 1)
    write_draws(h[h["n_sorteo"] == 1000 + N0], paths.incoming / "nuevo.csv")


def test_replay_reproduces_the_live_record_and_is_incremental(project: Paths) -> None:
    before = project.replay_report.read_text(encoding="utf-8")
    add_next_draw(project)
    result = run_pipeline(project, CFG, now=before_deadline(N0 + 1))
    assert result.simulados == 2  # solo el sorteo nuevo de cada juego
    after = project.replay_report.read_text(encoding="utf-8")
    assert after.startswith(before)

    sim = read_log(project.replay_report, REPLAY_COLUMNS)
    rec = read_log(project.reconciliation_log, RECONCILIATION_COLUMNS)
    both = sim.merge(rec, on=["juego", "n_sorteo"], suffixes=("_sim", "_vivo"))
    assert len(both) == 2  # el sorteo 1060 de cada juego está en los dos
    assert (both["resultado_sim"] == both["resultado_vivo"]).all()
    # Las probabilidades del registro en vivo se guardan con 6 decimales: diferencias mínimas.
    np.testing.assert_allclose(
        both["puntaje_resultado_sim"], both["puntaje_resultado_vivo"], atol=1e-3
    )
    np.testing.assert_allclose(
        both["percentil_resultado_sim"], both["percentil_resultado_vivo"], atol=0.1
    )

    assert stages.replay(project, CFG) == 0  # ya al día: no recalcula nada
    assert project.replay_report.read_text(encoding="utf-8") == after


def test_replay_never_uses_the_future() -> None:
    """La fila del sorteo t es la misma con o sin los sorteos posteriores en los datos."""
    data = GameData.from_draws(history(N0), "revancha")
    full = replay_game(data, START, cfg=EVAL_CFG)
    t = N0 - 3
    cut = replay_game(data.head(t + 1), START, cfg=EVAL_CFG)
    assert cut[-1] == next(r for r in full if r["n_sorteo"] == int(data.n_sorteo[t]))
    assert [r["sorteos_entrenamiento"] for r in full] == [
        t - EVAL_CFG.warmup for t in range(55, 60)
    ]
    assert [r["n_sorteo"] for r in replay_game(data, START, {1056, 1057}, EVAL_CFG)] == [
        1055,
        1058,
        1059,
    ]


def test_summarize_percentiles() -> None:
    s = summarize_percentiles(np.arange(0.5, 100, 1.0))  # 100 percentiles perfectamente parejos
    assert s["n"] == 100
    assert s["media"] == pytest.approx(50)
    assert s["entre_25_y_75"] == 0.5 and s["sobre_90"] == pytest.approx(0.1)
    assert s["deciles"] == [10] * 10
    assert s["p_valor_ks_uniforme"] > 0.99 and s["p_valor_binomial_mitad"] == pytest.approx(1)
    middle = summarize_percentiles(np.full(30, 50.0))  # todo en la mitad: no es azar
    assert middle["p_valor_binomial_mitad"] < 1e-6 and middle["p_valor_ks_uniforme"] < 1e-6
