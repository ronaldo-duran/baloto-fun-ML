from __future__ import annotations

import pandas as pd

from baloto_ml.config import EvalConfig, Paths
from baloto_ml.data.store import read_json, write_draws
from baloto_ml.evaluation.runner import run_evaluate, run_significance


def test_run_evaluate_writes_reports_idempotently(paths: Paths, history: pd.DataFrame) -> None:
    write_draws(history, paths.processed_draws)
    cfg = EvalConfig(warmup=10, min_train=20, refit_every=10)
    run_evaluate(paths, n_sims=200, cfg=cfg)

    analysis = read_json(paths.analysis_report)
    assert set(analysis["uniformidad"]) == {"baloto", "revancha"}
    assert "independencia_baloto_revancha" in analysis
    ev = read_json(paths.evaluation_report("baloto"))
    assert ev["ventana"]["n_sorteos_prueba"] == 30
    assert set(ev["modelos"]) == {"constante", "frecuencia", "logistica", "gradient_boosting"}
    assert ev["modelos"]["constante"]["balotas"]["skill_log_loss"] == 0.0
    assert ev["modelo_produccion"] == "logistica"
    mc = ev["modelos"]["logistica"]["significancia_monte_carlo"]
    assert set(mc) >= {"aciertos_top5", "log_loss_balotas", "n_simulaciones"}
    assert len(ev["modelos"]["logistica"]["percentil_ganadoras"]["percentiles"]) == 30

    before = {p: p.read_bytes() for p in paths.reports.rglob("*.json")}
    run_evaluate(paths, n_sims=200, cfg=cfg)
    assert {p: p.read_bytes() for p in paths.reports.rglob("*.json")} == before


def test_run_significance_quick(paths: Paths, history: pd.DataFrame) -> None:
    write_draws(history, paths.processed_draws)
    cfg = EvalConfig(warmup=10, min_train=20, refit_every=15)
    run_significance(paths, runs={"frecuencia": (3, 3)}, cfg=cfg, n_jobs=1, controls=False)
    rep = read_json(paths.significance_report("revancha"))
    assert rep["modelos"]["frecuencia"]["sintetico"]["n"] == 3
    assert not paths.controls_report.exists()
