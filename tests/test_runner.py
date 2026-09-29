from __future__ import annotations

import pandas as pd

from baloto_ml.config import EvalConfig, Paths
from baloto_ml.data.store import read_json, write_draws
from baloto_ml.evaluation.runner import run_evaluate


def test_run_evaluate_writes_reports_idempotently(paths: Paths, history: pd.DataFrame) -> None:
    write_draws(history, paths.processed_draws)
    cfg = EvalConfig(warmup=10, min_train=20, refit_every=5)
    run_evaluate(paths, n_sims=200, cfg=cfg)

    analysis = read_json(paths.analysis_report)
    assert set(analysis["uniformidad"]) == {"baloto", "revancha"}
    assert "independencia_baloto_revancha" in analysis
    ev = read_json(paths.evaluation_report("baloto"))
    assert ev["ventana"]["n_sorteos_prueba"] == 30
    assert set(ev["modelos"]) == {"constante", "frecuencia"}
    assert ev["modelos"]["constante"]["balotas"]["skill_log_loss"] == 0.0

    before = {p: p.read_bytes() for p in paths.reports.rglob("*.json")}
    run_evaluate(paths, n_sims=200, cfg=cfg)
    assert {p: p.read_bytes() for p in paths.reports.rglob("*.json")} == before
