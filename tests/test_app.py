"""Prueba de humo de la app: cada página corre sin excepciones con los datos del repo."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from helpers import write_raw_csv

from baloto_ml.config import EvalConfig, Paths, default_paths
from baloto_ml.data.store import write_draws
from baloto_ml.data.synthetic import synthetic_history
from baloto_ml.live.predict import deadline_for
from baloto_ml.models.base import GameData
from baloto_ml.models.registry import ModelRegistry
from baloto_ml.pipeline.orchestrator import run_pipeline
from baloto_ml.pipeline.stages import PipelineConfig

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PAGES = ["views/jugada.py", "views/registro.py", "views/metodo.py"]

pytestmark = pytest.mark.skipif(
    ModelRegistry(default_paths().models).latest_version("baloto") is None,
    reason="sin modelos registrados en el repo",
)


def run_page(page: str):
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    if page != PAGES[0]:
        at.switch_page(page)
        at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_errors(page: str) -> None:
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]
    assert any("Experimento educativo" in w.value for w in at.warning)  # aviso visible


def test_live_page_with_reconciled_predictions(tmp_path, monkeypatch) -> None:
    """La vista con predicciones conciliadas, sobre un proyecto sintético."""
    paths = Paths(tmp_path)
    cfg = PipelineConfig(eval=EvalConfig(warmup=10, min_train=20, refit_every=10), n_sims=100)

    def hist(n):
        return synthetic_history(n, np.random.default_rng(5), start_n=1000)

    def before(n):
        return deadline_for(GameData.from_draws(hist(n), "baloto").next_date) - timedelta(hours=6)

    h = hist(60)
    for juego in ("baloto", "revancha"):
        write_raw_csv(h[h["juego"] == juego], paths.raw_file(juego))
    run_pipeline(paths, cfg, now=before(60))
    new = hist(61)
    write_draws(new[new["n_sorteo"] == 1060], paths.incoming / "nuevo.csv")
    result = run_pipeline(paths, cfg, now=before(61))
    assert result.conciliadas and result.predicciones

    monkeypatch.setenv("BALOTO_ML_ROOT", str(tmp_path))
    sys.modules.pop("common", None)  # que la app relea la ruta del proyecto
    st.cache_data.clear()
    st.cache_resource.clear()
    at = run_page("views/registro.py")
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label == "Sorteos con resultado" for m in at.metric)
    sys.modules.pop("common", None)
    st.cache_data.clear()
    st.cache_resource.clear()


def test_play_page_scores_a_combination() -> None:
    at = run_page(PAGES[0])
    at.multiselect(key="balotas").set_value([1, 2, 3, 4, 5])
    at.selectbox(key="superbalota").set_value(16)
    at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Puntaje del modelo" in labels
    assert "Tu probabilidad real del premio mayor" in labels
    assert any(m.value == "1 en 15.401.568" for m in at.metric)
