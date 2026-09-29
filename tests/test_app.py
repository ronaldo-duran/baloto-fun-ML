"""Prueba de humo de la app: cada página corre sin excepciones con los datos del repo."""

from __future__ import annotations

from pathlib import Path

import pytest

from baloto_ml.config import default_paths
from baloto_ml.models.registry import ModelRegistry

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
