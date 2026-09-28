from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from helpers import FakeSession, site_routes, write_raw_csv

from baloto_ml.config import Paths
from baloto_ml.data.ingest import run_ingest, run_validate, scrape_to_incoming
from baloto_ml.data.schema import concat_draws
from baloto_ml.data.sources import WebSource
from baloto_ml.data.store import read_draws, read_json, write_draws
from baloto_ml.data.synthetic import synthetic_history
from baloto_ml.data.validation import DataValidationError


def bootstrap(paths: Paths, history: pd.DataFrame) -> None:
    for juego in ("baloto", "revancha"):
        write_raw_csv(history[history["juego"] == juego], paths.raw_file(juego))


def next_draws(history: pd.DataFrame, k: int = 1, skip: int = 0) -> pd.DataFrame:
    """Los `k` sorteos siguientes (ambos juegos), opcionalmente saltándose `skip`."""
    full = synthetic_history(
        60 + skip + k, np.random.default_rng(7), start=date(2025, 5, 3), start_n=1000
    )
    return full[full["n_sorteo"] >= 1060 + skip].reset_index(drop=True)


def test_bootstrap_then_idempotent(paths: Paths, history: pd.DataFrame) -> None:
    bootstrap(paths, history)
    assert len(run_ingest(paths)) == 120
    assert run_validate(paths).ok
    pd.testing.assert_frame_equal(read_draws(paths.processed_draws), history)
    assert not paths.new_draws.exists()

    processed = paths.processed_draws.read_bytes()
    report = paths.validation_report.read_bytes()
    assert run_ingest(paths).empty  # segunda vuelta: nada nuevo
    run_validate(paths)
    assert paths.processed_draws.read_bytes() == processed
    assert paths.validation_report.read_bytes() == report


def test_incoming_draws_are_appended(paths: Paths, history: pd.DataFrame) -> None:
    bootstrap(paths, history)
    run_ingest(paths)
    run_validate(paths)
    extra = next_draws(history, k=1)
    write_draws(extra, paths.incoming / "manual_1060.csv")
    assert len(run_ingest(paths)) == 2
    run_validate(paths)
    assert len(read_draws(paths.processed_draws)) == 122


def test_conflicting_incoming_draw_is_rejected(paths: Paths, history: pd.DataFrame) -> None:
    bootstrap(paths, history)
    run_ingest(paths)
    run_validate(paths)
    bad = history.iloc[[5]].copy()
    bad["superbalota"] = bad["superbalota"] % 16 + 1
    write_draws(bad, paths.incoming / "corregido.csv")
    with pytest.raises(DataValidationError, match="conflicto"):
        run_ingest(paths)


def test_invalid_new_draws_do_not_touch_processed(paths: Paths, history: pd.DataFrame) -> None:
    bootstrap(paths, history)
    run_ingest(paths)
    run_validate(paths)
    before = paths.processed_draws.read_bytes()
    write_draws(next_draws(history, k=1, skip=2), paths.incoming / "salto.csv")  # faltan 2
    run_ingest(paths)
    with pytest.raises(DataValidationError, match="saltos"):
        run_validate(paths)
    assert paths.processed_draws.read_bytes() == before
    assert read_json(paths.validation_report)["ok"] is False


def test_scrape_to_incoming(paths: Paths, history: pd.DataFrame) -> None:
    bootstrap(paths, history)
    published = next_draws(history, k=2)
    session = FakeSession(site_routes(concat_draws(history, published)))
    src = WebSource(session=session, sleep=lambda s: None)
    path = scrape_to_incoming(paths, src)
    assert path is not None and path.name.startswith("web_1060-1061_")
    pd.testing.assert_frame_equal(read_draws(path), published)
    # Ya conocidos (están en data/incoming): no se vuelve a escribir nada.
    assert scrape_to_incoming(paths, WebSource(session=session, sleep=lambda s: None)) is None
    assert len(run_ingest(paths)) == 124
