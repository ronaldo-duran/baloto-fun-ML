"""Chequeos de calidad sobre los datos reales del repositorio (se omiten si no existen)."""

from __future__ import annotations

import pytest

from baloto_ml.config import JUEGOS, default_paths
from baloto_ml.data.sources import RawHistorySource
from baloto_ml.data.store import read_draws
from baloto_ml.data.validation import validate_draws

PATHS = default_paths()


@pytest.mark.skipif(not all(PATHS.raw_file(j).exists() for j in JUEGOS), reason="sin datos crudos")
def test_raw_history_is_valid_and_aligned() -> None:
    df = RawHistorySource({j: PATHS.raw_file(j) for j in JUEGOS}).fetch()
    report = validate_draws(df)
    assert report.ok, report.to_dict()["problemas"]
    assert report.summary["cruce_baloto_revancha"]["sin_desfases"]
    for juego in JUEGOS:
        assert report.summary["juegos"][juego]["n_sorteo_min"] == 2081


@pytest.mark.skipif(not PATHS.processed_draws.exists(), reason="sin dataset procesado")
def test_processed_dataset_is_valid() -> None:
    report = validate_draws(read_draws(PATHS.processed_draws))
    assert report.ok, report.to_dict()["problemas"]
    assert report.summary["cruce_baloto_revancha"]["sin_desfases"]
