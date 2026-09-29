"""El pipeline completo: bootstrap, idempotencia y reentrenamiento solo con datos nuevos."""

from __future__ import annotations

import hashlib
import shutil
from datetime import date

import numpy as np
import pandas as pd
import pytest
from helpers import write_raw_csv

from baloto_ml.config import EvalConfig, Paths
from baloto_ml.data.store import read_draws, read_json, write_draws
from baloto_ml.data.synthetic import synthetic_history
from baloto_ml.models.registry import ModelRegistry
from baloto_ml.pipeline import stages
from baloto_ml.pipeline.orchestrator import run_pipeline
from baloto_ml.pipeline.stages import PipelineConfig

CFG = PipelineConfig(eval=EvalConfig(warmup=10, min_train=20, refit_every=10), n_sims=200)
N0 = 60


def full_history(n: int) -> pd.DataFrame:
    return synthetic_history(n, np.random.default_rng(21), start=date(2025, 5, 3), start_n=1000)


def bootstrap(paths: Paths) -> None:
    h = full_history(N0)
    for juego in ("baloto", "revancha"):
        write_raw_csv(h[h["juego"] == juego], paths.raw_file(juego))


def add_incoming(paths: Paths, k: int = 1) -> None:
    h = full_history(N0 + k)
    write_draws(h[h["n_sorteo"] >= 1000 + N0], paths.incoming / f"manual_{N0}_{k}.csv")


def snapshot(paths: Paths) -> dict[str, str]:
    """Hash de cada archivo del proyecto (excepto los derivados no versionados)."""
    out = {}
    for p in sorted(paths.root.rglob("*")):
        rel = p.relative_to(paths.root).as_posix()
        if p.is_file() and not rel.startswith(("data/interim", "data/processed/features")):
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Paths:
    """El pipeline se corre una vez; cada test trabaja sobre una copia."""
    paths = Paths(tmp_path_factory.mktemp("pipeline"))
    bootstrap(paths)
    result = run_pipeline(paths, CFG)
    assert set(result.entrenados) == {"baloto", "revancha"}
    return paths


@pytest.fixture
def ran(template: Paths, tmp_path) -> Paths:
    shutil.copytree(template.root, tmp_path / "proyecto")
    return Paths(tmp_path / "proyecto")


def test_bootstrap_trains_and_registers(ran: Paths) -> None:
    reg = ModelRegistry(ran.models)
    for juego in ("baloto", "revancha"):
        version = reg.latest_version(juego)
        assert version is not None and version.startswith("2025-")
        meta = reg.metadata(juego)
        assert meta["datos"]["n_sorteos"] == N0
        assert meta["datos"]["ultimo_sorteo"] == 1000 + N0 - 1
        assert meta["modelo_produccion"] == "logistica"
        assert meta["entrenamiento"]["semilla"] == 42
        assert "metricas_walk_forward" in meta
        assert set(meta["codigo"]) >= {"version_paquete", "hash"}
        bundle, _ = reg.load(juego)
        assert set(bundle.models) == {"logistica", "gradient_boosting"}
    assert ran.evaluation_report("baloto").exists()
    assert not ran.staging_dir("baloto").exists()  # el staging se limpia al registrar


def test_pipeline_is_idempotent(ran: Paths) -> None:
    before = snapshot(ran)
    result = run_pipeline(ran, CFG)
    assert result.nada_que_hacer and result.nuevos_sorteos == 0
    assert snapshot(ran) == before


def test_new_draw_triggers_retraining(ran: Paths) -> None:
    reg = ModelRegistry(ran.models)
    old = {j: reg.latest_version(j) for j in ("baloto", "revancha")}
    add_incoming(ran)
    result = run_pipeline(ran, CFG)
    assert result.nuevos_sorteos == 2
    for juego in ("baloto", "revancha"):
        assert result.entrenados[juego] != old[juego]
        assert reg.latest_version(juego) == result.entrenados[juego]
        assert old[juego] in reg.versions(juego)  # la versión anterior se conserva
        assert reg.metadata(juego)["datos"]["n_sorteos"] == N0 + 1
    assert len(read_draws(ran.processed_draws)) == 2 * (N0 + 1)


def test_ingest_can_be_disabled(ran: Paths) -> None:
    add_incoming(ran)
    before = snapshot(ran)
    result = run_pipeline(ran, PipelineConfig(**{**CFG.__dict__, "ingest_enabled": False}))
    assert result.nada_que_hacer
    assert snapshot(ran) == before


def test_force_retrains_to_the_same_version(ran: Paths) -> None:
    reg = ModelRegistry(ran.models)
    versions = {j: reg.versions(j) for j in ("baloto", "revancha")}
    result = run_pipeline(ran, CFG, force=True)
    for juego in ("baloto", "revancha"):
        assert result.entrenados[juego] == versions[juego][-1]  # mismas entradas, mismo id
        assert reg.versions(juego) == versions[juego]


def test_train_refuses_stale_features(ran: Paths) -> None:
    stages.build_features(ran, "baloto")
    add_incoming(ran)
    stages.ingest(ran, CFG)
    stages.validate(ran)  # ahora los datos tienen un sorteo más que las features en disco
    with pytest.raises(RuntimeError, match="features en disco"):
        stages.train(ran, "baloto", CFG)


def test_needs_training_tracks_data(ran: Paths) -> None:
    assert not stages.needs_training(ran, "baloto")
    add_incoming(ran)
    stages.ingest(ran, CFG)
    stages.validate(ran)
    assert stages.needs_training(ran, "baloto")
    assert read_json(ran.validation_report)["ok"]
