"""Etapas del pipeline. Cada una lee de disco, escribe en disco y se puede correr sola.

ingest -> validate -> build_features -> train -> evaluate -> register -> predict_next -> log
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from baloto_ml import __version__
from baloto_ml.config import EVAL, N_SIMS, SEED, EvalConfig, Juego, Paths
from baloto_ml.data.ingest import run_ingest, run_validate, scrape_to_incoming
from baloto_ml.data.schema import empty_draws
from baloto_ml.data.sources import WebSourceError
from baloto_ml.data.store import draws_hash, read_draws, read_json, write_draws, write_json
from baloto_ml.data.validation import ValidationReport
from baloto_ml.evaluation.runner import run_evaluate
from baloto_ml.features.build import (
    PER_NUMBER_NAMES,
    SHARED_NAMES,
    FeatureSet,
    NumberFeatures,
)
from baloto_ml.live.predict import predict_next as live_predict_next
from baloto_ml.live.reconcile import reconcile, write_live_summary
from baloto_ml.models.base import GameData
from baloto_ml.models.catalog import CLASSIFIERS, PRODUCTION_MODEL
from baloto_ml.models.classifiers import HGB_PARAMS, LOGISTIC_PARAMS
from baloto_ml.models.registry import (
    METADATA_FILE,
    MODEL_FILE,
    ModelBundle,
    ModelRegistry,
    version_id,
)
from baloto_ml.models.registry import code_hash as package_code_hash

logger = logging.getLogger(__name__)


def _flag(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "si", "sí", "on"}


@dataclass(frozen=True)
class PipelineConfig:
    eval: EvalConfig = EVAL
    n_sims: int = N_SIMS
    ingest_enabled: bool = True  # INGEST_ENABLED=false: el pipeline no busca sorteos nuevos
    ingest_web: bool = False  # INGEST_WEB=true intenta traer sorteos de baloto.com antes
    keep_versions: int = 10  # versiones de modelo que se conservan en el árbol de trabajo

    @classmethod
    def from_env(cls, **overrides) -> PipelineConfig:
        env = os.environ
        base = cls(
            ingest_enabled=_flag(env.get("INGEST_ENABLED"), True),
            ingest_web=_flag(env.get("INGEST_WEB"), False),
            keep_versions=int(env.get("KEEP_MODEL_VERSIONS") or 10),
        )
        return replace(base, **overrides)


def game_data(paths: Paths, juego: Juego) -> GameData:
    return GameData.from_draws(read_draws(paths.processed_draws), juego)


def game_data_hash(paths: Paths, juego: Juego) -> str:
    draws = read_draws(paths.processed_draws)
    return draws_hash(draws[draws["juego"] == juego])


# ------------------------------------------------------------------------ 1-2. ingest y validate


def ingest(paths: Paths, cfg: PipelineConfig) -> pd.DataFrame:
    """Fuentes -> data/interim/new_draws.csv (solo los sorteos que aún no están procesados)."""
    if not cfg.ingest_enabled:
        logger.warning("Ingesta desactivada (INGEST_ENABLED=false): no se buscan sorteos nuevos")
        write_draws(empty_draws(), paths.new_draws)
        return empty_draws()
    if cfg.ingest_web:
        try:
            scrape_to_incoming(paths)
        except WebSourceError as exc:
            logger.warning("Fuente web no disponible (%s); se sigue con las fuentes locales", exc)
    return run_ingest(paths)


def validate(paths: Paths) -> ValidationReport:
    """Valida histórico + nuevos y, si no hay errores, actualiza data/processed/draws.csv."""
    return run_validate(paths)


def needs_training(paths: Paths, juego: Juego) -> bool:
    """Hay que entrenar si no hay modelo o si los datos cambiaron desde el vigente."""
    meta = ModelRegistry(paths.models).metadata(juego)
    return meta is None or meta["datos"]["hash"] != game_data_hash(paths, juego)


# ----------------------------------------------------------------------------- 3. build_features


def build_features(paths: Paths, juego: Juego) -> Path:
    """Features de todos los sorteos del juego (y del siguiente) -> data/processed/features/."""
    data = game_data(paths, juego)
    fs = data.features
    path = paths.features_file(juego)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        balls=fs.balls.per_number,
        sb=fs.sb.per_number,
        shared=fs.balls.shared,
        n_sorteo=data.n_sorteo,
        per_number_names=np.array(PER_NUMBER_NAMES),
        shared_names=np.array(SHARED_NAMES),
    )
    logger.info("%s: features de %d sorteos -> %s", juego, len(data), path.name)
    return path


def load_features(path: Path) -> FeatureSet:
    with np.load(path) as z:
        shared = z["shared"]
        return FeatureSet(NumberFeatures(z["balls"], shared), NumberFeatures(z["sb"], shared))


def features_hash(fs: FeatureSet) -> str:
    h = hashlib.sha256()
    for arr in (fs.balls.per_number, fs.sb.per_number, fs.balls.shared):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------ 4. train


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True
        )
    except OSError:
        return None
    return out.stdout.strip() or None


def training_metadata(paths: Paths, data: GameData, fs: FeatureSet, cfg: PipelineConfig) -> dict:
    d_hash = game_data_hash(paths, data.juego)  # type: ignore[arg-type]
    c_hash = package_code_hash()
    hyper = {"logistica": LOGISTIC_PARAMS, "gradient_boosting": HGB_PARAMS}
    inputs = {
        "datos": d_hash,
        "codigo": c_hash,
        "hiperparametros": hyper,
        "warmup": cfg.eval.warmup,
        "semilla": SEED,
    }
    last_date = str(pd.Timestamp(data.fechas[-1]).date())
    return {
        "version": version_id(last_date, inputs),
        "juego": data.juego,
        "modelo_produccion": PRODUCTION_MODEL,
        "modelos": list(CLASSIFIERS),
        "datos": {
            "hash": d_hash,
            "n_sorteos": len(data),
            "primer_sorteo": int(data.n_sorteo[0]),
            "ultimo_sorteo": int(data.n_sorteo[-1]),
            "fecha_inicio": str(pd.Timestamp(data.fechas[0]).date()),
            "fecha_fin": last_date,
        },
        "entrenamiento": {
            "sorteos_objetivo": [int(data.n_sorteo[cfg.eval.warmup]), int(data.n_sorteo[-1])],
            "n_objetivo": len(data) - cfg.eval.warmup,
            "warmup": cfg.eval.warmup,
            "semilla": SEED,
            "hiperparametros": hyper,
        },
        "features": {
            "por_numero": list(PER_NUMBER_NAMES),
            "compartidas": list(SHARED_NAMES),
            "hash": features_hash(fs),
        },
        "codigo": {
            "version_paquete": __version__,
            "hash": c_hash,
            "git_commit": _git_commit(paths.root),
        },
        "entorno": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


def train(paths: Paths, juego: Juego, cfg: PipelineConfig) -> Path:
    """Entrena los clasificadores con todos los sorteos -> models/_staging/<juego>/."""
    data = game_data(paths, juego)
    fs = load_features(paths.features_file(juego))
    if features_hash(fs) != features_hash(data.features):
        raise RuntimeError(f"{juego}: las features en disco no corresponden a los datos actuales")
    idx = np.arange(cfg.eval.warmup, len(data))
    models = {name: factory().fit(data, idx) for name, factory in CLASSIFIERS.items()}
    bundle = ModelBundle(juego, PRODUCTION_MODEL, models, {"features": list(PER_NUMBER_NAMES)})
    staging = paths.staging_dir(juego)
    staging.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, staging / MODEL_FILE, compress=3)
    write_json(training_metadata(paths, data, fs, cfg), staging / METADATA_FILE)
    logger.info("%s: %s entrenados con %d sorteos objetivo", juego, ", ".join(models), len(idx))
    return staging


# ---------------------------------------------------------------------- 5-6. evaluate y register


def evaluate(paths: Paths, cfg: PipelineConfig) -> None:
    """Walk-forward de todos los modelos y chequeos de los datos -> reports/."""
    run_evaluate(paths, n_sims=cfg.n_sims, cfg=cfg.eval)


def summarize_evaluation(ev: dict) -> dict:
    """Lo esencial del reporte de evaluación, para la metadata del modelo."""
    out = {"ventana": ev["ventana"], "modelos": {}}
    for name, m in ev["modelos"].items():
        mc = m.get("significancia_monte_carlo", {})
        out["modelos"][name] = {
            "aciertos_top5": m["balotas"]["aciertos_top5_media"],
            "skill_log_loss_balotas": m["balotas"]["skill_log_loss"],
            "accuracy_superbalota": m["superbalota"]["accuracy"],
            "p_aciertos_monte_carlo": mc.get("aciertos_top5", {}).get("p_valor"),
            "p_log_loss_monte_carlo": mc.get("log_loss_balotas", {}).get("p_valor"),
        }
    return out


def register(paths: Paths, juego: Juego, cfg: PipelineConfig) -> str:
    """models/_staging/<juego>/ -> models/<juego>/<fecha>_<hash>/ y actualiza `latest`."""
    staging = paths.staging_dir(juego)
    meta = read_json(staging / METADATA_FILE)
    ev_path = paths.evaluation_report(juego)
    if ev_path.exists():
        ev = read_json(ev_path)
        if ev["ventana"]["ultimo_sorteo_prueba"] == meta["datos"]["ultimo_sorteo"]:
            meta["metricas_walk_forward"] = summarize_evaluation(ev)
        else:
            logger.warning("%s: el reporte de evaluación no corresponde a estos datos", juego)
    registry = ModelRegistry(paths.models)
    version = registry.register(staging, meta)
    registry.prune(juego, cfg.keep_versions)
    shutil.rmtree(staging)
    return version


# ------------------------------------------------------------------------ 7-8. predict_next y log


def predict_next(paths: Paths, juego: Juego, now: datetime | None = None) -> dict | None:
    """Registra (append-only) la predicción del próximo sorteo, si aún no ha pasado."""
    return live_predict_next(paths, juego, now)


def log(paths: Paths, now: datetime | None = None) -> list[dict]:
    """Concilia las predicciones con los resultados nuevos y actualiza el resumen en vivo."""
    rows = reconcile(paths, now)
    write_live_summary(paths)
    return rows
