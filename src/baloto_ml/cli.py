"""CLI `baloto-ml`: cada etapa del pipeline es un subcomando."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from baloto_ml.config import JUEGOS, N_SIMS, Paths, default_paths
from baloto_ml.data.ingest import scrape_to_incoming
from baloto_ml.data.sources import WebSource, WebSourceError
from baloto_ml.data.store import read_draws
from baloto_ml.data.validation import DataValidationError
from baloto_ml.evaluation.runner import run_significance
from baloto_ml.evaluation.significance import DEFAULT_RUNS
from baloto_ml.logging_utils import setup_logging
from baloto_ml.pipeline import stages
from baloto_ml.pipeline.orchestrator import run_pipeline
from baloto_ml.pipeline.stages import PipelineConfig

logger = logging.getLogger("baloto_ml.cli")

DISCLAIMER = (
    "Experimento educativo. El Baloto es aleatorio; este modelo no predice resultados. "
    "Juega con responsabilidad."
)


def _juegos(args: argparse.Namespace, paths: Paths) -> list[str]:
    if args.juego != "todos":
        return [args.juego]
    draws = read_draws(paths.processed_draws)
    return [j for j in JUEGOS if (draws["juego"] == j).any()]


def cmd_scrape(args: argparse.Namespace, paths: Paths) -> int:
    scrape_to_incoming(paths, WebSource(delay_s=args.delay, max_draws_per_game=args.max_draws))
    return 0


def cmd_ingest(args: argparse.Namespace, paths: Paths) -> int:
    stages.ingest(paths, PipelineConfig.from_env(**({"ingest_web": True} if args.web else {})))
    return 0


def cmd_validate(args: argparse.Namespace, paths: Paths) -> int:
    stages.validate(paths)
    return 0


def cmd_build_features(args: argparse.Namespace, paths: Paths) -> int:
    for juego in _juegos(args, paths):
        stages.build_features(paths, juego)
    return 0


def cmd_train(args: argparse.Namespace, paths: Paths) -> int:
    cfg = PipelineConfig.from_env()
    for juego in _juegos(args, paths):
        if not paths.features_file(juego).exists():
            stages.build_features(paths, juego)
        stages.train(paths, juego, cfg)
    return 0


def cmd_evaluate(args: argparse.Namespace, paths: Paths) -> int:
    stages.evaluate(paths, PipelineConfig.from_env(n_sims=args.n_sims))
    return 0


def cmd_register(args: argparse.Namespace, paths: Paths) -> int:
    cfg = PipelineConfig.from_env()
    for juego in _juegos(args, paths):
        stages.register(paths, juego, cfg)
    return 0


def cmd_pipeline(args: argparse.Namespace, paths: Paths) -> int:
    overrides: dict = {}
    if args.web:
        overrides["ingest_web"] = True
    if args.no_ingest:
        overrides["ingest_enabled"] = False
    run_pipeline(paths, PipelineConfig.from_env(**overrides), force=args.force)
    return 0


def cmd_significance(args: argparse.Namespace, paths: Paths) -> int:
    runs = {m: (10, 20) if args.quick else DEFAULT_RUNS[m] for m in args.models}
    run_significance(paths, runs=runs, n_jobs=args.jobs, controls=not args.no_controls)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="baloto-ml", description=DISCLAIMER)
    parser.add_argument(
        "--root", type=Path, default=None, help="raíz del proyecto (por defecto, el repo)"
    )
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING...")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_juego(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--juego", choices=[*JUEGOS, "todos"], default="todos")
        return p

    p = sub.add_parser(
        "scrape",
        help="descarga de baloto.com los sorteos posteriores al último conocido a data/incoming/",
    )
    p.add_argument("--delay", type=float, default=2.5, help="segundos entre peticiones")
    p.add_argument("--max-draws", type=int, default=200, help="máximo de sorteos por juego")
    p.set_defaults(func=cmd_scrape)

    p = sub.add_parser("ingest", help="1. detecta sorteos nuevos en las fuentes")
    p.add_argument("--web", action="store_true", help="intenta antes la fuente web")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("validate", help="2. valida y actualiza data/processed/draws.csv")
    p.set_defaults(func=cmd_validate)

    p = with_juego(sub.add_parser("build-features", help="3. features sin fuga por juego"))
    p.set_defaults(func=cmd_build_features)

    p = with_juego(sub.add_parser("train", help="4. entrena los modelos (a models/_staging)"))
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="5. walk-forward, Monte Carlo y chequeos de datos")
    p.add_argument("--n-sims", type=int, default=N_SIMS, help="simulaciones Monte Carlo")
    p.set_defaults(func=cmd_evaluate)

    p = with_juego(sub.add_parser("register", help="6. registra el modelo entrenado"))
    p.set_defaults(func=cmd_register)

    p = sub.add_parser(
        "pipeline", help="todas las etapas; solo reentrena si hay sorteos nuevos validados"
    )
    p.add_argument("--force", action="store_true", help="reentrena aunque no haya datos nuevos")
    p.add_argument("--web", action="store_true", help="intenta traer sorteos de baloto.com")
    p.add_argument("--no-ingest", action="store_true", help="desactiva la ingesta")
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser(
        "significance",
        help="permutación e historiales sintéticos con reentrenamiento (lento, no corre en CI)",
    )
    p.add_argument("--models", nargs="+", default=list(DEFAULT_RUNS), choices=list(DEFAULT_RUNS))
    p.add_argument("--jobs", type=int, default=-1, help="procesos en paralelo (-1: todos)")
    p.add_argument("--quick", action="store_true", help="pocas simulaciones (prueba rápida)")
    p.add_argument("--no-controls", action="store_true", help="omite los controles del método")
    p.set_defaults(func=cmd_significance)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    paths = Paths(args.root.resolve()) if args.root else default_paths()
    try:
        return args.func(args, paths)
    except (DataValidationError, WebSourceError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1
