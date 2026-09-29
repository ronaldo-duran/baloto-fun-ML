"""CLI `baloto-ml`: cada etapa del pipeline es un subcomando."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from baloto_ml.config import N_SIMS, Paths, default_paths
from baloto_ml.data.ingest import run_ingest, run_validate, scrape_to_incoming
from baloto_ml.data.sources import WebSource, WebSourceError
from baloto_ml.data.validation import DataValidationError
from baloto_ml.evaluation.runner import run_evaluate, run_significance
from baloto_ml.evaluation.significance import DEFAULT_RUNS
from baloto_ml.logging_utils import setup_logging

logger = logging.getLogger("baloto_ml.cli")

DISCLAIMER = (
    "Experimento educativo. El Baloto es aleatorio; este modelo no predice resultados. "
    "Juega con responsabilidad."
)


def cmd_scrape(args: argparse.Namespace, paths: Paths) -> int:
    source = WebSource(delay_s=args.delay, max_draws_per_game=args.max_draws)
    scrape_to_incoming(paths, source)
    return 0


def cmd_ingest(args: argparse.Namespace, paths: Paths) -> int:
    run_ingest(paths)
    return 0


def cmd_validate(args: argparse.Namespace, paths: Paths) -> int:
    run_validate(paths)
    return 0


def cmd_evaluate(args: argparse.Namespace, paths: Paths) -> int:
    run_evaluate(paths, n_sims=args.n_sims)
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

    p = sub.add_parser(
        "scrape",
        help="descarga de baloto.com los sorteos posteriores al último conocido a data/incoming/",
    )
    p.add_argument("--delay", type=float, default=2.5, help="segundos entre peticiones")
    p.add_argument("--max-draws", type=int, default=200, help="máximo de sorteos por juego")
    p.set_defaults(func=cmd_scrape)

    p = sub.add_parser("ingest", help="detecta sorteos nuevos en las fuentes locales")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("validate", help="valida y actualiza data/processed/draws.csv")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser(
        "evaluate", help="uniformidad, independencia entre juegos y walk-forward por juego"
    )
    p.add_argument("--n-sims", type=int, default=N_SIMS, help="simulaciones Monte Carlo")
    p.set_defaults(func=cmd_evaluate)

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
    except (DataValidationError, WebSourceError) as exc:
        logger.error("%s", exc)
        return 1
