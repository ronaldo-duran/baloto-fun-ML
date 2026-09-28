"""Ingesta y validación: de las fuentes a `data/processed/draws.csv`.

Flujo: `scrape` (opcional) deja CSV en data/incoming/ -> `ingest` detecta los sorteos que aún
no están procesados -> `validate` valida histórico + nuevos por juego y, solo si no hay errores,
actualiza el dataset procesado. Todo es idempotente: repetir sin datos nuevos no cambia nada.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from baloto_ml.config import JUEGOS, Paths
from baloto_ml.data.calendar import DEFAULT_SCHEDULE, DrawSchedule
from baloto_ml.data.schema import COLUMNS, KEY, canonicalize, concat_draws
from baloto_ml.data.sources import DataSource, ManualCSVSource, RawHistorySource, WebSource
from baloto_ml.data.store import (
    known_last,
    read_draws,
    write_draws,
    write_incoming,
    write_json,
)
from baloto_ml.data.validation import (
    Issue,
    ValidationReport,
    find_conflicts,
    validate_draws,
)

logger = logging.getLogger(__name__)


def local_sources(paths: Paths) -> list[DataSource]:
    """Fuentes locales: histórico original + CSV manuales de data/incoming/."""
    return [
        RawHistorySource({j: paths.raw_file(j) for j in JUEGOS}),
        ManualCSVSource(paths.incoming),
    ]


def collect(sources: Sequence[DataSource]) -> tuple[pd.DataFrame, list[Issue]]:
    """Une las fuentes, quita duplicados exactos y reporta sorteos contradictorios entre fuentes."""
    frames = []
    for src in sources:
        df = src.fetch()
        logger.info("Fuente %-7s -> %d filas", src.name, len(df))
        frames.append(df)
    df = concat_draws(*frames)
    df = canonicalize(df.drop_duplicates(subset=COLUMNS))
    dup = df[df.duplicated(KEY, keep=False)]
    issues = [
        Issue(
            "error",
            "conflicto_entre_fuentes",
            f"{g['n_sorteo'].nunique()} sorteos aparecen con resultados distintos según la fuente",
            str(juego),
            tuple(int(n) for n in g["n_sorteo"].unique()[:20]),
        )
        for juego, g in dup.groupby("juego")
    ]
    return df, issues


def split_new(existing: pd.DataFrame, collected: pd.DataFrame) -> pd.DataFrame:
    """Filas de `collected` cuyo (juego, n_sorteo) aún no está en `existing`."""
    if existing.empty:
        return collected
    known = pd.MultiIndex.from_frame(existing[KEY])
    mask = ~pd.MultiIndex.from_frame(collected[KEY]).isin(known)
    return canonicalize(collected[mask])


def scrape_to_incoming(paths: Paths, source: WebSource | None = None) -> Path | None:
    """Descarga los sorteos posteriores al último conocido y los guarda en data/incoming/.

    Se valida en contexto (histórico + descargados) ANTES de escribir, para no dejar nunca un
    archivo inválido en la entrada del pipeline.
    """
    local, issues = collect(local_sources(paths))
    local = concat_draws(local, read_draws(paths.processed_draws))
    local = canonicalize(local.drop_duplicates(subset=COLUMNS))
    last = known_last(local)
    logger.info("Último sorteo conocido: %s", last)
    fetched = (source or WebSource()).fetch(last)
    if fetched.empty:
        logger.info("La web no tiene sorteos nuevos")
        return None
    report = validate_draws(concat_draws(local, fetched))
    report.issues[:0] = issues + find_conflicts(local, fetched)
    report.raise_for_errors()
    path = write_incoming(fetched, paths.incoming, prefix="web")
    logger.info("%d filas nuevas guardadas en %s", len(fetched), path)
    return path


def run_ingest(paths: Paths, sources: Sequence[DataSource] | None = None) -> pd.DataFrame:
    """Etapa ingest: fuentes -> data/interim/new_draws.csv con los sorteos aún no procesados."""
    existing = read_draws(paths.processed_draws)
    collected, issues = collect(sources if sources is not None else local_sources(paths))
    issues += find_conflicts(existing, collected)
    ValidationReport(issues).raise_for_errors()
    new = split_new(existing, collected)
    write_draws(new, paths.new_draws)
    for juego, g in new.groupby("juego"):
        logger.info(
            "%s: %d sorteos nuevos (%d..%d)",
            juego,
            len(g),
            g["n_sorteo"].min(),
            g["n_sorteo"].max(),
        )
    if new.empty:
        logger.info("Sin sorteos nuevos")
    return new


def run_validate(paths: Paths, schedule: DrawSchedule = DEFAULT_SCHEDULE) -> ValidationReport:
    """Etapa validate: valida histórico + nuevos; si no hay errores, actualiza el procesado.

    El reporte (reports/validation.json) se escribe siempre, también cuando hay errores.
    """
    existing = read_draws(paths.processed_draws)
    new = read_draws(paths.new_draws)
    combined = concat_draws(existing, new)
    report = validate_draws(combined, schedule)
    report.issues[:0] = find_conflicts(existing, new)
    write_json(report.to_dict(), paths.validation_report)
    for issue in report.warnings:
        logger.warning("[%s] %s: %s", issue.juego or "-", issue.check, issue.message)
    report.raise_for_errors()
    if not new.empty:
        write_draws(combined, paths.processed_draws)
        logger.info("Dataset procesado actualizado: %d filas", len(combined))
    paths.new_draws.unlink(missing_ok=True)
    return report
