"""Validaciones de sorteos: formato 5/43 + 1/16, repetidos, duplicados, saltos y cruce de juegos."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from baloto_ml.config import JUEGOS, N_BALLS, N_SUPER
from baloto_ml.data.calendar import DEFAULT_SCHEDULE, WEEKDAY_NAMES_ES, DrawSchedule
from baloto_ml.data.schema import BALL_COLS, COLUMNS, KEY, SUPER_COL

Level = Literal["error", "warning"]
MAX_LISTED = 20  # sorteos listados como ejemplo en cada problema


@dataclass(frozen=True)
class Issue:
    """Un problema detectado. Los errores bloquean el pipeline; los avisos solo se reportan."""

    level: Level
    check: str
    message: str
    juego: str | None = None
    n_sorteos: tuple[int, ...] = ()


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_errores": len(self.errors),
            "n_avisos": len(self.warnings),
            "problemas": [asdict(i) | {"n_sorteos": list(i.n_sorteos)} for i in self.issues],
            "resumen": self.summary,
        }

    def raise_for_errors(self) -> None:
        if self.errors:
            raise DataValidationError(self)


class DataValidationError(ValueError):
    """La validación encontró errores; `report` trae el detalle."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        lines = [f"[{i.juego or '-'}] {i.check}: {i.message}" for i in report.errors[:10]]
        super().__init__("Validación fallida:\n  " + "\n  ".join(lines))


def _listed(values) -> tuple[int, ...]:
    return tuple(int(v) for v in list(values)[:MAX_LISTED])


def _issue_if(
    mask: pd.Series, frame: pd.DataFrame, level: Level, check: str, message: str, juego: str
) -> list[Issue]:
    if not mask.any():
        return []
    ns = frame.loc[mask, "n_sorteo"]
    return [Issue(level, check, f"{message} ({int(mask.sum())} sorteos)", juego, _listed(ns))]


def validate_game(
    df: pd.DataFrame, juego: str, schedule: DrawSchedule = DEFAULT_SCHEDULE
) -> list[Issue]:
    """Valida los sorteos de UN juego (se asume ya canonizado)."""
    g = df.sort_values("n_sorteo", kind="stable").reset_index(drop=True)
    if g.empty:
        return [Issue("warning", "sin_datos", f"No hay sorteos de {juego}", juego)]
    issues: list[Issue] = []

    nulls = g[COLUMNS].isna().any(axis=1)
    issues += _issue_if(nulls, g, "error", "nulos", "Filas con valores nulos", juego)
    g = g[~nulls].reset_index(drop=True)

    balls = g[BALL_COLS].to_numpy()
    sb = g[SUPER_COL].to_numpy()
    out_of_range = pd.Series(((balls < 1) | (balls > N_BALLS)).any(axis=1))
    issues += _issue_if(
        out_of_range, g, "error", "rango_balotas", f"Balotas fuera de 1..{N_BALLS}", juego
    )
    bad_sb = pd.Series((sb < 1) | (sb > N_SUPER))
    issues += _issue_if(
        bad_sb, g, "error", "rango_superbalota", f"Superbalota fuera de 1..{N_SUPER}", juego
    )
    repeated = pd.Series((np.diff(np.sort(balls, axis=1), axis=1) == 0).any(axis=1))
    issues += _issue_if(repeated, g, "error", "repetidas", "Balotas repetidas en un sorteo", juego)

    dup_rows = g.duplicated(subset=COLUMNS, keep="first")
    issues += _issue_if(dup_rows, g, "error", "filas_duplicadas", "Filas duplicadas", juego)
    dedup = g[~dup_rows].reset_index(drop=True)
    dup_n = dedup["n_sorteo"].duplicated(keep=False)
    issues += _issue_if(
        dup_n, dedup, "error", "n_sorteo_duplicado", "Número de sorteo con dos resultados", juego
    )
    dup_fecha = dedup["fecha"].duplicated(keep=False)
    issues += _issue_if(
        dup_fecha, dedup, "error", "fecha_duplicada", "Dos sorteos en la misma fecha", juego
    )

    unique = dedup.drop_duplicates("n_sorteo").reset_index(drop=True)
    ns = unique["n_sorteo"].to_numpy()
    diffs = np.diff(ns)
    gap_idx = np.nonzero(diffs > 1)[0]
    if len(gap_idx):
        ranges = [
            f"{ns[i] + 1}..{ns[i + 1] - 1}" if ns[i + 1] - ns[i] > 2 else str(ns[i] + 1)
            for i in gap_idx
        ]
        n_missing = int((diffs[gap_idx] - 1).sum())
        issues.append(
            Issue(
                "error",
                "saltos",
                f"Faltan {n_missing} sorteos en la numeración: {', '.join(ranges[:MAX_LISTED])}",
                juego,
                _listed(ns[gap_idx + 1]),
            )
        )

    not_increasing = unique["fecha"].diff().dt.days.fillna(1) <= 0
    issues += _issue_if(
        not_increasing,
        unique,
        "error",
        "orden_fechas",
        "Fecha no posterior a la del sorteo anterior",
        juego,
    )

    off_schedule = ~unique["fecha"].dt.date.map(schedule.is_draw_day).astype(bool)
    if off_schedule.any():
        days = sorted({WEEKDAY_NAMES_ES[d] for d in unique.loc[off_schedule, "fecha"].dt.dayofweek})
        issues += _issue_if(
            off_schedule,
            unique,
            "warning",
            "fuera_de_calendario",
            f"Sorteos en días fuera del calendario ({', '.join(days)})",
            juego,
        )
    return issues


def validate_cross_game(df: pd.DataFrame) -> tuple[list[Issue], dict]:
    """Compara numeración y fechas de Baloto y Revancha; los desfases se reportan como avisos."""
    b = df.loc[df["juego"] == "baloto", ["n_sorteo", "fecha"]].drop_duplicates("n_sorteo")
    r = df.loc[df["juego"] == "revancha", ["n_sorteo", "fecha"]].drop_duplicates("n_sorteo")
    if b.empty or r.empty:
        return [], {"comparable": False}
    m = b.merge(r, on="n_sorteo", how="outer", suffixes=("_baloto", "_revancha"), indicator=True)
    only_b = m.loc[m["_merge"] == "left_only", "n_sorteo"].sort_values()
    only_r = m.loc[m["_merge"] == "right_only", "n_sorteo"].sort_values()
    both = m[m["_merge"] == "both"]
    diff_date = both[both["fecha_baloto"] != both["fecha_revancha"]].sort_values("n_sorteo")

    issues: list[Issue] = []
    if len(only_b):
        issues.append(
            Issue(
                "warning",
                "cruce_solo_baloto",
                f"{len(only_b)} sorteos de Baloto sin su Revancha",
                "baloto",
                _listed(only_b),
            )
        )
    if len(only_r):
        issues.append(
            Issue(
                "warning",
                "cruce_solo_revancha",
                f"{len(only_r)} sorteos de Revancha sin su Baloto",
                "revancha",
                _listed(only_r),
            )
        )
    if len(diff_date):
        issues.append(
            Issue(
                "warning",
                "cruce_fecha_distinta",
                f"{len(diff_date)} sorteos con el mismo número pero distinta fecha",
                None,
                _listed(diff_date["n_sorteo"]),
            )
        )
    summary = {
        "comparable": True,
        "sorteos_en_ambos": int(len(both)),
        "solo_baloto": [int(n) for n in only_b],
        "solo_revancha": [int(n) for n in only_r],
        "misma_numeracion_distinta_fecha": [
            {
                "n_sorteo": int(row.n_sorteo),
                "fecha_baloto": row.fecha_baloto.date().isoformat(),
                "fecha_revancha": row.fecha_revancha.date().isoformat(),
            }
            for row in diff_date.itertuples()
        ],
        "sin_desfases": not (len(only_b) or len(only_r) or len(diff_date)),
    }
    return issues, summary


def find_conflicts(existing: pd.DataFrame, new: pd.DataFrame) -> list[Issue]:
    """Sorteos que ya existen (mismo juego y número) pero llegan con otro contenido."""
    if existing.empty or new.empty:
        return []
    value_cols = [c for c in COLUMNS if c not in KEY]
    m = existing.merge(new, on=KEY, suffixes=("_actual", "_nuevo"))
    if m.empty:
        return []
    differs = np.zeros(len(m), dtype=bool)
    for c in value_cols:
        differs |= (m[f"{c}_actual"] != m[f"{c}_nuevo"]).to_numpy()
    issues = []
    for juego, g in m[differs].groupby("juego"):
        issues.append(
            Issue(
                "error",
                "conflicto",
                f"{len(g)} sorteos ya registrados llegan con otro resultado o fecha",
                str(juego),
                _listed(g["n_sorteo"]),
            )
        )
    return issues


def summarize(df: pd.DataFrame) -> dict:
    """Resumen por juego: tamaño, rangos de fecha y numeración, sorteos por año y por día."""
    out: dict[str, dict] = {}
    for juego, g in df.groupby("juego", sort=True):
        years = g["fecha"].dt.year.value_counts().sort_index()
        days = g["fecha"].dt.dayofweek.value_counts().sort_index()
        out[str(juego)] = {
            "n_sorteos": int(len(g)),
            "fecha_min": g["fecha"].min().date().isoformat(),
            "fecha_max": g["fecha"].max().date().isoformat(),
            "n_sorteo_min": int(g["n_sorteo"].min()),
            "n_sorteo_max": int(g["n_sorteo"].max()),
            "sorteos_por_anio": {str(y): int(c) for y, c in years.items()},
            "sorteos_por_dia": {WEEKDAY_NAMES_ES[int(d)]: int(c) for d, c in days.items()},
        }
    return out


def validate_draws(df: pd.DataFrame, schedule: DrawSchedule = DEFAULT_SCHEDULE) -> ValidationReport:
    """Valida cada juego por separado y luego el cruce Baloto/Revancha."""
    report = ValidationReport()
    unknown = sorted(set(df["juego"]) - set(JUEGOS))
    if unknown:
        report.issues.append(
            Issue("error", "juego_desconocido", f"Juegos no reconocidos: {unknown}")
        )
    for juego in JUEGOS:
        report.issues += validate_game(df[df["juego"] == juego], juego, schedule)
    cross_issues, cross_summary = validate_cross_game(df)
    report.issues += cross_issues
    report.summary = {"juegos": summarize(df), "cruce_baloto_revancha": cross_summary}
    return report
