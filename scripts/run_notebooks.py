"""Ejecuta los notebooks en su lugar (los outputs quedan versionados) con el entorno actual.

Uso: uv run python scripts/run_notebooks.py [notebooks/01_...ipynb ...]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from baloto_ml.logging_utils import setup_logging

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("run_notebooks")


def run(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=3600,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        record_timing=False,  # sin marcas de tiempo: re-ejecutar no ensucia el diff
    )
    client.execute()
    text = nbformat.writes(nb).rstrip("\n") + "\n"
    # LF explícito: en Windows nbformat.write usaría CRLF.
    path.write_text(text, encoding="utf-8", newline="")


def main(argv: list[str]) -> int:
    setup_logging()
    targets = [Path(a).resolve() for a in argv] or sorted((ROOT / "notebooks").glob("*.ipynb"))
    for path in targets:
        logger.info("Ejecutando %s", path.relative_to(ROOT))
        run(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
