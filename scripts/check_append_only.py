"""Verifica que los registros en vivo solo crecieron respecto a una revisión de git.

Uso: uv run python scripts/check_append_only.py [--base HEAD]
Sale con código 1 si alguna fila existente cambió o desapareció. Se corre en CI antes de
hacer commit de un registro actualizado.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from baloto_ml.config import default_paths
from baloto_ml.live.log import is_append_only
from baloto_ml.logging_utils import setup_logging

logger = logging.getLogger("check_append_only")


def committed_text(root: Path, rel: str, base: str) -> str | None:
    out = subprocess.run(
        ["git", "show", f"{base}:{rel}"], cwd=root, capture_output=True, text=True, encoding="utf-8"
    )
    return out.stdout.replace("\r\n", "\n") if out.returncode == 0 else None


def main(argv: list[str]) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD", help="revisión de referencia")
    args = parser.parse_args(argv)
    paths = default_paths()
    ok = True
    for path in (paths.predictions_log, paths.reconciliation_log):
        rel = path.relative_to(paths.root).as_posix()
        old = committed_text(paths.root, rel, args.base)
        if old is None:
            logger.info("%s: no existe en %s (archivo nuevo)", rel, args.base)
            continue
        if not path.exists():
            logger.error("%s: existía en %s y fue borrado", rel, args.base)
            ok = False
            continue
        new = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if is_append_only(old, new):
            added = new[len(old) :].count("\n")
            logger.info("%s: append-only OK (%d filas nuevas)", rel, added)
        else:
            logger.error("%s: se modificaron filas existentes respecto a %s", rel, args.base)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
