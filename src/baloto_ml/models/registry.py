"""Registro de modelos en archivos: models/<juego>/<fecha>_<hash>/{model.joblib, metadata.json}.

- El identificador depende SOLO de las entradas (datos, código y configuración): entrenar otra
  vez con lo mismo produce el mismo id y el registro no se duplica (idempotente).
- `models/<juego>/latest` es un archivo de texto con el id vigente: los symlinks no son
  portables en Windows ni sobreviven bien a git.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

import baloto_ml
from baloto_ml.data.store import read_json, write_json, write_text
from baloto_ml.models.base import Forecaster, GameData, Probabilities

logger = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
LATEST_FILE = "latest"


@dataclass
class ModelBundle:
    """Los modelos entrenados de un juego; `production` es el que alimenta el registro en vivo."""

    juego: str
    production: str
    models: dict[str, Forecaster]
    info: dict = field(default_factory=dict)

    def predict(self, data: GameData, idx: np.ndarray, model: str | None = None) -> Probabilities:
        return self.models[model or self.production].predict(data, idx)


def code_hash(package_dir: Path | None = None) -> str:
    """SHA-256 del código del paquete (rutas relativas + contenido con fin de línea LF)."""
    root = package_dir or Path(baloto_ml.__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(path.relative_to(root).as_posix().encode("utf-8"))
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def version_id(last_date: str, inputs: dict) -> str:
    """`<fecha del último sorteo>_<hash de las entradas>`, p. ej. 2026-09-26_1a2b3c4d."""
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{last_date}_{digest[:8]}"


class ModelRegistry:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir

    def game_dir(self, juego: str) -> Path:
        return self.models_dir / juego

    def versions(self, juego: str) -> list[str]:
        d = self.game_dir(juego)
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if (p / METADATA_FILE).exists())

    def latest_version(self, juego: str) -> str | None:
        pointer = self.game_dir(juego) / LATEST_FILE
        if not pointer.exists():
            return None
        version = pointer.read_text(encoding="utf-8").strip()
        return version or None

    def metadata(self, juego: str, version: str | None = None) -> dict | None:
        version = version or self.latest_version(juego)
        if version is None:
            return None
        path = self.game_dir(juego) / version / METADATA_FILE
        return read_json(path) if path.exists() else None

    def load(self, juego: str, version: str | None = None) -> tuple[ModelBundle, dict]:
        version = version or self.latest_version(juego)
        if version is None:
            raise FileNotFoundError(f"No hay modelos registrados para {juego}")
        vdir = self.game_dir(juego) / version
        return joblib.load(vdir / MODEL_FILE), read_json(vdir / METADATA_FILE)

    def register(self, staging: Path, metadata: dict) -> str:
        """Mueve el modelo de `staging` a su versión y apunta `latest` a ella.

        Si la versión ya existe (mismas entradas), no la toca: solo actualiza el puntero.
        """
        juego, version = metadata["juego"], metadata["version"]
        vdir = self.game_dir(juego) / version
        if vdir.exists():
            logger.info("%s: la versión %s ya estaba registrada", juego, version)
        else:
            vdir.mkdir(parents=True)
            shutil.copy2(staging / MODEL_FILE, vdir / MODEL_FILE)
            write_json(metadata, vdir / METADATA_FILE)
            logger.info("%s: registrada la versión %s", juego, version)
        write_text(self.game_dir(juego) / LATEST_FILE, version + "\n")
        return version

    def prune(self, juego: str, keep: int) -> list[str]:
        """Borra del árbol de trabajo las versiones más antiguas (git conserva la historia)."""
        latest = self.latest_version(juego)
        old = [v for v in self.versions(juego) if v != latest][
            : max(len(self.versions(juego)) - keep, 0)
        ]
        for v in old:
            shutil.rmtree(self.game_dir(juego) / v)
            logger.info("%s: versión antigua %s retirada del árbol de trabajo", juego, v)
        return old
