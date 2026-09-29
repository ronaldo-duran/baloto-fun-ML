"""Orquestación: encadena las etapas y aplica la regla de reentrenamiento.

Regla: se reentrena SOLO si hay sorteos nuevos y validados (o si un juego aún no tiene modelo).
Si no hay datos nuevos, el pipeline termina sin hacer nada y sin tocar ningún archivo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from baloto_ml.config import JUEGOS, Paths
from baloto_ml.data.store import read_draws
from baloto_ml.pipeline import stages
from baloto_ml.pipeline.stages import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    nuevos_sorteos: int = 0
    entrenados: dict[str, str] = field(default_factory=dict)  # juego -> versión registrada

    @property
    def nada_que_hacer(self) -> bool:
        return not self.entrenados


def run_pipeline(
    paths: Paths, cfg: PipelineConfig | None = None, force: bool = False
) -> PipelineResult:
    cfg = cfg or PipelineConfig.from_env()
    result = PipelineResult()
    new = stages.ingest(paths, cfg)
    result.nuevos_sorteos = len(new)
    stages.validate(paths)

    draws = read_draws(paths.processed_draws)
    present = [j for j in JUEGOS if (draws["juego"] == j).any()]
    pending = [j for j in present if force or stages.needs_training(paths, j)]
    if not pending:
        logger.info("Sin sorteos nuevos validados: el pipeline termina sin hacer nada")
        return result

    for juego in pending:
        stages.build_features(paths, juego)
        stages.train(paths, juego, cfg)
    stages.evaluate(paths, cfg)
    for juego in pending:
        result.entrenados[juego] = stages.register(paths, juego, cfg)
    logger.info("Pipeline completo: %s", result.entrenados)
    return result
