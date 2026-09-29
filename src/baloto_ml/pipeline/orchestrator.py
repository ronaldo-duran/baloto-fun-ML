"""Orquestación: encadena las etapas y aplica la regla de reentrenamiento.

Regla: se reentrena SOLO si hay sorteos nuevos y validados (o si un juego aún no tiene modelo).
Si no hay datos nuevos, el pipeline termina sin hacer nada y sin tocar ningún archivo.
Tras registrar: predict_next (predicción del próximo sorteo, antes de que ocurra) y log
(conciliación de las predicciones anteriores con los resultados que acaban de llegar).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from baloto_ml.config import JUEGOS, Paths
from baloto_ml.data.store import read_draws
from baloto_ml.pipeline import stages
from baloto_ml.pipeline.stages import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    nuevos_sorteos: int = 0
    entrenados: dict[str, str] = field(default_factory=dict)  # juego -> versión registrada
    predicciones: list[str] = field(default_factory=list)  # ids registrados en esta corrida
    conciliadas: list[str] = field(default_factory=list)

    @property
    def nada_que_hacer(self) -> bool:
        return not self.entrenados


def run_pipeline(
    paths: Paths,
    cfg: PipelineConfig | None = None,
    force: bool = False,
    now: datetime | None = None,
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
    for juego in present:
        row = stages.predict_next(paths, juego, now)
        if row is not None:
            result.predicciones.append(row["prediccion_id"])
    result.conciliadas = [r["prediccion_id"] for r in stages.log(paths, now)]
    logger.info(
        "Pipeline completo: modelos %s | predicciones %s | conciliadas %s",
        result.entrenados,
        result.predicciones,
        result.conciliadas,
    )
    return result
