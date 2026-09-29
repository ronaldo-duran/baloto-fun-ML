# Experimento educativo. El Baloto es aleatorio; este modelo no predice resultados.
# En Windows sin `make`: usa `mingw32-make <objetivo>` o directamente `uv run baloto-ml <etapa>`.

UV ?= uv
RUN = $(UV) run baloto-ml

.DEFAULT_GOAL := help
.PHONY: help install scrape ingest validate features train evaluate register pipeline significance notebooks test lint

install: ## instala el entorno (uv)
	$(UV) sync

scrape: ## trae de baloto.com los sorteos que falten (a data/incoming/)
	$(RUN) scrape

ingest: ## 1. detecta sorteos nuevos
	$(RUN) ingest

validate: ## 2. valida y actualiza data/processed/draws.csv
	$(RUN) validate

features: ## 3. features sin fuga
	$(RUN) build-features

train: ## 4. entrena los modelos
	$(RUN) train

evaluate: ## 5. walk-forward, Monte Carlo y chequeos de datos
	$(RUN) evaluate

register: ## 6. registra el modelo entrenado
	$(RUN) register

pipeline: ## todas las etapas (no hace nada si no hay sorteos nuevos)
	$(RUN) pipeline

significance: ## permutación e historiales sintéticos (lento, ~1 h)
	$(RUN) significance

notebooks: ## re-ejecuta los notebooks
	$(UV) run python scripts/run_notebooks.py

test: ## tests
	$(UV) run pytest

lint: ## ruff (lint + formato)
	$(UV) run ruff check src tests scripts
	$(UV) run ruff format --check src tests scripts

help: ## muestra esta ayuda
	@$(UV) run python -c "import re,sys; [print(f'  {m[0]:<14} {m[1]}') for m in re.findall(r'^([a-z-]+):.*?## (.*)$$', open('Makefile', encoding='utf-8').read(), re.M)]"
