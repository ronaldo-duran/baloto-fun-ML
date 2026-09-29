# baloto-fun-ML

> **Experimento educativo. El Baloto es aleatorio; este modelo no predice resultados. Juega con responsabilidad.**

Pipeline reproducible de ciencia de datos que busca señal predictiva en los sorteos del Baloto de
Colombia y comunica con honestidad el resultado esperado: **no hay señal explotable**, porque el
sorteo es aleatorio. El valor del proyecto está en la ingeniería (validación correcta, baselines,
reentrenamiento automatizado y un registro de predicciones en vivo que nadie puede maquillar),
no en "ganarle" a la lotería.

## El juego en números

| Concepto | Valor |
|---|---|
| Balotas principales | 5 distintas entre 1 y 43 (conjunto: el orden no importa) |
| Superbalota | 1 entre 1 y 16 (otro bombo) |
| Probabilidad del premio mayor | 1 en C(43,5) x 16 = **15.401.568** |
| Aciertos esperados por azar (5 números) | 5 x 5/43 ≈ **0,581** por sorteo |
| Baloto Revancha | Sorteo independiente con el mismo formato; se analiza por separado |
| Calendario | Miércoles y sábado; **desde el 2 de junio de 2025 (sorteo 2508) también lunes** |

Los números son etiquetas nominales, no cantidades: nunca se promedian balotas ni se hace
regresión sobre su valor.

## Datos

| Fuente | Qué trae | Formato |
|---|---|---|
| `data/raw/resultados_{baloto,revancha}.csv` | Histórico 2021-05-01 -> 2026-05-23 (sorteos 2081-2660) | Original: `Date, C1..C5, SB, #Sorteo` |
| `data/incoming/*.csv` | Sorteos nuevos (manuales o descargados); hoy, 2661-2714 (hasta el 2026-09-26) traídos de baloto.com | Esquema común, con columna `juego` |
| baloto.com (`WebSource`) | Página pública por sorteo | Se guarda en `data/incoming/web_*.csv` |

Estado actual de `data/processed/draws.csv`: **634 sorteos por juego** (2081-2714, del 2021-05-01 al
2026-09-26), sin errores ni avisos, y Baloto/Revancha alineados 1:1 en fecha y numeración.

**Esquema común** (`data/processed/draws.csv`): `fecha, n_sorteo, juego, b1..b5, superbalota`, con
`juego` en {`baloto`, `revancha`} y las balotas ordenadas de menor a mayor.

**Validaciones, por juego:** formato 5/43 + 1/16, balotas sin repetir, rangos, sin filas ni
números de sorteo duplicados, numeración sin saltos, fechas crecientes, fechas dentro del
calendario (aviso) y ningún sorteo ya registrado que llegue con otro resultado. Además se
cruzan Baloto y Revancha: cada sorteo debe existir en ambos juegos con la misma fecha. El
resultado queda en `reports/validation.json`.

### Descarga desde baloto.com, de forma responsable

`WebSource` lee `https://www.baloto.com/resultados-{baloto,revancha}/<n_sorteo>`:

- respeta `robots.txt` (hoy solo prohíbe `/admin-baloto/` y `/api/`; no se usa la API);
- se identifica con un User-Agent honesto que enlaza a este repo, y espera 2,5 s entre peticiones;
- **no intenta eludir bloqueos**: ante 401/403/429 se detiene. Desde IPs de datacenter (Kaggle,
  posiblemente GitHub Actions) el sitio responde 403; en ese caso se usa la fuente manual;
- no sigue redirecciones: un sorteo que aún no existe redirige a `/resultados` (que muestra el
  último sorteo), y seguirla duplicaría ese resultado con otro número;
- valida lo descargado junto con el histórico **antes** de escribir nada en `data/incoming/`.

## ¿Hay algo que aprender? Chequeos de los datos

Antes de entrenar nada se comprueba lo que tendría que fallar para que existiera señal: que las
balotas salgan con la misma frecuencia (uniformidad) y que los sorteos sean independientes. Detalle
y gráficos en [`notebooks/01_exploracion_uniformidad.ipynb`](notebooks/01_exploracion_uniformidad.ipynb);
cifras en `reports/analysis.json`. Son 634 sorteos por juego, y los p-valores Monte Carlo usan
10.000 historiales simulados.

| Uniformidad | Baloto | Revancha |
|---|---|---|
| Balotas: X² (42 gl) | 25,8 | 33,2 |
| p aproximado (χ² de libro) | 0,976 | 0,833 |
| p corregido (X² · 42/38) | 0,944 | 0,704 |
| p Monte Carlo (exacto) | 0,944 | 0,715 |
| Superbalota: X² (15 gl) y p Monte Carlo | 11,2 · p = 0,725 | 14,5 · p = 0,490 |

**Independencia Baloto vs. Revancha** (634 fechas con ambos sorteos):
- comparten en promedio 0,576 balotas por fecha, frente a 0,581 esperadas (hipergeométrica);
  p de permutación = 0,82;
- la superbalota coincide 30 veces, frente a 39,6 esperadas; p binomial = 0,12;
- la mayor |correlación| entre las 43 × 43 parejas de balotas es 0,149; p de permutación = 0,47.

Ninguna prueba rechaza la hipótesis de un sorteo justo e independiente.

![Veces que salió cada balota, con la banda del 95 % esperable por azar](reports/figures/frecuencia_balotas.png)

![Distribución de X² bajo azar: la de libro está corrida a la derecha](reports/figures/chi2_nulo_balotas.png)

![Balotas compartidas por fecha entre Baloto y Revancha, y su prueba de permutación](reports/figures/independencia_baloto_revancha.png)

### El listón: baselines en walk-forward

Todo modelo se evalúa fuera de muestra en la misma ventana: 384 sorteos por juego (del 2331 al
2714, entre el 2023-09-23 y el 2026-09-26). Detalle en `reports/evaluation/<juego>.json`.

| Juego | Baseline | Aciertos top-5 (IC95 %) | Log-loss por balota | Skill vs. constante | Accuracy superbalota |
|---|---|---|---|---|---|
| Baloto | constante | 0,622 [0,551; 0,694] | 0,3594 | 0 | 0,049 |
| Baloto | frecuencia | 0,529 [0,462; 0,595] | 0,3608 | −0,38 % | 0,057 |
| Revancha | constante | 0,599 [0,526; 0,672] | 0,3594 | 0 | 0,068 |
| Revancha | frecuencia | 0,549 [0,483; 0,616] | 0,3606 | −0,32 % | 0,062 |

Por azar se esperan 0,581 aciertos por sorteo (banda del 95 % para 384 sorteos: 0,513-0,650), una
log-loss de 0,3594 por balota y un accuracy de 1/16 = 0,0625 en la superbalota. El "top-5" del
baseline constante es una jugada al azar, con empates rotos con semilla.

## Lecciones sobre validación y baselines

1. **Revisa los supuestos de la prueba de libro.** El χ² clásico supone balotas independientes,
   pero un sorteo no repite números: bajo azar E[X²] = 43 − 5 = 38, no 42. La receta de libro
   infla el p-valor (0,976 contra 0,944 exacto en Baloto). Si hay dudas, simula.
2. **Si el proceso es justo, el baseline constante es imbatible en log-loss.** Cualquier
   desviación del 5/43 cuesta: el baseline de frecuencia histórica empeora la log-loss de forma
   medible (IC95 % del Δ por encima de cero en ambos juegos).
3. **Con 43 números siempre hay una balota "caliente".** El 9 salió 92 veces en Baloto
   (z = 2,26), pero en historiales simulados la mayor desviación es igual o mayor el 68 % de las
   veces. Mirar muchas cosas a la vez exige corregir por comparaciones múltiples.

## Cómo ejecutarlo

Requisitos: [uv](https://docs.astral.sh/uv/) (instala Python 3.12 automáticamente).

```bash
uv sync                      # entorno y dependencias
uv run baloto-ml scrape      # (opcional) trae de baloto.com los sorteos que falten
uv run baloto-ml ingest      # detecta sorteos nuevos en data/raw y data/incoming
uv run baloto-ml validate    # valida y actualiza data/processed/draws.csv
uv run baloto-ml evaluate    # uniformidad, independencia y walk-forward -> reports/
uv run python scripts/run_notebooks.py   # re-ejecuta los notebooks (outputs versionados)
uv run pytest                # tests
```

## Estado

Proyecto en construcción, por fases: datos -> baselines -> modelos y validación -> pipeline ->
registro en vivo -> app -> CI. Hechas: datos, y baselines con chequeos estadísticos.
