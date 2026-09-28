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

## Cómo ejecutarlo

Requisitos: [uv](https://docs.astral.sh/uv/) (instala Python 3.12 automáticamente).

```bash
uv sync                      # entorno y dependencias
uv run baloto-ml scrape      # (opcional) trae de baloto.com los sorteos que falten
uv run baloto-ml ingest      # detecta sorteos nuevos en data/raw y data/incoming
uv run baloto-ml validate    # valida y actualiza data/processed/draws.csv
uv run pytest                # tests
```

## Estado

Proyecto en construcción, por fases: datos -> baselines -> modelos y validación -> pipeline ->
registro en vivo -> app -> CI.
