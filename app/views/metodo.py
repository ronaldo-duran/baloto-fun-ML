"""Página 3: método, validación y el resultado negativo, con las cifras de los reportes."""

import common as c
import pandas as pd
import streamlit as st

from baloto_ml.config import EXPECTED_HITS, JACKPOT_ODDS, JUEGOS

NOMBRES = {
    "constante": "constante (5/43)",
    "frecuencia": "frecuencia histórica",
    "logistica": "logística por balota",
    "gradient_boosting": "gradient boosting",
}

st.title("Método y resultados")
st.markdown(
    f"""
Este proyecto busca, con las mejores prácticas de ciencia de datos, una señal que permita
predecir el Baloto. El resultado esperado, y el que se obtiene, es que **no la hay**: el sorteo
es aleatorio. La probabilidad del premio mayor es 1 en {c.es_int(JACKPOT_ODDS)} para cualquier
combinación, y jugando 5 números se esperan {c.es(EXPECTED_HITS)} aciertos por sorteo. El valor
está en cómo se llega a esa conclusión sin engañarse.
"""
)

ev = {j: c.load_report(f"reports/evaluation/{j}.json") for j in JUEGOS}
analysis = c.load_report("reports/analysis.json")
sig = {j: c.load_report(f"reports/significance/{j}.json") for j in JUEGOS}
controles = c.load_report("reports/significance/controles.json")

st.header("El resultado", divider="gray")
if all(ev.values()):
    v = ev["baloto"]["ventana"]
    rows = []
    for j in JUEGOS:
        for name, m in ev[j]["modelos"].items():
            mc = m.get("significancia_monte_carlo", {})
            rows.append(
                {
                    "juego": c.NOMBRE[j],
                    "modelo": NOMBRES[name],
                    "aciertos top-5": m["balotas"]["aciertos_top5_media"],
                    "skill log-loss (%)": 100 * m["balotas"]["skill_log_loss"],
                    "p (aciertos)": mc.get("aciertos_top5", {}).get("p_valor"),
                    "p (log-loss)": mc.get("log_loss_balotas", {}).get("p_valor"),
                }
            )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        column_config={
            "aciertos top-5": st.column_config.NumberColumn(format="%.3f"),
            "skill log-loss (%)": st.column_config.NumberColumn(format="%+.2f"),
            "p (aciertos)": st.column_config.NumberColumn(format="%.2f"),
            "p (log-loss)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        f"Walk-forward sobre {v['n_sorteos_prueba']} sorteos por juego "
        f"(del {v['fecha_inicio_prueba']} al {v['fecha_fin_prueba']}). "
        f"Azar: {c.es(EXPECTED_HITS)} aciertos. Skill: mejora de la log-loss frente al baseline "
        "constante (negativo = peor). p: probabilidad de un resultado "
        "igual o mejor por puro azar (Monte Carlo, 10 000 simulaciones)."
    )
st.image(str(c.PATHS.figures / "modelos_walk_forward.png"), width="stretch")

st.header("Cómo se evaluó", divider="gray")
st.markdown(
    """
- **Datos**: los sorteos de Baloto y Revancha desde 2021, validados (formato 5/43 + 1/16, sin
  repetidos, sin saltos) y analizados por separado.
- **Features sin fuga de información**: para cada número, su frecuencia en los últimos 10, 30 y
  100 sorteos, los sorteos desde su última aparición, una tendencia y el día de la semana, siempre
  calculados con sorteos **anteriores** al que se predice (hay un test que lo verifica).
- **Modelos**: dos baselines (probabilidad constante y frecuencia histórica), una regresión
  logística por balota y un gradient boosting compartido, con hiperparámetros **fijados antes**
  de ver resultados.
- **Validación walk-forward**: se entrena solo con el pasado y se predice el sorteo siguiente,
  como en la vida real. El split aleatorio está prohibido.
- **Significancia**: Monte Carlo sobre los resultados, percentil de las combinaciones ganadoras,
  permutación e historiales sintéticos con reentrenamiento.
- **Control positivo**: el mismo método sí detecta una señal plantada en datos sintéticos, así que
  el "no" con los datos reales es informativo.
"""
)

if analysis:
    st.header("¿Hay algo raro en los datos? No", divider="gray")
    u = analysis["uniformidad"]
    ind = analysis.get("independencia_baloto_revancha", {})
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "prueba": f"uniformidad de las balotas ({c.NOMBRE[j]})",
                    "p-valor": u[j]["balotas"]["p_valor_monte_carlo"],
                }
                for j in JUEGOS
            ]
            + [
                {
                    "prueba": f"uniformidad de la superbalota ({c.NOMBRE[j]})",
                    "p-valor": u[j]["superbalota"]["p_valor_monte_carlo"],
                }
                for j in JUEGOS
            ]
            + (
                [
                    {
                        "prueba": "independencia Baloto / Revancha (balotas en común)",
                        "p-valor": ind["balotas_en_comun"]["p_valor_permutacion"],
                    }
                ]
                if ind
                else []
            )
        ),
        hide_index=True,
        column_config={"p-valor": st.column_config.NumberColumn(format="%.3f")},
    )
    st.image(str(c.PATHS.figures / "frecuencia_balotas.png"), width="stretch")

if controles:
    st.header("¿El método vería una señal si la hubiera? Sí", divider="gray")
    pc = pd.DataFrame(controles["control_positivo"])
    pc["modelo"] = pc["modelo"].map(NOMBRES)
    st.dataframe(
        pc.pivot(index="modelo", columns="efecto", values="aciertos_top5").rename(
            columns={0.0: "sin señal", 0.5: "señal débil", 1.5: "señal fuerte"}
        ),
        column_config={
            k: st.column_config.NumberColumn(format="%.3f")
            for k in ("sin señal", "señal débil", "señal fuerte")
        },
    )
    st.caption(
        "Aciertos por sorteo en historiales sintéticos con una señal plantada (lo que salió en el "
        "sorteo anterior pesa más). El gradient boosting la detecta; con los datos reales, "
        "no ve nada."
    )
    st.image(str(c.PATHS.figures / "control_positivo.png"), width="stretch")

st.header('La trampa: cómo "predecir" la lotería sin darse cuenta', divider="gray")
st.markdown(
    "Con una sola línea mal escrita (una ventana móvil que incluye el sorteo que se quiere "
    'predecir) y un split aleatorio, un modelo "acierta" más del doble que el azar. Es falso: '
    "la feature ya contiene la respuesta. Por eso las features se prueban con tests y el modelo se "
    "juzga con un registro en vivo."
)
st.image(str(c.PATHS.figures / "trampa_fuga.png"), width="stretch")

st.header("Lecciones", divider="gray")
st.markdown(
    """
1. **El baseline correcto puede ser imbatible.** Si el proceso es justo, la probabilidad
   constante es el mejor pronóstico posible; todo lo demás añade ruido.
2. **Una fuga de información fabrica señal**, y el walk-forward no la detecta: la detectan los
   tests y el registro en vivo.
3. **Probar muchas cosas garantiza encontrar "algo"**: las decisiones se fijan antes de mirar.
4. **Un resultado negativo necesita un control positivo**: si el método no ve una señal plantada,
   su "no" no prueba nada.
5. **Las probabilidades imposibles son la prueba de cordura más barata**: un 0,99 para una
   balota es un defecto, no una predicción audaz.
"""
)
st.markdown(
    f"Código, datos, notebooks y la historia completa del registro en vivo: "
    f"[{c.REPO_URL.removeprefix('https://')}]({c.REPO_URL})."
)
