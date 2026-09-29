"""App educativa: el Baloto es aleatorio y este proyecto lo demuestra con honestidad.

Local: `uv run streamlit run app/streamlit_app.py` (o `make app`).
Streamlit Community Cloud: archivo principal `app/streamlit_app.py`; dependencias en
`app/requirements.txt`. La app lee datos, modelos y reportes directamente del repositorio.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # para `import common` en las páginas

import common  # noqa: E402  (también agrega src/ al path)
import streamlit as st  # noqa: E402

st.set_page_config(page_title="Baloto: un experimento honesto", page_icon="🎱", layout="wide")

pages = [
    st.Page("views/jugada.py", title="Prueba tu jugada", icon=":material/casino:", default=True),
    st.Page("views/registro.py", title="Registro en vivo", icon=":material/history:"),
    st.Page("views/metodo.py", title="Método y resultados", icon=":material/science:"),
]
navigation = st.navigation(pages, position="top")
st.warning(common.DISCLAIMER, icon=":material/warning:")
navigation.run()
st.divider()
st.caption(
    "Proyecto personal y educativo, sin relación con el operador del Baloto. Si el juego deja de "
    "ser entretenimiento, busca ayuda. [Código en GitHub](" + common.REPO_URL + "). "
    "Datos históricos: [Resultados Baloto](" + common.KAGGLE_DATASET_URL + ") de Javier Forero "
    "(Kaggle, licencia MIT)."
)
