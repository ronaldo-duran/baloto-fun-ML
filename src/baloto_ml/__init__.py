"""baloto_ml: experimento educativo de ML sobre los sorteos del Baloto.

El sorteo es aleatorio; este paquete NO predice resultados. Su valor está en la
ingeniería: validación correcta, baselines, pipeline reproducible y registro honesto.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("baloto-ml")
except PackageNotFoundError:  # p. ej., ejecutado desde el repo sin instalar (Streamlit Cloud)
    __version__ = "0.0.0+local"
