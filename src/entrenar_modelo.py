"""
entrenar_modelo.py
------------------
Entrena el modelo principal (K-means) y un modelo de comparación (agrupamiento
jerárquico aglomerativo con enlace de Ward).
  Modelo guardado = Pipeline(preprocesador + KMeans)
"""
import joblib
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.pipeline import Pipeline

from src import config as cfg
from src.preprocesamiento import construir_preprocesador


def construir_pipeline(k: int) -> Pipeline:
    return Pipeline([
        ("preprocesador", construir_preprocesador()),
        ("kmeans", KMeans(n_clusters=k, n_init=cfg.N_INICIOS_KMEANS, random_state=cfg.SEMILLA)),
    ])


def entrenar(X, k: int) -> Pipeline:
    """Ajusta el preprocesador y K-means con los datos completos."""
    return construir_pipeline(k).fit(X)


def etiquetas(modelo: Pipeline):
    """Grupo asignado a cada estación del entrenamiento."""
    return modelo.named_steps["kmeans"].labels_


def entrenar_jerarquico(X_proc, k: int):
    """Agrupamiento jerárquico (Ward) usado solo para comparar con K-means."""
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_proc)


def guardar_modelo(modelo, ruta=cfg.ARCHIVO_MODELO) -> None:
    cfg.crear_directorios()
    joblib.dump(modelo, ruta)


def cargar_modelo(ruta=cfg.ARCHIVO_MODELO):
    return joblib.load(ruta)
