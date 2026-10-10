"""
asignar_cluster.py
------------------
Asigna una estación NUEVA al grupo más cercano usando el modelo entrenado.
"""
import numpy as np
import pandas as pd

from src import config as cfg
from src.entrenar_modelo import cargar_modelo


def asignar_grupo(modelo, estacion: dict) -> dict:
    """Recibe un diccionario con las 11 variables y devuelve el grupo y las distancias a cada centroide."""
    faltantes = set(cfg.VARIABLES) - set(estacion)
    if faltantes:
        raise ValueError(f"Faltan variables en la estación: {sorted(faltantes)}")
    X = pd.DataFrame([estacion])[cfg.VARIABLES]
    grupo = int(modelo.predict(X)[0])
    distancias = modelo.transform(X)[0]
    return {
        "grupo": grupo,
        "distancia_al_centroide": round(float(distancias[grupo]), 3),
        "distancias": {int(g): round(float(d), 3) for g, d in enumerate(distancias)},
    }


EJEMPLOS = {
    "Estación grande de intercambio": dict(
        pasajeros_pico_manana=1450, pasajeros_pico_tarde=1400, pasajeros_valle=950,
        proporcion_fin_semana=0.85, ocupacion_media_pct=80, variabilidad_diaria=0.25,
        tiempo_espera_min=3, porcentaje_transbordos=62, numero_rutas=10,
        distancia_centro_km=8, sensibilidad_lluvia_pct=10),
    "Estación en barrio periférico": dict(
        pasajeros_pico_manana=950, pasajeros_pico_tarde=420, pasajeros_valle=180,
        proporcion_fin_semana=0.75, ocupacion_media_pct=55, variabilidad_diaria=0.30,
        tiempo_espera_min=6, porcentaje_transbordos=10, numero_rutas=3,
        distancia_centro_km=12, sensibilidad_lluvia_pct=8),
}

if __name__ == "__main__":
    modelo = cargar_modelo()
    for nombre, caso in EJEMPLOS.items():
        print(f"{nombre}: {asignar_grupo(modelo, caso)}")
