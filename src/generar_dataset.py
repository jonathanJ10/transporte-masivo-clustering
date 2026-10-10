"""
generar_dataset.py
------------------
Genera un dataset SINTÉTICO de estaciones de un sistema de transporte masivo.
Cada fila es una estación descrita por 11 variables de demanda y operación.

Se simulan 5 perfiles de estación (residencial, comercial, universitaria,
industrial y terminal). El perfil NO se incluye en el dataset principal, porque
el aprendizaje no supervisado debe descubrirlo; se guarda aparte, solo para
verificar al final qué tan bien lo recuperó el algoritmo.
"""
import numpy as np
import pandas as pd

from src import config as cfg

# Para cada perfil: cantidad de estaciones y (media, desviación) de cada variable.
PERFILES = {
    "Residencial": (50, {
        "pasajeros_pico_manana": (900, 150), "pasajeros_pico_tarde": (450, 100),
        "pasajeros_valle": (200, 50), "proporcion_fin_semana": (0.75, 0.08),
        "ocupacion_media_pct": (55, 8), "variabilidad_diaria": (0.30, 0.05),
        "tiempo_espera_min": (6, 1.2), "porcentaje_transbordos": (12, 5),
        "numero_rutas": (3, 1), "distancia_centro_km": (11, 2.5),
        "sensibilidad_lluvia_pct": (8, 3)}),
    "Comercial": (45, {
        "pasajeros_pico_manana": (500, 100), "pasajeros_pico_tarde": (650, 120),
        "pasajeros_valle": (600, 100), "proporcion_fin_semana": (0.95, 0.07),
        "ocupacion_media_pct": (50, 8), "variabilidad_diaria": (0.22, 0.04),
        "tiempo_espera_min": (4, 1), "porcentaje_transbordos": (30, 7),
        "numero_rutas": (7, 1.5), "distancia_centro_km": (2, 1),
        "sensibilidad_lluvia_pct": (12, 4)}),
    "Universitaria": (35, {
        "pasajeros_pico_manana": (750, 120), "pasajeros_pico_tarde": (800, 130),
        "pasajeros_valle": (400, 80), "proporcion_fin_semana": (0.35, 0.08),
        "ocupacion_media_pct": (60, 8), "variabilidad_diaria": (0.40, 0.06),
        "tiempo_espera_min": (5, 1), "porcentaje_transbordos": (20, 6),
        "numero_rutas": (4, 1), "distancia_centro_km": (5, 1.5),
        "sensibilidad_lluvia_pct": (15, 4)}),
    "Industrial": (35, {
        "pasajeros_pico_manana": (1100, 150), "pasajeros_pico_tarde": (950, 130),
        "pasajeros_valle": (150, 40), "proporcion_fin_semana": (0.40, 0.08),
        "ocupacion_media_pct": (68, 8), "variabilidad_diaria": (0.18, 0.04),
        "tiempo_espera_min": (7, 1.5), "porcentaje_transbordos": (8, 4),
        "numero_rutas": (2, 0.8), "distancia_centro_km": (14, 3),
        "sensibilidad_lluvia_pct": (6, 3)}),
    "Terminal": (25, {
        "pasajeros_pico_manana": (1500, 200), "pasajeros_pico_tarde": (1450, 200),
        "pasajeros_valle": (1000, 150), "proporcion_fin_semana": (0.85, 0.08),
        "ocupacion_media_pct": (78, 7), "variabilidad_diaria": (0.25, 0.05),
        "tiempo_espera_min": (3, 0.7), "porcentaje_transbordos": (65, 8),
        "numero_rutas": (11, 2), "distancia_centro_km": (9, 4),
        "sensibilidad_lluvia_pct": (10, 4)}),
}


def generar_dataset(semilla: int = cfg.SEMILLA):
    """Devuelve (dataset, referencia): el dataset sin perfil y la tabla id -> perfil."""
    rng = np.random.default_rng(semilla)
    filas, perfiles = [], []
    for nombre, (n, vars_) in PERFILES.items():
        bloque = {v: rng.normal(m, s, n) for v, (m, s) in vars_.items()}
        filas.append(pd.DataFrame(bloque))
        perfiles += [nombre] * n
    df = pd.concat(filas, ignore_index=True)

    # Límites físicos razonables
    df["proporcion_fin_semana"] = df["proporcion_fin_semana"].clip(0.1, 1.3)
    df["ocupacion_media_pct"] = df["ocupacion_media_pct"].clip(10, 100)
    df["porcentaje_transbordos"] = df["porcentaje_transbordos"].clip(0, 100)
    df["variabilidad_diaria"] = df["variabilidad_diaria"].clip(0.05, None)
    df["tiempo_espera_min"] = df["tiempo_espera_min"].clip(1, None)
    df["distancia_centro_km"] = df["distancia_centro_km"].clip(0.2, None)
    df["sensibilidad_lluvia_pct"] = df["sensibilidad_lluvia_pct"].clip(0, None)
    df["numero_rutas"] = df["numero_rutas"].round().clip(1, None).astype(int)
    for c in ("pasajeros_pico_manana", "pasajeros_pico_tarde", "pasajeros_valle"):
        df[c] = df[c].clip(50, None).round(0)
    df = df.round({"proporcion_fin_semana": 2, "ocupacion_media_pct": 1, "variabilidad_diaria": 2,
                   "tiempo_espera_min": 1, "porcentaje_transbordos": 1,
                   "distancia_centro_km": 1, "sensibilidad_lluvia_pct": 1})

    # Se mezclan las estaciones para que el orden no revele el perfil
    orden = rng.permutation(len(df))
    df = df.iloc[orden].reset_index(drop=True)
    perfiles = np.array(perfiles)[orden]
    df.insert(0, cfg.COLUMNA_ID, [f"E{i:03d}" for i in range(1, len(df) + 1)])

    # Valores faltantes simulados (para practicar la imputación)
    df.loc[rng.random(len(df)) < 0.02, "tiempo_espera_min"] = np.nan
    df.loc[rng.random(len(df)) < 0.02, "sensibilidad_lluvia_pct"] = np.nan

    referencia = pd.DataFrame({cfg.COLUMNA_ID: df[cfg.COLUMNA_ID], "perfil_simulado": perfiles})
    return df[cfg.COLUMNAS_DATASET], referencia


def guardar_dataset(df: pd.DataFrame, referencia: pd.DataFrame) -> None:
    cfg.crear_directorios()
    df.to_csv(cfg.ARCHIVO_DATASET, index=False, encoding="utf-8")
    referencia.to_csv(cfg.ARCHIVO_REFERENCIA, index=False, encoding="utf-8")


if __name__ == "__main__":
    datos, ref = generar_dataset()
    guardar_dataset(datos, ref)
    print(f"Dataset generado: {len(datos)} estaciones -> {cfg.ARCHIVO_DATASET}")
    print(ref["perfil_simulado"].value_counts())
