"""
preprocesamiento.py
-------------------
Carga, validación y preparación de los datos para el agrupamiento.

K-means y el agrupamiento jerárquico usan distancias; por eso las variables DEBEN
estar en la misma escala. Pasos: 1) imputar nulos con la mediana, 2) estandarizar
(media 0, desviación 1).
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as cfg


def cargar_dataset(ruta=cfg.ARCHIVO_DATASET) -> pd.DataFrame:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecute primero: python -m src.generar_dataset")
    return pd.read_csv(ruta, encoding="utf-8")


def validar_dataset(df: pd.DataFrame) -> None:
    """Comprueba columnas requeridas, identificadores únicos y valores numéricos."""
    faltantes = set(cfg.COLUMNAS_DATASET) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en el dataset: {sorted(faltantes)}")
    if df[cfg.COLUMNA_ID].duplicated().any():
        raise ValueError("Hay identificadores de estación duplicados.")
    no_numericas = [c for c in cfg.VARIABLES if not pd.api.types.is_numeric_dtype(df[c])]
    if no_numericas:
        raise ValueError(f"Variables no numéricas: {no_numericas}")


def separar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo las variables usadas por el modelo (sin el identificador)."""
    return df[cfg.VARIABLES].copy()


def construir_preprocesador() -> Pipeline:
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
