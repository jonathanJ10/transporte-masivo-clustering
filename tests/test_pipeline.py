"""
Pruebas automatizadas del componente de agrupamiento de estaciones.
Ejecutar con:  python -m pytest -v
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src import config as cfg
from src.asignar_cluster import EJEMPLOS, asignar_grupo
from src.entrenar_modelo import cargar_modelo, entrenar, etiquetas, guardar_modelo
from src.generar_dataset import generar_dataset
from src.preprocesamiento import construir_preprocesador, separar_variables, validar_dataset
from src.seleccion_k import elegir_k, evaluar_k


@pytest.fixture(scope="module")
def datos():
    return generar_dataset(semilla=cfg.SEMILLA)


@pytest.fixture(scope="module")
def df(datos):
    return datos[0]


@pytest.fixture(scope="module")
def modelo(df):
    return entrenar(separar_variables(df), k=5)


# ---------- Pruebas de datos ----------
def test_pt01_dataset_tiene_columnas_y_tamano(df):
    assert list(df.columns) == cfg.COLUMNAS_DATASET
    assert len(df) == 190


def test_pt02_identificadores_unicos(df):
    assert df[cfg.COLUMNA_ID].is_unique


def test_pt03_dataset_sin_columna_de_perfil(df):
    assert "perfil_simulado" not in df.columns


def test_pt04_generacion_reproducible():
    a, _ = generar_dataset(semilla=7)
    b, _ = generar_dataset(semilla=7)
    pd.testing.assert_frame_equal(a, b)


def test_pt05_valores_dentro_de_rangos(df):
    assert (df["pasajeros_pico_manana"] > 0).all()
    assert df["porcentaje_transbordos"].between(0, 100).all()
    assert df["ocupacion_media_pct"].between(10, 100).all()
    assert (df["numero_rutas"] >= 1).all()


def test_pt06_existen_nulos_simulados(df):
    assert df["tiempo_espera_min"].isna().any()
    assert df["sensibilidad_lluvia_pct"].isna().any()


def test_pt07_validar_detecta_columna_faltante(df):
    with pytest.raises(ValueError, match="Faltan columnas"):
        validar_dataset(df.drop(columns=["numero_rutas"]))


def test_pt08_validar_detecta_ids_duplicados(df):
    malo = df.copy()
    malo.loc[1, cfg.COLUMNA_ID] = malo.loc[0, cfg.COLUMNA_ID]
    with pytest.raises(ValueError, match="duplicados"):
        validar_dataset(malo)


# ---------- Pruebas de preprocesamiento ----------
def test_pt09_preprocesador_elimina_nulos_y_estandariza(df):
    X_proc = construir_preprocesador().fit_transform(separar_variables(df))
    assert not np.isnan(X_proc).any()
    assert np.allclose(X_proc.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(X_proc.std(axis=0), 1, atol=1e-6)


# ---------- Pruebas del modelo ----------
def test_pt10_seleccion_de_k_en_rango(df):
    X_proc = construir_preprocesador().fit_transform(separar_variables(df))
    tabla = evaluar_k(X_proc)
    assert list(tabla["k"]) == list(cfg.RANGO_K)
    assert elegir_k(tabla) in list(cfg.RANGO_K)
    assert {"inercia", "silueta", "davies_bouldin", "calinski_harabasz"} <= set(tabla.columns)


def test_pt11_kmeans_asigna_todas_las_estaciones_sin_grupos_vacios(df, modelo):
    et = etiquetas(modelo)
    assert len(et) == len(df)
    assert set(et) == set(range(5))


def test_pt12_calidad_de_la_silueta(df, modelo):
    X_proc = modelo.named_steps["preprocesador"].transform(separar_variables(df))
    assert silhouette_score(X_proc, etiquetas(modelo)) > 0.25


def test_pt13_agrupamiento_reproducible(df):
    X = separar_variables(df)
    assert list(etiquetas(entrenar(X, 5))) == list(etiquetas(entrenar(X, 5)))


def test_pt14_recupera_los_perfiles_simulados(datos, modelo):
    ref = datos[1]["perfil_simulado"].values
    assert adjusted_rand_score(ref, etiquetas(modelo)) > 0.8


# ---------- Pruebas de asignación y persistencia ----------
def test_pt15_asignacion_devuelve_grupo_valido(modelo):
    res = asignar_grupo(modelo, EJEMPLOS["Estación grande de intercambio"])
    assert res["grupo"] in range(5)
    assert res["distancia_al_centroide"] >= 0
    assert res["distancia_al_centroide"] == min(res["distancias"].values())


def test_pt16_asignacion_tolera_valores_faltantes(modelo):
    caso = dict(EJEMPLOS["Estación en barrio periférico"], tiempo_espera_min=np.nan)
    assert asignar_grupo(modelo, caso)["grupo"] in range(5)


def test_pt17_asignacion_exige_todas_las_variables(modelo):
    incompleto = {k: v for k, v in EJEMPLOS["Estación en barrio periférico"].items() if k != "numero_rutas"}
    with pytest.raises(ValueError, match="Faltan variables"):
        asignar_grupo(modelo, incompleto)


def test_pt18_guardar_y_cargar_modelo(df, modelo, tmp_path):
    X = separar_variables(df)
    ruta = tmp_path / "m.joblib"
    guardar_modelo(modelo, ruta)
    assert list(cargar_modelo(ruta).predict(X)) == list(modelo.predict(X))
