"""
main.py
-------
Ejecuta el flujo completo del proyecto de aprendizaje NO supervisado:

    python main.py

Pasos: 1) datos  2) exploración  3) preprocesamiento  4) elección de k
       5) entrenamiento K-means  6) evaluación  7) asignación de ejemplo
"""
import argparse

import pandas as pd

from src import asignar_cluster, config as cfg, evaluar_modelo, explorar_datos, generar_dataset, seleccion_k
from src.entrenar_modelo import entrenar, etiquetas, guardar_modelo
from src.preprocesamiento import (cargar_dataset, construir_preprocesador, separar_variables, validar_dataset)


def main(regenerar: bool = False) -> None:
    print("=== PASO 1: Datos ===")
    if regenerar or not cfg.ARCHIVO_DATASET.exists():
        datos, ref = generar_dataset.generar_dataset()
        generar_dataset.guardar_dataset(datos, ref)
        print("Dataset sintético generado.")
    df = cargar_dataset()
    validar_dataset(df)
    print(f"Estaciones: {len(df)} | Variables: {len(cfg.VARIABLES)}")

    print("\n=== PASO 2: Exploración ===")
    resumen = explorar_datos.ejecutar()
    print("Valores nulos:", resumen["nulos"])

    print("\n=== PASO 3: Preprocesamiento (imputación + estandarización) ===")
    X = separar_variables(df)
    X_proc = construir_preprocesador().fit_transform(X)
    print("Matriz procesada:", X_proc.shape)

    print("\n=== PASO 4: Elección del número de grupos (k) ===")
    tabla_k, k = seleccion_k.ejecutar(X_proc)
    print(tabla_k.to_string(index=False))
    print(f"k elegido (máxima silueta): {k}")

    print("\n=== PASO 5: Entrenamiento de K-means ===")
    modelo = entrenar(X, k)
    guardar_modelo(modelo)
    df_out = df.copy()
    df_out["grupo"] = etiquetas(modelo)
    cfg.crear_directorios()
    df_out.to_csv(cfg.ARCHIVO_ASIGNACION, index=False, encoding="utf-8")
    print("Estaciones por grupo:", df_out["grupo"].value_counts().sort_index().to_dict())

    print("\n=== PASO 6: Evaluación ===")
    referencia = pd.read_csv(cfg.ARCHIVO_REFERENCIA) if cfg.ARCHIVO_REFERENCIA.exists() else None
    m = evaluar_modelo.ejecutar(modelo, df, X, tabla_k, referencia)
    print(f"Silueta: {m['silueta']} | Davies-Bouldin: {m['davies_bouldin']} | Calinski-Harabasz: {m['calinski_harabasz']}")
    print(f"Jerárquico (Ward): silueta {m['jerarquico']['silueta']} | ARI contra K-means: {m['jerarquico']['ari_contra_kmeans']}")
    if "ari_contra_perfil_simulado" in m:
        print(f"Validación externa: ARI {m['ari_contra_perfil_simulado']} | NMI {m['nmi_contra_perfil_simulado']}")

    print("\n=== PASO 7: Asignación de estaciones nuevas ===")
    for nombre, caso in asignar_cluster.EJEMPLOS.items():
        print(f"{nombre}: {asignar_cluster.asignar_grupo(modelo, caso)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agrupamiento de estaciones de transporte masivo")
    parser.add_argument("--regenerar", action="store_true", help="vuelve a crear el dataset sintético")
    main(parser.parse_args().regenerar)
