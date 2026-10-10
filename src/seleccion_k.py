"""
seleccion_k.py
--------------
Ayuda a decidir cuántos grupos (k) usar. Para cada k de 2 a 8 se calculan:
  * Inercia (método del codo): suma de distancias al centroide; menor es mejor.
  * Silueta: qué tan separados están los grupos (de -1 a 1); mayor es mejor.
  * Davies-Bouldin: similitud entre grupos; menor es mejor.
  * Calinski-Harabasz: dispersión entre/dentro de grupos; mayor es mejor.
El k elegido es el que maximiza la silueta.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src import config as cfg

plt.rcParams.update({"font.family": "serif", "font.size": 10})


def evaluar_k(X_proc, rango=cfg.RANGO_K) -> pd.DataFrame:
    """Entrena K-means para cada k y devuelve la tabla de métricas."""
    filas = []
    for k in rango:
        km = KMeans(n_clusters=k, n_init=cfg.N_INICIOS_KMEANS, random_state=cfg.SEMILLA).fit(X_proc)
        filas.append({
            "k": k,
            "inercia": round(float(km.inertia_), 2),
            "silueta": round(float(silhouette_score(X_proc, km.labels_)), 4),
            "davies_bouldin": round(float(davies_bouldin_score(X_proc, km.labels_)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(X_proc, km.labels_)), 1),
        })
    return pd.DataFrame(filas)


def elegir_k(tabla: pd.DataFrame) -> int:
    """Devuelve el k con mayor coeficiente de silueta."""
    return int(tabla.loc[tabla["silueta"].idxmax(), "k"])


def graficar_k(tabla: pd.DataFrame, k_elegido: int) -> None:
    cfg.crear_directorios()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.3))
    a1.plot(tabla["k"], tabla["inercia"], marker="o", color="black")
    a1.set_xlabel("Número de grupos (k)"); a1.set_ylabel("Inercia"); a1.set_title("Método del codo", fontsize=10)
    a2.plot(tabla["k"], tabla["silueta"], marker="s", color="black")
    a2.axvline(k_elegido, linestyle="--", color="gray")
    a2.set_xlabel("Número de grupos (k)"); a2.set_ylabel("Coeficiente de silueta")
    a2.set_title(f"Silueta (máximo en k = {k_elegido})", fontsize=10)
    for ax in (a1, a2):
        ax.set_xticks(list(tabla["k"]))
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    plt.tight_layout(); plt.savefig(cfg.DIR_FIGURAS / "fig_codo_silueta.png", dpi=200); plt.close()


def ejecutar(X_proc):
    tabla = evaluar_k(X_proc)
    k = elegir_k(tabla)
    tabla.to_csv(cfg.ARCHIVO_TABLA_K, index=False)
    graficar_k(tabla, k)
    return tabla, k
