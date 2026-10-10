"""
evaluar_modelo.py
-----------------
Evalúa la calidad de los grupos y genera los resultados del proyecto:
  * Métricas internas (silueta, Davies-Bouldin, Calinski-Harabasz).
  * Validación externa con el perfil simulado (ARI y NMI), posible solo porque el
    dataset es sintético.
  * Comparación con el agrupamiento jerárquico.
  * Perfil (promedios) de cada grupo, gráfico PCA, mapa de calor y dendrograma.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_samples, silhouette_score)

from src import config as cfg
from src.entrenar_modelo import entrenar_jerarquico

plt.rcParams.update({"font.family": "serif", "font.size": 10})
ESTILOS = [("#000000", "o"), ("#555555", "s"), ("#999999", "^"), ("#cccccc", "D"), ("#ffffff", "v"),
           ("#333333", "P"), ("#777777", "X"), ("#bbbbbb", "*")]


def calcular_perfiles(df: pd.DataFrame, etiquetas) -> pd.DataFrame:
    """Promedio de cada variable por grupo (en unidades originales)."""
    perfiles = df[cfg.VARIABLES].groupby(etiquetas).mean().round(2)
    perfiles.insert(0, "n_estaciones", pd.Series(etiquetas).value_counts().sort_index().values)
    perfiles.index.name = "grupo"
    return perfiles


def graficar_pca(X_proc, etiquetas, centroides) -> list:
    pca = PCA(n_components=2, random_state=cfg.SEMILLA).fit(X_proc)
    Z, C = pca.transform(X_proc), pca.transform(centroides)
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for g in sorted(set(etiquetas)):
        color, marca = ESTILOS[g % len(ESTILOS)]
        ax.scatter(Z[etiquetas == g, 0], Z[etiquetas == g, 1], c=color, marker=marca,
                   edgecolor="black", s=36, label=f"Grupo {g}")
    ax.scatter(C[:, 0], C[:, 1], c="white", marker="*", edgecolor="black", s=260, linewidth=1.5, label="Centroides")
    ax.set_xlabel(f"Componente 1 ({pca.explained_variance_ratio_[0] * 100:.1f} %)")
    ax.set_ylabel(f"Componente 2 ({pca.explained_variance_ratio_[1] * 100:.1f} %)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    plt.tight_layout(); plt.savefig(cfg.DIR_FIGURAS / "fig_clusters_pca.png", dpi=200); plt.close()
    return [round(float(v) * 100, 1) for v in pca.explained_variance_ratio_]


def graficar_mapa_calor(perfiles: pd.DataFrame, df: pd.DataFrame) -> None:
    """Mapa de calor con el promedio de cada grupo en puntaje z (frente al total)."""
    medias = perfiles[cfg.VARIABLES]
    z = (medias - df[cfg.VARIABLES].mean()) / df[cfg.VARIABLES].std()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(z.T, cmap="Greys", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(z))); ax.set_xticklabels([f"Grupo {g}" for g in z.index])
    ax.set_yticks(range(len(cfg.VARIABLES))); ax.set_yticklabels([v.replace("_", " ") for v in cfg.VARIABLES], fontsize=8)
    for i in range(z.shape[1]):
        for j in range(z.shape[0]):
            ax.text(j, i, f"{z.iloc[j, i]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if z.iloc[j, i] > 0.9 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Puntaje z")
    plt.tight_layout(); plt.savefig(cfg.DIR_FIGURAS / "fig_perfiles_grupos.png", dpi=200); plt.close()


def graficar_tamanos(etiquetas) -> None:
    conteo = pd.Series(etiquetas).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3))
    barras = ax.bar([f"Grupo {g}" for g in conteo.index], conteo.values, color="#888888", edgecolor="black")
    for b, v in zip(barras, conteo.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, str(v), ha="center", fontsize=9)
    ax.set_ylabel("Número de estaciones"); ax.set_ylim(0, conteo.max() * 1.15)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    plt.tight_layout(); plt.savefig(cfg.DIR_FIGURAS / "fig_tamano_grupos.png", dpi=200); plt.close()


def graficar_dendrograma(X_proc) -> None:
    Z = linkage(X_proc, method="ward")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    dendrogram(Z, truncate_mode="lastp", p=20, ax=ax, color_threshold=0, above_threshold_color="black",
               leaf_font_size=8, no_labels=False)
    ax.set_xlabel("Estaciones agrupadas (entre paréntesis, cantidad)"); ax.set_ylabel("Distancia de Ward")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    plt.tight_layout(); plt.savefig(cfg.DIR_FIGURAS / "fig_dendrograma.png", dpi=200); plt.close()


def evaluar_estabilidad(X_proc, etiquetas_base, k: int, semillas=range(10)) -> dict:
    """Reentrena K-means con distintas semillas y mide cuánto coinciden los grupos (ARI)."""
    from sklearn.cluster import KMeans
    aris = []
    for s in semillas:
        et = KMeans(n_clusters=k, n_init=cfg.N_INICIOS_KMEANS, random_state=s).fit_predict(X_proc)
        aris.append(adjusted_rand_score(etiquetas_base, et))
    return {"n_semillas": len(list(semillas)), "ari_minimo": round(float(min(aris)), 4),
            "ari_promedio": round(float(np.mean(aris)), 4)}


def ejecutar(modelo, df: pd.DataFrame, X, tabla_k, referencia=None) -> dict:
    cfg.crear_directorios()
    X_proc = modelo.named_steps["preprocesador"].transform(X)
    km = modelo.named_steps["kmeans"]
    et = km.labels_
    k = km.n_clusters
    sil_muestras = silhouette_samples(X_proc, et)
    jer = entrenar_jerarquico(X_proc, k)

    m = {
        "k": int(k),
        "silueta": round(float(silhouette_score(X_proc, et)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X_proc, et)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X_proc, et)), 1),
        "inercia": round(float(km.inertia_), 2),
        "tamanos": {int(g): int(n) for g, n in pd.Series(et).value_counts().sort_index().items()},
        "silueta_por_grupo": {int(g): round(float(sil_muestras[et == g].mean()), 3) for g in sorted(set(et))},
        "jerarquico": {
            "silueta": round(float(silhouette_score(X_proc, jer)), 4),
            "davies_bouldin": round(float(davies_bouldin_score(X_proc, jer)), 4),
            "ari_contra_kmeans": round(float(adjusted_rand_score(et, jer)), 4),
        },
    }
    if referencia is not None:
        ref = referencia["perfil_simulado"].values
        m["ari_contra_perfil_simulado"] = round(float(adjusted_rand_score(ref, et)), 4)
        m["nmi_contra_perfil_simulado"] = round(float(normalized_mutual_info_score(ref, et)), 4)
        m["jerarquico"]["ari_contra_perfil_simulado"] = round(float(adjusted_rand_score(ref, jer)), 4)
        m["tabla_cruzada"] = pd.crosstab(pd.Series(et, name="grupo"), pd.Series(ref, name="perfil")).to_dict()

    m["estabilidad"] = evaluar_estabilidad(X_proc, et, k)
    perfiles = calcular_perfiles(df, et)
    perfiles.to_csv(cfg.ARCHIVO_PERFILES)
    m["perfiles"] = json.loads(perfiles.to_json(orient="index"))
    m["varianza_pca"] = graficar_pca(X_proc, et, km.cluster_centers_)
    graficar_mapa_calor(perfiles, df)
    graficar_tamanos(et)
    graficar_dendrograma(X_proc)
    m["tabla_k"] = json.loads(tabla_k.to_json(orient="records"))
    cfg.ARCHIVO_METRICAS.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return m
