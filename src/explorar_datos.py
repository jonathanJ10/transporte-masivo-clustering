"""
explorar_datos.py
-----------------
Análisis exploratorio (EDA): estadísticas, valores faltantes y gráficos.
Guarda reports/resumen_datos.json y figuras en reports/figures/.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as cfg
from src.preprocesamiento import cargar_dataset, validar_dataset

plt.rcParams.update({"font.family": "serif", "font.size": 10})


def resumir(df: pd.DataFrame) -> dict:
    corr = df[cfg.VARIABLES].corr()
    pares = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(cfg.VARIABLES)
             for b in cfg.VARIABLES[i + 1:]]
    pares.sort(key=lambda x: abs(x[2]), reverse=True)
    return {
        "n_estaciones": int(len(df)),
        "n_variables": len(cfg.VARIABLES),
        "nulos": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "descriptivos": json.loads(df[cfg.VARIABLES].describe().round(2).to_json()),
        "correlaciones_mas_fuertes": [(a, b, round(c, 2)) for a, b, c in pares[:5]],
    }


def graficar(df: pd.DataFrame) -> None:
    cfg.crear_directorios()
    # 1. Histogramas de todas las variables
    fig, ejes = plt.subplots(3, 4, figsize=(9, 6))
    for ax, var in zip(ejes.ravel(), cfg.VARIABLES):
        ax.hist(df[var].dropna(), bins=15, color="#888888", edgecolor="black")
        ax.set_title(var.replace("_", " "), fontsize=8)
        ax.tick_params(labelsize=7)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    for ax in ejes.ravel()[len(cfg.VARIABLES):]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(cfg.DIR_FIGURAS / "fig_histogramas.png", dpi=200); plt.close()

    # 2. Matriz de correlación
    corr = df[cfg.VARIABLES].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap="Greys", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    nombres = [v.replace("_", " ") for v in corr.columns]
    ax.set_xticklabels(nombres, rotation=60, ha="right", fontsize=7); ax.set_yticklabels(nombres, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if corr.iloc[i, j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(); plt.savefig(cfg.DIR_FIGURAS / "fig_correlaciones.png", dpi=200); plt.close()


def ejecutar() -> dict:
    df = cargar_dataset()
    validar_dataset(df)
    resumen = resumir(df)
    cfg.ARCHIVO_RESUMEN_DATOS.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    graficar(df)
    return resumen


if __name__ == "__main__":
    r = ejecutar()
    print(f"Estaciones: {r['n_estaciones']} | Nulos: {r['nulos']}")
