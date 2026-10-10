"""
config.py
---------
Configuración central del proyecto: rutas, constantes y listas de variables.
Todos los demás módulos importan desde aquí; cualquier cambio se hace en un solo lugar.
"""
from pathlib import Path

# ---------- Rutas ----------
RAIZ = Path(__file__).resolve().parents[1]
DIR_DATOS_RAW = RAIZ / "data" / "raw"
DIR_DATOS_PROC = RAIZ / "data" / "processed"
DIR_MODELOS = RAIZ / "models"
DIR_REPORTES = RAIZ / "reports"
DIR_FIGURAS = DIR_REPORTES / "figures"

ARCHIVO_DATASET = DIR_DATOS_RAW / "estaciones_transporte_masivo.csv"
ARCHIVO_REFERENCIA = DIR_DATOS_RAW / "perfiles_simulados_referencia.csv"
ARCHIVO_ASIGNACION = DIR_DATOS_PROC / "estaciones_con_cluster.csv"
ARCHIVO_MODELO = DIR_MODELOS / "kmeans_estaciones.joblib"
ARCHIVO_METRICAS = DIR_REPORTES / "metricas.json"
ARCHIVO_RESUMEN_DATOS = DIR_REPORTES / "resumen_datos.json"
ARCHIVO_TABLA_K = DIR_REPORTES / "evaluacion_k.csv"
ARCHIVO_PERFILES = DIR_REPORTES / "perfiles_clusters.csv"

# ---------- Parámetros generales ----------
SEMILLA = 42            # resultados reproducibles
RANGO_K = range(2, 9)   # valores de k a evaluar (2 a 8)
N_INICIOS_KMEANS = 10   # reinicios aleatorios de K-means

# ---------- Variables ----------
COLUMNA_ID = "id_estacion"
VARIABLES = [
    "pasajeros_pico_manana",   # pasajeros/hora entre 6:00 y 9:00
    "pasajeros_pico_tarde",    # pasajeros/hora entre 16:00 y 19:00
    "pasajeros_valle",         # pasajeros/hora entre 10:00 y 15:00
    "proporcion_fin_semana",   # demanda fin de semana / demanda entre semana
    "ocupacion_media_pct",     # ocupación media de los buses (%)
    "variabilidad_diaria",     # coeficiente de variación de la demanda diaria
    "tiempo_espera_min",       # espera media del usuario (minutos)
    "porcentaje_transbordos",  # % de usuarios que hacen transbordo
    "numero_rutas",            # rutas que pasan por la estación
    "distancia_centro_km",     # distancia al centro de la ciudad (km)
    "sensibilidad_lluvia_pct", # aumento de demanda con lluvia (%)
]
COLUMNAS_DATASET = [COLUMNA_ID] + VARIABLES


def crear_directorios() -> None:
    """Crea las carpetas de salida si no existen."""
    for carpeta in (DIR_DATOS_RAW, DIR_DATOS_PROC, DIR_MODELOS, DIR_FIGURAS):
        carpeta.mkdir(parents=True, exist_ok=True)
