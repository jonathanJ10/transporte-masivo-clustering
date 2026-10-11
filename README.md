# Agrupamiento de estaciones de transporte masivo (aprendizaje no supervisado)

Proyecto de la asignatura **Inteligencia Artificial** (Ingeniería de Software, Universidad Iberoamericana).
Con **K-means** se descubren perfiles de estaciones de un sistema de transporte masivo (por ejemplo, residencial, comercial, industrial) a partir de su demanda y operación, sin etiquetas previas. El resultado se compara con un **agrupamiento jerárquico** (Ward).

## Integrantes

| Integrante | Rol principal |
|---|---|
| Johathan Granobles Penagos | Datos y generación del dataset |
| Danilo Moreno | QA e Implementador |


Docente: William Ruiz.

## Estructura del proyecto

```
transporte-masivo-clustering/
├── main.py                  # Ejecuta todo el flujo
├── requirements.txt         # Librerías necesarias
├── pytest.ini
├── data/
│   ├── raw/estaciones_transporte_masivo.csv        # Dataset (190 estaciones)
│   ├── raw/perfiles_simulados_referencia.csv       # Solo para validar al final
│   └── processed/estaciones_con_cluster.csv        # Estaciones con su grupo
├── src/
│   ├── config.py            # Rutas, constantes y variables
│   ├── generar_dataset.py   # Crea el dataset sintético
│   ├── explorar_datos.py    # Análisis exploratorio
│   ├── preprocesamiento.py  # Carga, validación, imputación y estandarización
│   ├── seleccion_k.py       # Codo, silueta y otras métricas para elegir k
│   ├── entrenar_modelo.py   # K-means y agrupamiento jerárquico
│   ├── evaluar_modelo.py    # Métricas, perfiles, PCA y dendrograma
│   └── asignar_cluster.py   # Asigna estaciones nuevas a un grupo
├── tests/test_pipeline.py   # 18 pruebas automatizadas
├── models/                  # Modelo entrenado (.joblib)
├── reports/                 # Métricas, perfiles y figuras
└── docs/                    # Documentos Word/PDF de la entrega
```

## Cómo ejecutarlo

```bash
git clone https://github.com/<usuario>/transporte-masivo-clustering.git
cd transporte-masivo-clustering

python -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                     # Flujo completo
python -m pytest -v                # Pruebas
python -m src.asignar_cluster      # Asignar estaciones de ejemplo
```

Para volver a crear el dataset: `python main.py --regenerar`.

## Nota sobre los datos

No se halló una fuente abierta con todas las variables, por lo que el dataset es **sintético** (semilla fija `42`). El perfil con el que se generó cada estación se guarda aparte y **no se usa para entrenar**; solo permite comprobar al final si el algoritmo recuperó los grupos.

## Video

Enlace: [pegar enlace]
