# 13MBID – Predicción del sentimiento de reseñas de clientes

Proyecto de ciencia de datos desarrollado en la asignatura **13MBID – Metodologías de Gestión y Diseño de Proyectos de Big Data** (Máster Universitario en Big Data y Ciencia de Datos, VIU, ABR26-27).

Una empresa de retail y e-commerce quiere aprovechar sus reseñas históricas para **predecir el sentimiento** (positivo, neutral o negativo) de cada reseña a partir de los datos de la reseña y del cliente. El proyecto sigue la metodología **CRISP-DM**, gestionada con **Scrum** e integrada con prácticas de **DataOps** y **MLOps**.

- **Repositorio de código (GitHub):** https://github.com/sortaria24-max/13MBID_Proyectos
- **Datos y salidas versionadas (DagsHub + DVC):** https://dagshub.com/sortaria24/13MBID_Proyectos (requiere iniciar sesión con una cuenta gratuita)
- **Gestión de tareas (GitHub Projects):** https://github.com/users/sortaria24-max/projects/2

## Estado del proyecto

| Actividad | Fases CRISP-DM | Estado |
|---|---|---|
| Actividad Práctica I (Iteración 1) | Comprensión del negocio, comprensión de los datos y preparación de los datos | Completada |
| Actividad Práctica II (Iteración 2) | Modelado, evaluación y despliegue | En curso |

## Estructura del repositorio

| Carpeta / archivo | Contenido |
|---|---|
| `data/raw/` | Datos originales (`datos_clientes.csv`, `datos_resenas.csv`), versionados con DVC mediante sus archivos `.dvc` |
| `data/processed/` | Dataset integrado y preparado (`datos_integrados.csv`), salida del pipeline |
| `src/data_visualization.py` | Exploración de los datos: gráficos y reporte con ydata-profiling |
| `src/data_prep.py` | Preparación de los datos: selección, limpieza, imputación, integración, construcción de atributos y formateo |
| `test/test_data_quality.py` | Tests automatizados de calidad de datos (Pytest + Pandera): esquema, completitud, unicidad, integridad referencial y consistencia |
| `docs/` | Salidas del pipeline: figuras, `reporte_exploracion.html`, `reporte_calidad_datos.html` y `reporte_transformaciones.json` |
| `notebooks/` | Libretas de exploración, procesamiento y experimentación |
| `dvc.yaml` / `dvc.lock` | Definición del pipeline y registro de la versión exacta de cada dependencia y salida |
| `examples/` | Ejemplos de uso de la plantilla de la asignatura (DVC, MLflow, reportes); `data/raw/ejemplo.csv.dvc` pertenece a estos ejemplos |

## Pipeline de datos (DVC)

| Etapa | Qué hace | Salidas |
|---|---|---|
| `visualize_data` | Genera los gráficos de exploración y el reporte de ydata-profiling | `docs/figures/*.png`, `docs/reporte_exploracion.html` |
| `test_data_quality` | Ejecuta los tests de calidad sobre los datos originales y genera un informe HTML | `docs/reporte_calidad_datos.html` |
| `data_prep` | Aplica las transformaciones de preparación y registra cada una | `data/processed/datos_integrados.csv`, `docs/reporte_transformaciones.json` |

La etapa `test_data_quality` termina con `|| exit 0` a propósito: los tests documentan problemas reales de los datos originales (edades fuera de rango e identificadores de reseña duplicados), que se corrigen en la etapa `data_prep`. Así el fallo queda registrado en el informe sin detener el pipeline.

## Cómo reproducir el proyecto

Requisitos: [uv](https://docs.astral.sh/uv/) y acceso a DagsHub.

```bash
git clone https://github.com/sortaria24-max/13MBID_Proyectos.git
cd 13MBID_Proyectos
uv sync                 # crea el entorno con las versiones exactas de uv.lock
uv run dvc pull         # descarga los datos y salidas desde DagsHub
uv run dvc repro        # ejecuta las etapas del pipeline que hayan cambiado
uv run pytest test/test_data_quality.py -v   # (opcional) ejecuta los tests de calidad
```

Para subir nuevas versiones: commit y push en Git para el código, y `uv run dvc push` para los datos y salidas.

Para el registro de experimentos con MLflow (AP II), copia `.env.example` a `.env` y completa las credenciales de DagsHub. El archivo `.env` está excluido de Git.

## Autoría

Proyecto realizado por Jose Fulgencio Soriano García para la asignatura 13MBID, a partir de la plantilla del Prof. Dr. Horacio Kuna.