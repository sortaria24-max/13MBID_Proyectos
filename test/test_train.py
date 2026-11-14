import json
from pathlib import Path

import pytest

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

from src.train_model import load_data, train_model

# Objetivo de negocio: 80% de precisión en el conjunto de prueba
TARGET_ACCURACY = 0.80


def test_train_model(tmp_path):
    # Configurar acceso a los archivos
    project_root = Path(__file__).resolve().parents[1]
    baseline_path = project_root / "metrics" / "train_metrics.json"

    if not baseline_path.exists():
        pytest.skip(f"Archivo de referencia {baseline_path} no encontrado. Se omite la prueba.")

    # Cargar métricas de referencia
    with open(baseline_path, "r") as f:
        baseline = json.load(f)

    # Ejecutar la función de entrenamiento del modelo
    data_path = project_root / "data" / "processed" / "datos_integrados.csv"
    model_output_path = tmp_path / "prod_model.pkl"
    preprocessor_output_path = tmp_path / "prod_preprocessor.pkl"
    metrics_output_path = tmp_path / "train_metrics.json"

    _, _, metrics = train_model(
        data_path=str(data_path),
        model_output_path=str(model_output_path),
        preprocessor_output_path=str(preprocessor_output_path),
        metrics_output_path=str(metrics_output_path),
        figures_output_path=str(tmp_path),
        # El test no debe registrar versiones nuevas del modelo ni mover el alias champion
        registered_model_name=None,
    )

    # Se corroboran las métricas generadas
    assert set(metrics.keys()) == set(baseline.keys()), "Las métricas generadas no coinciden con las métricas de referencia."

    atol = 1e-9  # Tolerancia absoluta para la comparación de métricas
    for k in baseline.keys():
        assert metrics[k] == pytest.approx(baseline[k], rel=0, abs=atol), f"La métrica '{k}' no coincide con la referencia."


def test_train_model_accuracy_threshold():
    # Configurar acceso a los archivos
    project_root = Path(__file__).resolve().parents[1]
    metrics_path = project_root / "metrics" / "train_metrics.json"
    if not metrics_path.exists():
        pytest.skip(f"Archivo de métricas {metrics_path} no encontrado. Se omite la prueba.")
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert metrics["test_accuracy"] >= TARGET_ACCURACY, f"La precisión del modelo en el conjunto de prueba es menor que el umbral objetivo de {TARGET_ACCURACY}."


# Umbrales adicionales definidos en el plan de pruebas (issue #25)
TARGET_F1_MACRO = 0.80
TARGET_RECALL_NEGATIVO = 0.85
MEJORA_MINIMA_SOBRE_BASELINE = 0.10


def test_train_model_f1_y_recall_negativo():
    """El modelo funciona bien en las tres clases y detecta la mayoría de reseñas negativas."""
    project_root = Path(__file__).resolve().parents[1]
    metrics_path = project_root / "metrics" / "train_metrics.json"
    if not metrics_path.exists():
        pytest.skip(f"Archivo de métricas {metrics_path} no encontrado. Se omite la prueba.")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert metrics["test_f1_macro"] >= TARGET_F1_MACRO, f"El F1 macro es menor que {TARGET_F1_MACRO}."
    assert metrics["test_recall_negativo"] >= TARGET_RECALL_NEGATIVO, (
        f"El modelo detecta menos del {TARGET_RECALL_NEGATIVO:.0%} de las reseñas negativas."
    )


def test_train_model_supera_baseline():
    """El modelo supera en al menos 10 puntos a un modelo base que siempre predice la clase mayoritaria."""
    project_root = Path(__file__).resolve().parents[1]
    metrics_path = project_root / "metrics" / "train_metrics.json"
    if not metrics_path.exists():
        pytest.skip(f"Archivo de métricas {metrics_path} no encontrado. Se omite la prueba.")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    data_path = project_root / "data" / "processed" / "datos_integrados.csv"
    X_train, _, X_test, y_train, _, y_test, _, _ = load_data(str(data_path))
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    accuracy_baseline = baseline.score(X_test, y_test)

    assert metrics["test_accuracy"] >= accuracy_baseline + MEJORA_MINIMA_SOBRE_BASELINE, (
        f"El modelo no mejora lo suficiente al baseline (accuracy {accuracy_baseline:.3f})."
    )


def test_modelo_admite_valores_nulos():
    """En producción pueden llegar campos vacíos: el pipeline los imputa y predice igualmente."""
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "trained_model.pkl"
    if not model_path.exists():
        pytest.skip(f"Modelo {model_path} no encontrado. Se omite la prueba.")

    modelo = joblib.load(model_path)
    data_path = project_root / "data" / "processed" / "datos_integrados.csv"
    _, _, X_test, _, _, _, _, _ = load_data(str(data_path))
    muestra = X_test.head(5).copy()
    muestra.loc[:, ["edad", "ciudad", "texto_resena"]] = np.nan

    predicciones = modelo.predict(muestra)
    assert set(predicciones) <= {"Negativo", "Neutral", "Positivo"}
    
if __name__ == "__main__":
    test_train_model()
    test_train_model_accuracy_threshold()