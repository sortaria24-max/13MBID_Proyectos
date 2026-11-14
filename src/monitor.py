import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.data_prep import create_attributes

# ---------------------------------------------------------------------------
# Umbrales de alerta
# ---------------------------------------------------------------------------
UMBRAL_PSI = 0.2               # PSI > 0.2: cambio relevante en la distribución
UMBRAL_DIF_CLASES = 0.10       # variación de más de 10 puntos en la proporción de una clase
UMBRAL_ACCURACY = 0.80         # objetivo de negocio
UMBRAL_RECALL_NEGATIVO = 0.85  # en test fue 0.926; avisar si cae por debajo de 0.85

VARIABLES_NUMERICAS = ["tiempo_respuesta_horas", "importe_compra"]


def calcular_psi(referencia: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index: mide cuánto cambió la distribución de una variable
    numérica respecto de la de referencia. Valores habituales de interpretación:
    < 0.1 sin cambios, 0.1 - 0.2 cambio moderado, > 0.2 cambio significativo."""
    # Los cortes se definen con los deciles de los datos de referencia
    cortes = np.unique(np.quantile(referencia, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    pct_ref = np.histogram(referencia, cortes)[0] / len(referencia)
    pct_act = np.histogram(actual, cortes)[0] / len(actual)
    # Se evita la división por cero en intervalos vacíos
    pct_ref = np.clip(pct_ref, 1e-6, None)
    pct_act = np.clip(pct_act, 1e-6, None)
    return float(np.sum((pct_act - pct_ref) * np.log(pct_act / pct_ref)))


def monitor(
    lote_path: str = "data/raw/resenas_nuevas.csv",
    referencia_path: str = "data/processed/datos_integrados.csv",
    model_path: str = "models/trained_model.pkl",
    output_path: str = "metrics/monitoring.json",
) -> dict:
    # ---------------------------------------------------------------------------
    # Lectura del modelo, los datos de referencia (entrenamiento) y el lote nuevo
    # ---------------------------------------------------------------------------
    pipeline = joblib.load(model_path)
    referencia = pd.read_csv(referencia_path)
    lote = pd.read_csv(lote_path, sep=";")

    # ---------------------------------------------------------------------------
    # Predicción sobre el lote, con los mismos atributos derivados que en el entrenamiento
    # ---------------------------------------------------------------------------
    X_lote = create_attributes(lote.drop(columns=["sentimiento"]))
    predicciones = pd.Series(pipeline.predict(X_lote))

    metricas = {"cantidad_resenas": len(lote)}
    alertas = []

    # ---------------------------------------------------------------------------
    # 1. Drift en los datos de entrada (variables numéricas)
    # ---------------------------------------------------------------------------
    for col in VARIABLES_NUMERICAS:
        psi = calcular_psi(referencia[col], lote[col])
        metricas[f"psi_{col}"] = round(psi, 4)
        if psi > UMBRAL_PSI:
            alertas.append(f"Drift en '{col}': PSI = {psi:.3f} (umbral {UMBRAL_PSI})")

    # ---------------------------------------------------------------------------
    # 2. Distribución de las clases predichas frente a la del entrenamiento
    # ---------------------------------------------------------------------------
    dist_ref = referencia["sentimiento"].value_counts(normalize=True)
    dist_pred = predicciones.value_counts(normalize=True)
    for clase in dist_ref.index:
        dif = dist_pred.get(clase, 0) - dist_ref[clase]
        metricas[f"proporcion_predicha_{clase.lower()}"] = round(float(dist_pred.get(clase, 0)), 4)
        if abs(dif) > UMBRAL_DIF_CLASES:
            alertas.append(f"Cambio en la proporción de '{clase}': {dist_ref[clase]:.1%} -> {dist_pred.get(clase, 0):.1%}")

    # ---------------------------------------------------------------------------
    # 3. Desempeño del modelo (el lote incluye el sentimiento real)
    # ---------------------------------------------------------------------------
    y_real = lote["sentimiento"]
    metricas["accuracy"] = round(accuracy_score(y_real, predicciones), 4)
    metricas["f1_macro"] = round(f1_score(y_real, predicciones, average="macro"), 4)
    metricas["recall_negativo"] = round(recall_score(y_real, predicciones, labels=["Negativo"], average="macro"), 4)
    if metricas["accuracy"] < UMBRAL_ACCURACY:
        alertas.append(f"Accuracy por debajo del objetivo: {metricas['accuracy']:.3f} < {UMBRAL_ACCURACY}")

        # ---------------------------------------------------------------------------
    # 4. EXTRA: controles adicionales
    # ---------------------------------------------------------------------------
    # 4a. Recall de la clase 'Negativo'. El objetivo de negocio es detectar a los clientes
    #     insatisfechos, así que se vigila aparte: en test fue 0.926 y se tolera una caída
    #     de unos 8 puntos antes de avisar.
    if metricas["recall_negativo"] < UMBRAL_RECALL_NEGATIVO:
        alertas.append(f"Recall de 'Negativo' por debajo del objetivo: {metricas['recall_negativo']:.3f} < {UMBRAL_RECALL_NEGATIVO}")

    # 4b. Proporción de reclamos formales en los datos de entrada. Es una variable muy
    #     relacionada con el sentimiento negativo: si cambia mucho, cambia la población
    #     de clientes aunque las métricas del modelo todavía no lo reflejen.
    prop_ref = (referencia["reclamo_formal"] == "Sí").mean()
    prop_lote = (lote["reclamo_formal"] == "Sí").mean()
    metricas["proporcion_reclamo_formal"] = round(float(prop_lote), 4)
    if abs(prop_lote - prop_ref) > UMBRAL_DIF_CLASES:
        alertas.append(f"Cambio en la proporción de reclamos formales: {prop_ref:.1%} -> {prop_lote:.1%}")

    # ---------------------------------------------------------------------------
    # Reporte
    # ---------------------------------------------------------------------------
    reporte = {"lote": str(lote_path), "metricas": metricas, "alertas": alertas}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print(f"Lote analizado: {lote_path} ({len(lote)} reseñas)")
    for k, v in metricas.items():
        print(f"  {k}: {v}")
    if alertas:
        print("ALERTAS:")
        for alerta in alertas:
            print(f"  - {alerta}")
        print("Se recomienda revisar los datos y evaluar el reentrenamiento del modelo.")
    else:
        print("Sin alertas: el modelo se comporta según lo esperado.")

    return reporte


if __name__ == "__main__":
    monitor(*sys.argv[1:2])