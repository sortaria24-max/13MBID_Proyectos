# Se importan las librerías necesarias y se suprimen las advertencias
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REGISTERED_MODEL_NAME = "sentimiento-resenas"
ALIAS_PRODUCCION = "champion"
UMBRAL_ACCURACY = 0.8
UMBRAL_F1_MACRO = 0.8

def load_data(data_path: str = "data/processed/datos_integrados.csv"):
    """
    Carga los datos desde un archivo CSV y devuelve un DataFrame de pandas.

    Args:
        data_path (str): Ruta al archivo CSV que contiene los datos procesados.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y: Tupla que contiene los conjuntos de entrenamiento, validación y prueba, así como las características y etiquetas originales.
    """
    # Lectura de los datos
    df = pd.read_csv(data_path)

    target = "sentimiento"
    features_X = df.drop(columns=[target])
    labels_y = df[target]

    # Primero se separa el test final (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features_X, labels_y, test_size=0.10, random_state=42, stratify=labels_y
    )

    # Luego se separan train y validation (22% del 90% es aprox. el 20% del total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.22, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y


def create_preprocessor(features_X: pd.DataFrame, text_col: str = "texto_resena"):
    """
    Crea el preprocesador del pipeline. Incluye la imputación de valores faltantes,
    para que el modelo en producción pueda recibir datos sin procesar previamente:
    - Numéricas: imputación por la mediana y estandarización.
    - Categóricas: imputación por el valor más frecuente y one-hot encoding.
    - Texto de la reseña: los nulos se sustituyen por texto vacío y se vectoriza con TF-IDF.
    El texto se excluye de las categóricas para no codificarlo dos veces.
    """
    numeric_features = features_X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [
        col for col in features_X.select_dtypes(include=["object", "string"]).columns
        if col != text_col
    ]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    # El imputador devuelve una matriz de una columna; TF-IDF necesita una lista de textos,
    # por eso se aplana con np.ravel antes de vectorizar.
    text_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
        ("aplanar", FunctionTransformer(np.ravel)),
        ("tfidf", TfidfVectorizer(max_features=300)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("text", text_transformer, [text_col]),
        ]
    )
    return preprocessor


def train_model(
    data_path: str = "data/processed/datos_integrados.csv",
    model_output_path: str = "models/trained_model.pkl",
    preprocessor_output_path: str = "models/preprocessor.pkl",
    metrics_output_path: str = "metrics/train_metrics.json",
    figures_output_path: str = "docs/figures",
    registered_model_name: str | None = REGISTERED_MODEL_NAME,
):
    """
    Entrena un modelo de clasificación utilizando los datos procesados y guarda el modelo, el preprocesador y las métricas.

    Args:
        data_path (str): Ruta al archivo CSV que contiene los datos procesados.
        model_output_path (str): Ruta donde se guardará el modelo entrenado.
        preprocessor_output_path (str): Ruta donde se guardará el preprocesador.
        metrics_output_path (str): Ruta donde se guardarán las métricas de entrenamiento.
        figures_output_path (str): Ruta donde se guardarán las figuras generadas durante el entrenamiento.
    """
    # Configuración de las variables de entorno para MLFlow
    # Recordar que debe existir un archivo .env en la raíz del proyecto con las variables de entorno necesarias
    # Las credenciales de MLFlow se pueden obtener desde la interfaz web de DagsHub en la sección de "Settings" y Tokens
    load_dotenv(find_dotenv())
    mlflow.set_experiment(
        "Proyecto 13MBID - V2 - Entrenamiento de modelos de clasificación de sentimientos"
    )

    # Carga de datos
    X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y = load_data(
        data_path
    )

    # Creación del preprocesador
    preprocessor = create_preprocessor(features_X)

    # Modelo elegido en la experimentación (MD01): regresión logística balanceada
    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    # Creación del pipeline que incluye el preprocesador y el modelo
    pipeline = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    # Entrenamiento del modelo
    pipeline.fit(X_train, y_train)

    # Se evalúa el modelo en validación y en test con las mismas métricas
    def calcular_metricas(y_true, y_pred, y_proba, prefijo):
        return {
            f"{prefijo}_accuracy": accuracy_score(y_true, y_pred),
            f"{prefijo}_precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            f"{prefijo}_recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            f"{prefijo}_f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            # Recall de la clase Negativo: proporción de reseñas negativas que el modelo detecta
            f"{prefijo}_recall_negativo": recall_score(y_true, y_pred, labels=["Negativo"], average="macro", zero_division=0),
            f"{prefijo}_roc_auc_ovr": roc_auc_score(y_true, y_proba, multi_class="ovr", labels=pipeline.classes_),
        }

    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)
    metrics = {
        **calcular_metricas(y_val, y_val_pred, pipeline.predict_proba(X_val), "val"),
        **calcular_metricas(y_test, y_test_pred, pipeline.predict_proba(X_test), "test"),
    }

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Guardado de la matriz de confusión
    Path(figures_output_path).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_test_pred, labels=pipeline.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Logistic Regression - Test Set")
    cm_fig_path = f"{figures_output_path}/confusion_matrix_test.png"
    plt.savefig(cm_fig_path)
    plt.close()

    # Registro general en MLFlow
    signature = infer_signature(X_train, pipeline.predict(X_train))

    with mlflow.start_run(run_name="Pipeline (prod) - Logistic Regression") as run:
        mlflow.log_params(model.get_params())
        mlflow.log_params(
            {
                "train_sample_size": len(X_train),
                "validation_sample_size": len(X_val),
                "test_sample_size": len(X_test),
                "tfidf_max_features": 300,
            }
        )

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_fig_path, artifact_path="figures")
        model_info = mlflow.sklearn.log_model(
            pipeline, name="model", signature=signature, input_example=X_train.head(3),
            registered_model_name=registered_model_name,
        # MLflow serializa el modelo con skops, que exige declarar los tipos no estándar del pipeline
        skops_trusted_types=["numpy.dtype", "numpy.ravel"],
        )

        run_id = mlflow.active_run().info.run_id
        print(f"Modelo registrado en MLFlow a raíz del run ID: {run_id}")

    if registered_model_name:
        version = model_info.registered_model_version
        if metrics["test_accuracy"] >= UMBRAL_ACCURACY and metrics["test_f1_macro"] >= UMBRAL_F1_MACRO:
            mlflow.MlflowClient().set_registered_model_alias(
                name=registered_model_name, version=version, alias=ALIAS_PRODUCCION
            )
            print(f"Modelo registrado como '{registered_model_name}' con alias '{ALIAS_PRODUCCION}'")
        else:
            print(f"El modelo no alcanza los umbrales (accuracy {UMBRAL_ACCURACY}, F1 macro {UMBRAL_F1_MACRO}). No se registra como '{ALIAS_PRODUCCION}'.")

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(preprocessor_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)

    # Guardado del modelo entrenado y del preprocesador
    joblib.dump(pipeline, model_output_path)
    joblib.dump(pipeline.named_steps["prep"], preprocessor_output_path)

    # Guardado de métricas en un archivo JSON
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return pipeline, pipeline.named_steps["prep"], metrics


if __name__ == "__main__":
    train_model()
