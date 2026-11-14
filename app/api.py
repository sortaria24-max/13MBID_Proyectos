import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib

from contextlib import asynccontextmanager
from src.data_prep import create_attributes


MODEL_PATH = os.getenv("MODEL_PATH", "models/trained_model.pkl")
MODEL_URI = os.getenv("MODEL_URI", "models:/sentimiento-resenas@champion")

# Definimos los dominios de los atributos de entrada
CATEGORIAS = Literal["Alimentos", "Belleza", "Calzado", "Deportes",
                     "Electrónica", "Hogar", "Indumentaria", "Libros"]
PLATAFORMAS = Literal["Amazon", "Falabella", "Flipkart", "JioMart",
                      "Meesho", "MercadoLibre", "Nykaa", "Zepto"]
CIUDADES = Literal["Buenos Aires", "Córdoba", "La Plata", "Mar del Plata", "Mendoza",
                   "Posadas", "Rosario", "Salta", "San Miguel de Tucumán", "Santa Fe"]
GENEROS = Literal["Femenino", "Masculino", "Otro"]
CANALES = Literal["Online", "Offline"]
SI_NO = Literal["Sí", "No"]


class Resena(BaseModel):
    """ Modelo de datos para la reseña"""
    # datos de la reseña y la compra
    texto_resena: str = Field(min_length=1, examples=["El envío fue rápido y el parlante llegó en perfectas condiciones, lo recomiendo."], description="Texto de la reseña del producto")
    categoria_producto: CATEGORIAS = Field(examples=["Electrónica"], description="Categoría del producto")
    canal_compra: CANALES = Field(examples=["Online"], description="Canal de compra del producto")
    plataforma: PLATAFORMAS = Field(examples=["Amazon"], description="Plataforma de compra del producto")
    importe_compra: float = Field(gt=0, examples=[199.99], description="Importe de la compra del producto")
    tiempo_respuesta_horas: float = Field(gt=0, examples=[24.5], description="Tiempo de respuesta del servicio al cliente en horas")
    problema_resuelto: SI_NO = Field(examples=["Sí"], description="Indica si el problema fue resuelto")
    reclamo_formal: SI_NO = Field(examples=["No"], description="Indica si se realizó un reclamo formal")
    # datos del cliente
    edad: float = Field(ge=18, le=90, examples=[30], description="Edad del usuario")
    genero: GENEROS = Field(examples=["Masculino"], description="Género del usuario")
    ciudad: CIUDADES = Field(examples=["Buenos Aires"], description="Ciudad del usuario")
    antiguedad_cliente_meses: int = Field(ge=0, examples=[12], description="Antigüedad del cliente en meses")

    # TODO: incorporar los atributos adicionales que hayan sido definidos en el proceso de transformación


class PredictionResponse(BaseModel):
    """ Modelo de datos para la respuesta de predicción"""
    sentimiento: str
    probabilidad: dict[str, float]


modelo = {}


def cargar_modelo():
    """ Carga el modelo entrenado desde el archivo especificado en MODEL_PATH """
    if os.path.exists(MODEL_PATH):
        modelo["pipeline"] = joblib.load(MODEL_PATH)
        modelo["origen"] = MODEL_PATH
        return
    if os.getenv("MLFLOW_TRACKING_URI"):
        import mlflow.sklearn
        try:
            modelo["pipeline"] = mlflow.sklearn.load_model(MODEL_URI)
            modelo["origen"] = MODEL_URI
        except Exception as e:
            print(f"Error al cargar el modelo desde {MODEL_URI}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cargar el modelo entrenado al iniciar la aplicación (archivo local o registro de MLflow)
    cargar_modelo()
    yield
    modelo.clear()


app = FastAPI(title="API de Clasificación de Sentimientos",
              description="API para predecir el sentimiento de reseñas de productos utilizando un modelo de clasificación entrenado.",
              version="1.0.0",
              lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Clasificación de Sentimientos. Use el endpoint /predict para obtener predicciones."}


@app.get("/health")
def health_check():
    """ Endpoint para verificar el estado de la API """
    if "pipeline" in modelo:
        return {"status": "ok", "message": "La API está funcionando correctamente."}
    else:
        raise HTTPException(status_code=503, detail="El modelo no está cargado. La API no está disponible.")


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(resena: Resena):
    """ Endpoint para predecir el sentimiento de una reseña de producto """
    if "pipeline" not in modelo:
        raise HTTPException(status_code=503, detail="El modelo no está cargado. La API no está disponible.")

    pipeline = modelo["pipeline"]

    # Crear los atributos adicionales
    df = create_attributes(pd.DataFrame([resena.dict()]))

    sentimiento = pipeline.predict(df)[0]
    probabilidades = {}

    if hasattr(pipeline, "predict_proba"):
        probabilidades = dict(zip(pipeline.classes_, pipeline.predict_proba(df)[0].round(4).tolist()))

    return PredictionResponse(sentimiento=sentimiento, probabilidad=probabilidades)



@app.get("/opciones")
def get_options():
    """ Endpoint para obtener las opciones de los atributos categóricos """
    return {
        nombre: list(campo.annotation.__args__)
        for nombre, campo in Resena.model_fields.items()
        if getattr(campo.annotation, "__origin__", None) is Literal
    }