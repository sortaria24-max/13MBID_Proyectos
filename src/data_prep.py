import pandas as pd
from pathlib import Path
import json



def create_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye los atributos derivados que usa el modelo. Se usa en la preparación de los
    datos y en la API, para que el modelo reciba las mismas variables en entrenamiento
    y en producción.
    - De la reseña: longitud del texto, número de palabras y tiempo de respuesta en días.
    - Del cliente: número de reseñas, gasto promedio y proporción de reclamos, calculados
      con todas sus reseñas. Si no hay identificador de cliente (por ejemplo, en la API,
      que recibe una sola reseña), se calculan como si el cliente tuviera solo esa reseña.
    """
    df = df.copy()
    # Longitud del texto de la reseña, en caracteres
    df["longitud_resena"] = df["texto_resena"].str.len()
    # Cantidad de palabras del texto de la reseña
    df["cantidad_palabras_resena"] = df["texto_resena"].str.split().apply(len)
    # Tiempo de respuesta expresado en días, más interpretable que en horas
    df["tiempo_respuesta_dias"] = df["tiempo_respuesta_horas"] / 24

    # No se calcula la calificación promedio del cliente: incluye la calificación de la propia reseña, que determina el sentimiento (fuga de información)
    con_reclamo = df["reclamo_formal"] == "Sí"
    if "id_cliente" in df.columns:
        por_cliente = df.groupby("id_cliente")
        df["cantidad_resenas_cliente"] = por_cliente["id_resena"].transform("count")
        df["gasto_promedio_cliente"] = por_cliente["importe_compra"].transform("mean")
        df["proporcion_reclamos_cliente"] = con_reclamo.groupby(df["id_cliente"]).transform("mean")
    else:
        df["cantidad_resenas_cliente"] = 1
        df["gasto_promedio_cliente"] = df["importe_compra"]
        df["proporcion_reclamos_cliente"] = con_reclamo.astype(float)
    return df

def process_data(
    datos_cliente: str = "data/raw/datos_clientes.csv",
    datos_resenas: str = "data/raw/datos_resenas.csv",
    output_dir: str = "data/processed",
) -> None:
    """
    Función para procesar los datos de clientes y reseñas

    Args:
        datos_cliente (str): Ruta al archivo CSV de datos de clientes.
        datos_resenas (str): Ruta al archivo CSV de datos de reseñas.
        output_dir (str): Directorio donde se guardarán los archivos procesados.
    
    Returns:
        None
    """
    #Lectura de los datos
    df_clientes = pd.read_csv(datos_cliente, sep=";")
    df_resenas = pd.read_csv(datos_resenas, sep=";")

    # Registro de las transformaciones aplicadas (se exporta al final)
    reporte = {
        "filas_iniciales": {"clientes": len(df_clientes), "resenas": len(df_resenas)},
        "nulos_iniciales": {
            "edad": int(df_clientes["edad"].isna().sum()),
            "tiempo_respuesta_horas": int(df_resenas["tiempo_respuesta_horas"].isna().sum()),
        },
    }

    ####################################################################
    # Selección de los datos (filtro de columnas)
    # Agregar si corresponde otras acciones en función de la evaluación
    # de la calidad de los datos del paso previo.
    ####################################################################
    col_eliminar_clientes = ['canal_registro']
    df_clientes.drop(col_eliminar_clientes, inplace=True, axis=1)

    ####################################################################
    # Limpieza de los datos (filtro de filas)
    # Agregar si corresponde otras acciones en función de la evaluación
    # de la calidad de los datos del paso previo.
    ####################################################################
    # Se admiten edades de hasta 90 años inclusive, en coherencia con el rango 18-90 validado en test_data_quality.py
    df_clientes_filtrado = df_clientes[(df_clientes['edad'] <= 90) | (df_clientes['edad'].isna())].copy()
    df_resenas_filtrado = df_resenas.drop_duplicates().copy()

    reporte["clientes_eliminados_edad_fuera_de_rango"] = len(df_clientes) - len(df_clientes_filtrado)
    reporte["resenas_duplicadas_eliminadas"] = len(df_resenas) - len(df_resenas_filtrado)

    ####################################################################
    # Tratamiento de valores nulos
    ####################################################################
    df_clientes_filtrado['edad'] = df_clientes_filtrado.groupby("ciudad")["edad"]\
                       .transform(lambda x: x.fillna(x.median()))
    df_resenas_filtrado['tiempo_respuesta_horas'] = df_resenas_filtrado.groupby("categoria_producto")["tiempo_respuesta_horas"]\
                       .transform(lambda x: x.fillna(x.median()))

    ####################################################################
    # Integración de los datos
    ####################################################################
    df_integrado = pd.merge(
        df_resenas_filtrado,
        df_clientes_filtrado,
        on='id_cliente',
        how='inner'
    )

    reporte["resenas_perdidas_en_integracion"] = len(df_resenas_filtrado) - len(df_integrado)

    ####################################################################
    # Creación de atributos nuevos
    ####################################################################
    df_integrado = create_attributes(df_integrado)

    ####################################################################
    # Formateo de los datos
    ####################################################################
    # Se eliminan columnas identificadoras, la fecha (no se utiliza en esta etapa)
    # y 'calificacion', dado que determina directamente a 'sentimiento' y generaría data leakage
    columnas_a_eliminar = [
        "id_cliente",
        "id_resena",
        "calificacion",
        "fecha_resena",
    ]
    df_final = df_integrado.drop(columns=columnas_a_eliminar)

    ####################################################################
    # Generar el csv de salida
    ####################################################################
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df_final.to_csv(f"{output_dir}/datos_integrados.csv", index=False)

    ####################################################################
    # Reporte de las transformaciones aplicadas
    ####################################################################
    reporte["atributos_nuevos"] = [
        "longitud_resena",
        "cantidad_palabras_resena",
        "tiempo_respuesta_dias",
        "cantidad_resenas_cliente",
        "gasto_promedio_cliente",
        "proporcion_reclamos_cliente",
    ]
    reporte["dataset_final"] = {
        "filas": len(df_final),
        "columnas": df_final.shape[1],
        "filas_duplicadas": int(df_final.duplicated().sum()),
        "nulos_por_columna": {col: int(n) for col, n in df_final.isna().sum().items()},
    }
    Path("docs").mkdir(exist_ok=True)
    with open("docs/reporte_transformaciones.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_data()