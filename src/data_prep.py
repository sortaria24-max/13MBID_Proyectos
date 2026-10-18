import pandas as pd
from pathlib import Path

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
    df_clientes = pd.read_csv("../data/raw/datos_clientes.csv", sep=";")
    df_resenas = pd.read_csv("../data/raw/datos_resenas.csv", sep=";")

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
    df_clientes_filtrado = df_clientes[(df_clientes['edad'] < 90) | (df_clientes['edad'].isna())]
    df_resenas_filtrado = df_resenas.drop_duplicates().copy()

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

    ####################################################################
    # Creación de atributos nuevos
    ####################################################################
    # Longitud del texto de la reseña, en caracteres
    df_integrado["longitud_resena"] = df_integrado["texto_resena"].str.len()
    # Cantidad de palabras del texto de la reseña
    df_integrado["cantidad_palabras_resena"] = df_integrado["texto_resena"].str.split().apply(len)
    # Tiempo de respuesta expresado en días, más interpretable que en horas
    df_integrado["tiempo_respuesta_dias"] = df_integrado["tiempo_respuesta_horas"] / 24

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
    # TODO: otros atributos que pueden generarse (implementar 2 al menos),
    # por ejemplo:
    # - Cantidad de reseñas por cliente
    # - Promedio de calificación por cliente
    # - Gasto promedio por cliente
    # - Proporción de reclamos por compras totales de un cliente
    ####################################################################

    ####################################################################
    # EXTRA: generar un reporte de las transformaciones aplicadas, por ejemplo:
    # - Cantidad de filas iniciales y finales, cantidad de valores nulos 
    # por columna, cantidad de duplicados eliminados, etc.
    # Se puede exportar a un JSON o CSV o txt que sirva como registro de las
    # transformaciones aplicadas a los datos.
    ####################################################################

if __name__ == "__main__":
    process_data()