# Importación de librerías y supresión de advertencias
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def visualize_data(
    datos_cliente: str = "data/raw/datos_clientes.csv",
    datos_resenas: str = "data/raw/datos_resenas.csv",
    output_dir: str = "docs/figures",
) -> None:
    """
    Función para visualizar los datos de clientes y reseñas
    """
    # Creación de directorio de salida si no existe
    Output_dir = Path(output_dir)
    Output_dir.mkdir(parents=True, exist_ok=True)

    # Lectura de los datos
    df_clientes = pd.read_csv(datos_cliente, sep=";")
    df_resenas = pd.read_csv(datos_resenas, sep=";")

    # Configuración general
    sns.set(style="whitegrid", palette="muted")

    # Grafico 1 - Distribución de la variable target (sentimiento)
    orden_valores = ["Positivo", "Negativo", "Neutro"]   
    sns.countplot(x="sentimiento", data=df_resenas, order=orden_valores)
    plt.title("Distribución de la variable objetivo (sentimiento)")
    plt.xlabel("Sentimiento expresado en la reseña")
    plt.ylabel("Cantidad de reseñas")
    plt.savefig(Output_dir / "distribucion_sentimiento.png")
    plt.close()

    # Grafico 2 - Longitud del texto de las reseñas
    df_resenas["longitud_texto"] = df_resenas["texto_resena"].str.len()
    sns.boxplot(x="sentimiento", y="longitud_texto", data=df_resenas, order=["Negativo", "Neutral", "Positivo"])
    plt.title("Longitud del texto de la reseña según sentimiento")
    plt.xlabel("Sentimiento")
    plt.ylabel("Cantidad de caracteres")
    plt.savefig(Output_dir / "longitud_texto.png")
    plt.close()

    # se descarta la columna auxiliar, ya que solo se utilizó para esta visualización
    df_resenas.drop(columns=["longitud_texto"], inplace=True)

    # Grafico 3 - Matriz de correlación entre variables numéricas
    num_df = df_clientes.select_dtypes(include=['float64', 'int64'])
    corr = num_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de correlaciones - Clientes')
    plt.savefig(Output_dir / "matriz_correlacion.png")
    plt.close()

    num_df = df_resenas.select_dtypes(include=['float64', 'int64'])
    corr = num_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de correlaciones - Reseñas')
    plt.savefig(Output_dir / "matriz_correlacion_resenas.png")
    plt.close()

##############################################################################################
# TODO: Agregar al menos dos (2) visualizaciones más del escenario.
#       Posiblemente, alguna que vincule a alguno de los atributos con la variable objetivo.
#       Por ejmplo: sentimiento según tiempo de respuesta.
#       También puedes ser la distribución de valores de otras vsariable.
#-------
# EXTRA: Generar un reporte automático con algunaa librería como sweetviz, dtale.
##############################################################################################

if __name__ == "__main__":
    visualize_data()
