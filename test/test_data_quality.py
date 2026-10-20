import pandas as pd
import pytest
from pandera.pandas import DataFrameSchema, Column, Check

GENEROS = ["Femenino", "Masculino", "Otro"]
SENTIMIENTOS = ["Positivo", "Negativo", "Neutral"]

@pytest.fixture
def datos_cliente():
    """ Función para cargar los datos de clientes a un dataframe

    Args:
        None
    
    Returns:
        pd.DataFrame: Dataframe con los datos de clientes
    """
    df = pd.read_csv("data/raw/datos_clientes.csv", sep=";")
    return df


@pytest.fixture
def datos_resenas():
    """ Función para cargar los datos de reseñas a un dataframe

    Args:
        None
    
    Returns:
        pd.DataFrame: Dataframe con los datos de reseñas
    """
    df = pd.read_csv("data/raw/datos_resenas.csv", sep=";")
    return df


def test_schema_datos_clientes(datos_cliente):
    """ Función para validar el esquema de los datos de clientes

    Args:
        datos_cliente (pd.DataFrame): Dataframe con los datos de clientes
    
    Returns:
        None
    """
    schema = DataFrameSchema(
        {
            "id_cliente": Column(int, nullable=False),
            "edad": Column(float, nullable=True, checks=Check.in_range(18, 90)),
            "genero": Column(str, nullable=False, checks=Check.isin(GENEROS)),
            "ciudad": Column(str, nullable=False),
            "antiguedad_cliente_meses": Column(int, nullable=False, checks=Check.ge(0)),
            "canal_registro": Column(str, nullable=False),
        }
    )

    # Lazy = True permite que se validen todas las columnas y se muestren todos los
    #       errores encontrados en lugar de detenerse en el primer error.
    schema.validate(datos_cliente, lazy=True)

def test_schema_datos_resenas(datos_resenas):
    """ Función para validar el esquema de los datos de reseñas

    Args:
        datos_resenas (pd.DataFrame): Dataframe con los datos de reseñas
        
    Returns:
        None
    """
    schema = DataFrameSchema(
        {
            "id_resena": Column(int, nullable=False),
            "id_cliente": Column(int, nullable=False),
            "categoria_producto": Column(str, nullable=False),
            "canal_compra": Column(str, nullable=False, checks=Check.isin(["Online", "Offline"])),
            "plataforma": Column(str, nullable=False),
            "calificacion": Column(int, nullable=False, checks=Check.in_range(1, 5)),
            "texto_resena": Column(str, nullable=False),
            "tiempo_respuesta_horas": Column(float, nullable=True, checks=Check.ge(0)),
            "problema_resuelto": Column(str, nullable=False, checks=Check.isin(["Sí", "No"])),
            "reclamo_formal": Column(str, nullable=False, checks=Check.isin(["Sí", "No"])),
            "importe_compra": Column(float, nullable=False, checks=Check.gt(0)),
            "fecha_resena": Column(str, nullable=False, checks=Check.str_matches(r"^\d{4}-\d{2}-\d{2}$")),
            "sentimiento": Column(str, nullable=False, checks=Check.isin(SENTIMIENTOS)),
        }
   )
        
    schema.validate(datos_resenas, lazy=True)

def test_basicos_clientes(datos_cliente):
    """ Función para validar los datos de clientes

    Args:
        datos_cliente (pd.DataFrame): Dataframe con los datos de clientes
    
    Returns:
        None
    """
    # Validar que no existan valores nulos en las columnas obligatorias
    assert not datos_cliente.empty, "El dataframe de clientes está vacío"
    assert datos_cliente.shape[1] == 6, "El dataframe de clientes no tiene 6 columnas"
    assert datos_cliente["id_cliente"].notnull().all(), "La columna id_cliente tiene valores nulos"
    assert (datos_cliente["edad"].dropna() > 0).all(), "Hay clientes con edad menor o igual a 0"

def test_basicos_resenas(datos_resenas):
    """ Función para validar los datos de reseñas

    Args:
        datos_resenas (pd.DataFrame): Dataframe con los datos de reseñas
    
    Returns:
        None
    """
    # Validar que no existan valores nulos en las columnas obligatorias
    assert not datos_resenas.empty, "El dataframe de reseñas está vacío"
    assert datos_resenas.shape[1] == 13, "El dataframe de reseñas no tiene 13 columnas"
    assert datos_resenas["id_resena"].notnull().all(), "La columna id_resena tiene valores nulos"
    assert datos_resenas["id_cliente"].notnull().all(), "La columna id_cliente tiene valores nulos"
    assert datos_resenas["calificacion"].between(1, 5).all(), "Hay calificaciones fuera del rango 1-5"

def test_integridad_referencial(datos_cliente, datos_resenas):
    """ Función para validar la integridad referencial entre los datos de clientes y reseñas

    Args:
        datos_cliente (pd.DataFrame): Dataframe con los datos de clientes
        datos_resenas (pd.DataFrame): Dataframe con los datos de reseñas
    
    Returns:
        None
    """
    # Validar que todos los id_cliente en datos_resenas existan en datos_cliente
    df_ids = datos_resenas[['id_cliente']].merge(
        datos_cliente[['id_cliente']],
        on='id_cliente',
        how='left',
        indicator=True # genera la columna _merge con el resultado de la unión
    )
    integridad_referencial = DataFrameSchema(
        {
            "_merge": Column(str, nullable=False, checks=Check.isin(["both"]))
        }
    )
    integridad_referencial.validate(df_ids, lazy=True)

def test_unicidad_id_cliente(datos_cliente):
    """ Función para validar que no existan id_cliente repetidos

    Args:
        datos_cliente (pd.DataFrame): Dataframe con los datos de clientes

    Returns:
        None
    """
    repetidos = datos_cliente["id_cliente"].duplicated().sum()
    assert repetidos == 0, f"Hay {repetidos} id_cliente repetidos"


def test_unicidad_id_resena(datos_resenas):
    """ Función para validar que no existan id_resena repetidos

    Args:
        datos_resenas (pd.DataFrame): Dataframe con los datos de reseñas

    Returns:
        None
    """
    repetidos = datos_resenas["id_resena"].duplicated().sum()
    assert repetidos == 0, f"Hay {repetidos} id_resena repetidos"


def test_consistencia_calificacion_sentimiento(datos_resenas):
    """ Función para validar la consistencia entre calificación y sentimiento.
    Una calificación baja (1-2) no debería tener sentimiento positivo, ni una
    calificación alta (4-5) sentimiento negativo, en más de un 5 % de los casos.

    Args:
        datos_resenas (pd.DataFrame): Dataframe con los datos de reseñas

    Returns:
        None
    """
    bajas = datos_resenas[datos_resenas["calificacion"] < 3]
    altas = datos_resenas[datos_resenas["calificacion"] > 3]

    prop_bajas_positivas = (bajas["sentimiento"] == "Positivo").mean()
    prop_altas_negativas = (altas["sentimiento"] == "Negativo").mean()

    assert prop_bajas_positivas <= 0.05, f"{prop_bajas_positivas:.2%} de las calificaciones bajas son positivas"
    assert prop_altas_negativas <= 0.05, f"{prop_altas_negativas:.2%} de las calificaciones altas son negativas"