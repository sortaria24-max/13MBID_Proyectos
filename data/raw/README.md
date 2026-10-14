# Diccionario de datos del escenario

## 1. Contexto del negocio

Este escenario (sintético) es de una empresa de retail y comercio electrónico que quiere comprender mejor la experiencia de sus clientes a partir de reseñas de compras. El objetivo principal es predecir el sentimiento expresado por cada cliente en su reseña para priorizar la atención a casos con mayor probabilidad de insatisfacción.

La información disponible se organiza en dos fuentes de datos relacionadas:

- `datos_clientes.csv`: información demográfica del cliente.
- `datos_resenas.csv`: información de la compra y del texto de la reseña.

La relación entre ambas tablas se realiza mediante el identificador `id_cliente`.

---

## 2. Fuentes de datos

### 2.1. Dataset de clientes

Archivo: `data/raw/datos_clientes.csv`

Este archivo contiene una fila por cliente y describe quienes son los compradores del negocio. La clave principal es `id_cliente`, que permite vincular cada cliente con sus reseñas.

### 2.2. Dataset de reseñas

Archivo: `data/raw/datos_resenas.csv`

Este archivo contiene una fila por reseña y reúne tanto el contexto de la compra como el texto de la experiencia del cliente. Además, incluye el atributo objetivo `sentimiento`, que será el campo a predecir en un problema de clasificación multiclase.

---

## 3. Diccionario de datos

### 3.1. `datos_clientes.csv`

| Campo | Tipo | Descripción | Dominio / ejemplo |
|---|---|---|---|
| `id_cliente` | entero | Identificador único del cliente. | 1, 2, 3, ... |
| `edad` | numérico | Edad del cliente en años. | 18 a 79 (con valores nulos y atípicos presentes en datos crudos) |
| `genero` | categórico | Género declarado por el cliente. | `Femenino`, `Masculino`, `Otro` |
| `ciudad` | categórico | Ciudad de residencia del cliente. | `Buenos Aires`, `Córdoba`, `La Plata`, `Mendoza`, `Posadas`, `Rosario`, `Salta`, `San Miguel de Tucumán`, `Santa Fe`, `Mar del Plata` |
| `antiguedad_cliente_meses` | entero | Tiempo que el cliente lleva asociado a la empresa, medido en meses. | 1 a 95 |
| `canal_registro` | categórico | Canal por el cual el cliente se registró en la organización. | `Online`, `Tienda física` |

#### Observaciones

- El dataset representa la dimensión de cliente.
- Se usa como base demográfica para analizar si ciertas características de los clientes están asociadas con su experiencia y con el sentimiento de las reseñas.
- El atributo `id_cliente` sirve como clave de enlace con el dataset de reseñas.

### 3.2. `datos_resenas.csv`

| Campo | Tipo | Descripción | Dominio / ejemplo |
|---|---|---|---|
| `id_resena` | entero | Identificador único de cada reseña. | 1, 2, 3, ... |
| `id_cliente` | entero | Cliente asociado a la reseña. Permite unir con `datos_clientes.csv`. | Referencia a `id_cliente` |
| `categoria_producto` | categórico | Categoría del producto comprado. | `Alimentos`, `Belleza`, `Calzado`, `Deportes`, `Electrónica`, `Hogar`, `Indumentaria`, `Libros` |
| `canal_compra` | categórico | Canal a través del cual se realizó la compra. | `Online`, `Offline` |
| `plataforma` | categórico | Plataforma donde se publicó la reseña o donde se realizó la transacción. | `Amazon`, `Falabella`, `Flipkart`, `JioMart`, `Meesho`, `MercadoLibre`, `Nykaa`, `Zepto` |
| `calificacion` | entero | Puntaje otorgado por el cliente a la compra. | 1 a 5 |
| `texto_resena` | texto libre | Comentario escrito por el cliente sobre la experiencia. | Ej.: “Muy conforme con la compra...” |
| `tiempo_respuesta_horas` | numérico | Tiempo de respuesta del servicio postventa, medido en horas. | Valor real con algunos nulos |
| `problema_resuelto` | categórico | Indica si el problema reportado fue resuelto. | `Sí`, `No` |
| `reclamo_formal` | categórico | Indica si hubo reclamo formal por parte del cliente. | `Sí`, `No` |
| `importe_compra` | numérico | Monto de la compra asociada a la reseña. | Valor monetario |
| `fecha_resena` | fecha | Fecha en la que se registró la reseña. | Formato ISO `YYYY-MM-DD` |
| `sentimiento` | categórico | Etiqueta objetivo del problema de análisis de sentimiento. | `Positivo`, `Neutral`, `Negativo` |

#### Observaciones

- Es la tabla de mayor riqueza analítica del escenario.
- Combina información transaccional, de servicio postventa y de texto libre.
- `sentimiento` es el target del proyecto y se obtiene a partir de la experiencia del cliente reflejada en la reseña.

---

## 4. Relación entre ambos datasets

La relación lógica del escenario es la siguiente:

- Un cliente puede aparecer en `datos_clientes.csv` con un único `id_cliente`.
- Ese mismo cliente puede tener múltiples reseñas en `datos_resenas.csv`.
- La unión se hace por `id_cliente`.

Esto permite construir una vista integrada donde se mezclan:

- atributos demográficos del cliente,
- atributos del producto y la compra,
- atributos del servicio postventa,
- características del texto de la reseña,
- y la etiqueta de sentimiento.

---

## 5. Variables clave para el análisis

### 5.1. Variables demográficas

- `edad`
- `genero`
- `ciudad`
- `antiguedad_cliente_meses`
- `canal_registro`

Estas variables describen el perfil del cliente y ayudan a detectar patrones por segmento.

### 5.2. Variables de compra y experiencia

- `categoria_producto`
- `canal_compra`
- `plataforma`
- `calificacion`
- `importe_compra`
- `fecha_resena`

Estas variables contextualizan la transacción y la experiencia de compra.

### 5.3. Variables de atención y servicio

- `tiempo_respuesta_horas`
- `problema_resuelto`
- `reclamo_formal`

Son indicadores del soporte postventa y suelen tener alta relación con el sentimiento final del cliente.

### 5.4. Variable objetivo

- `sentimiento`

La variable objetivo indica el estado emocional o de satisfacción del cliente respecto a la compra. En este problema se trabaja con tres clases:

- `Positivo`
- `Neutral`
- `Negativo`

---

## 6. Calidad de datos y notas de dominio

Los datasets crudos presentan algunas anomalías esperables en datos reales:

- valores faltantes en `edad` y `tiempo_respuesta_horas`,
- edades fuera del rango habitual,
- filas duplicadas de reseñas,
- categorías textuales con distintos niveles de calidad,
- sesgo de clase en la etiqueta objetivo, donde `Positivo` suele ser la categoría más frecuente.

Estas situaciones forman parte del análisis de calidad y deben manejarse antes de entrenar modelos predictivos.

---

## 7. Dataset integrado

En la etapa de preparación de datos, normalmente se integra la información de ambos archivos en un único dataset, utilizando `id_cliente` como llave. A partir de esa unión se suelen crear variables derivadas como:

- `longitud_resena`
- `cantidad_palabras_resena`
- `tiempo_respuesta_dias`

El objetivo es transformar la información original en una estructura más útil para el modelado supervisado.

---

## 8. Resumen del escenario

En síntesis, el problema consiste en predecir el sentimiento de una reseña a partir de:

- información del cliente,
- información de la compra,
- comportamiento de atención al cliente,
- y el texto libre de la reseña.

Es un caso típico de clasificación con datos tabulares y texto, donde la combinación de variables estructuradas y no estructuradas es clave para obtener un modelo útil y explicable.

---

## 9. Referencia de origen

El escenario está inspirado en un problema de análisis de sentimiento para comercio electrónico y comparte la lógica de datasets de reseñas de clientes con un objetivo de clasificación por sentimiento. La estructura de los datos presentada aquí refleja el repositorio del proyecto y la lógica del generador del escenario.
