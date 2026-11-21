# Pronóstico del consumo eléctrico residencial por distrito (ODS 7)

Backend en **Python 3.11 + scikit-learn + FastAPI** del proyecto de aprendizaje automático de
Sistemas Inteligentes (UNMSM, Grupo 6). A partir de los **datos abiertos de ENOSA** (34 meses,
enero de 2024 a octubre de 2026, 19,7 millones de recibos), el proyecto:

- **pronostica el consumo medio mensual de los hogares (tarifa BT5B) de cada distrito** de Piura y
  Tumbes (regresión);
- **agrupa los distritos** según su perfil estacional con **k-means**;
- **detecta distrito-meses atípicos** con **DBSCAN**.

Es la continuación de los proyectos de sistema experto, sistema difuso y algoritmo genético del
grupo, sobre el mismo tema: el uso eficiente de la energía en la vivienda (ODS 7).

## Datos

| Dato | Valor |
|---|---|
| Conjunto de datos | *Consumo energético de clientes Enosa [Distriluz - DLZ]*, Plataforma Nacional de Datos Abiertos ([enlace](https://www.datosabiertos.gob.pe/dataset/consumo-energ%C3%A9tico-de-clientes-enosa-distriluz-dlz)) |
| Publicador | Empresa Regional de Servicio Público de Electricidad Electronoroeste S.A. - ENOSA |
| Licencia | Open Data Commons Open Database License (ODbL) |
| Cobertura | 34 archivos mensuales (CSV, XLSX o ZIP), enero de 2024 a octubre de 2026 |
| Registros | 19 713 482 recibos; 17 766 272 recibos residenciales BT5B activos tras la limpieza |
| Panel | 78 distritos (65 de Piura y 13 de Tumbes) × 34 meses = 2652 distrito-meses |

**Datos personales.** Seis meses (diciembre de 2024, enero y noviembre de 2025, enero,
setiembre y octubre de 2026) se publicaron sin anonimizar. `ml/fuente.py` lee **solo las columnas no
personales**: las columnas 2 a 6 del diccionario (número de servicio, nombre, dirección, tipo y
número de documento) se descartan al leer cada línea y nunca se guardan ni se imprimen. Como varios
CSV no traen encabezado, las columnas se identifican **por posición**. Si un nombre o una dirección
trae comas sin comillas, se leen las 19 últimas columnas, que no son personales. La auditoría
(`datos/limpios/auditoria.json`) solo guarda conteos.

El repositorio incluye el **panel distrito-mes ya agregado** (`datos/procesados/panel.parquet`), que
no contiene información de personas. Los archivos mensuales no se incluyen: se reconstruyen con
`python cli.py descargar` (unos 3,7 GB de descarga).

## Metodología

1. **Preprocesamiento** (`ml/preprocesamiento.py`): clientes residenciales (cartera `C`) con tarifa
   BT5B de Piura y Tumbes; periodos de facturación de 20 a 40 días; atípicos extremos con las cercas
   de Tukey a 3·RIC sobre el logaritmo del consumo (0,23 y 10 773 kWh). Los recibos en cero cuentan
   para la proporción de recibos en cero, no para el consumo medio.
2. **Variables** (`ml/caracteristicas.py`): rezagos de 1, 2, 12 y 13 meses, días facturados, número
   de suministros, proporción de recibos en cero, mes, provincia y grupo k-means.
3. **No supervisado** (`ml/agrupamiento.py`): k-means sobre los perfiles mensuales 2024-2025 (k por
   silueta, con el codo como referencia; para verlos en un plano, los perfiles se proyectan en sus
   dos primeras componentes principales con PCA); DBSCAN sobre `[ln y, ln(y/y₋₁), ln(y/y₋₁₂)]` estandarizado,
   con `min_samples = 2·dimensión` y ε en la rodilla de la curva k-distancia.
4. **Supervisado** (`ml/gradiente.py`, `ml/modelos.py`): regresión lineal propia por **gradiente
   descendiente** comparada con la **ecuación normal**, Ridge, Random Forest y Gradient Boosting, en
   `Pipeline` con `ColumnTransformer` y `GridSearchCV` (k = 5). Referencias ingenuas: mes anterior,
   mismo mes del año anterior y estacional ajustado.
5. **Validación** (`ml/corridas.py`): MAE, RMSE y R² en kWh con dos esquemas:
   - **A**: división aleatoria 80/20 (semilla 42) con validación cruzada k = 5;
   - **B**: prueba temporal, entrenamiento de febrero a diciembre de 2025 y prueba de enero a octubre
     de 2026.

   Se comparan dos formulaciones: **corrida 1**, el nivel del consumo, y **corrida 2**, su variación
   logarítmica respecto del mes anterior, `ln(yₜ/yₜ₋₁)`.

## Resultados

Los resultados (`resultados/resultados.json`) son los del informe y se reproducen exactamente con
`python cli.py corridas`.

| Prueba temporal 2026 (esquema B) | Corrida 1: nivel, MAE (kWh) | Corrida 2: variación, MAE (kWh) |
|---|---:|---:|
| Ingenuo (mes anterior) | 4,29 | 4,29 |
| Ingenuo estacional ajustado | 3,89 | 3,89 |
| Regresión lineal (gradiente descendiente) | 5,77 | 3,64 |
| Ridge | 5,54 | 3,55 |
| Random Forest | 5,96 | **3,19** |
| Gradient Boosting | 5,69 | 3,45 |

- **Corrida 1**: con la división aleatoria todos los modelos superan al ingenuo, pero en la prueba
  temporal ninguno lo hace. El nivel de consumo de 2026 es mayor que el de 2025.
- **Corrida 2**: el **Random Forest** (500 árboles, hojas de 3 o más) obtiene un MAE de 3,19 kWh
  (RMSE 5,20; R² 0,986). Es 25,6 % menos que el ingenuo y 18,0 % menos que el estacional ajustado.
  Es el modelo adoptado.
- **Gradiente descendiente**: con α = 0,1 llega a la solución de la ecuación normal; con α = 0,5
  diverge en la corrida 1.
- **k-means**: k = 3 (silueta 0,587):

  | Grupo | Distritos | Consumo medio (kWh) |
  |---|---:|---:|
  | 1, sierra | 19 | 28,1 |
  | 2 | 41 | 64,2 |
  | 3, ciudades de la costa | 18 | 124,7 |

  En PCA, la primera componente explica el 99,4 % de la varianza de los perfiles: los grupos se
  separan sobre todo por el nivel de consumo, no por la forma de la curva estacional.

- **DBSCAN**: ε = 0,837 y `min_samples` = 6; marca 30 distrito-meses atípicos de 1716 evaluados.

## Estructura

```
ml/                 Aprendizaje automático (no depende de FastAPI)
  config.py           Rutas, semilla y columnas del diccionario de ENOSA
  fuente.py           Descarga y lectura de los archivos mensuales sin datos personales
  preprocesamiento.py Filtros, atípicos y panel distrito-mes
  caracteristicas.py  Rezagos y formulaciones de las corridas 1 y 2
  agrupamiento.py     k-means de distritos y DBSCAN de distrito-meses
  gradiente.py        Regresión lineal por gradiente descendiente y ecuación normal
  modelos.py          Pipelines, búsqueda de hiperparámetros, métricas, ingenuos, importancia
  corridas.py         Corridas del informe y análisis del modelo adoptado
  pronostico.py       Pronóstico del mes siguiente con el modelo adoptado
  figuras.py          Las 13 figuras del informe
core/ schemas/ services/ routers/   Configuración, esquemas Pydantic, servicio y endpoints
cli.py              Línea de comandos: descargar, preparar, corridas, figuras, pronosticar
datos/fuentes/      URLs de los 34 archivos mensuales y del diccionario
datos/procesados/   Panel distrito-mes y resumen del preprocesamiento
resultados/         Resultados de las corridas y figuras
tests/              Pruebas (lectura sin datos personales, panel, modelos, artefactos, API)
```

## Instalación y uso

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements-dev.txt
cp .env.example .env

python cli.py corridas          # k-means, DBSCAN y corridas 1 y 2 (≈ 3 min)
python cli.py figuras           # figuras del informe
python cli.py pronosticar --ubigeo 200101 --dias 31

python cli.py descargar         # opcional: vuelve a descargar los 34 meses (~3,7 GB)
python cli.py preparar          # opcional: reconstruye el panel desde datos/limpios

uvicorn main:app --reload       # http://localhost:8000/docs
pytest                          # pruebas
ruff check . && black --check . # estilo (igual que el CI)
```

Docker: `docker compose up --build`.

## API

Todos los endpoints son `GET` y responden JSON. La documentación interactiva está en `/docs`.

| Endpoint | Descripción |
|---|---|
| `/health` | Estado del servicio |
| `/api/resumen` | Fuente, conteos del preprocesamiento y desempeño del modelo adoptado |
| `/api/distritos` | Distritos con su grupo k-means y su último consumo medio |
| `/api/distritos/{ubigeo}` | Serie mensual, meses atípicos, pronósticos de la prueba 2026 y pronóstico del mes siguiente |
| `/api/grupos` | k-means: codo, silueta, perfil de cada grupo y proyección PCA de los distritos |
| `/api/atipicos` | DBSCAN: parámetros, curva k-distancia, distrito-meses evaluados y atípicos |
| `/api/modelos` | Métricas de las corridas, validación cruzada, gradiente descendiente, importancia de variables y valores reales frente a pronosticados en 2026 |
| `/api/regional` | Serie regional mensual y pronóstico agregado de 2026 |
| `/api/pronostico` | Pronóstico del mes siguiente para los 78 distritos |
| `/api/pronostico/{ubigeo}?dias=31` | Pronóstico de un distrito; `dias` = días facturados supuestos (20 a 40) |

Al iniciar, el servicio reentrena el Random Forest adoptado con todos los distrito-meses que tienen
sus rezagos completos. El pronóstico del mes siguiente supone dos valores que aún no se conocen:

- el número de suministros, que se toma igual al del último mes;
- los días facturados, que son los indicados en `dias` o, si no se indican, los del último mes.

La respuesta declara ambos supuestos.

### Despliegue en Render

1. En Render, elegir **New → Blueprint** y este repositorio. `render.yaml` crea el servicio web en
   el plan gratuito, con Docker.
2. Configurar `CORS_ALLOWED_ORIGINS` con el dominio del frontend, por ejemplo
   `["https://tu-proyecto.vercel.app"]`.

## Licencia

- **Código:** MIT (`LICENSE`).
- **Datos derivados** (`datos/procesados/` y `resultados/`): se distribuyen bajo la licencia
  **ODbL**, como el conjunto de datos de origen. Contienen información de ENOSA, *Consumo energético
  de clientes Enosa [Distriluz - DLZ]*, Plataforma Nacional de Datos Abiertos.
