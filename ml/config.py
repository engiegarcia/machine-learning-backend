"""Constantes del proyecto: rutas, semilla y columnas del diccionario de ENOSA."""

from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_FUENTES = RAIZ / "datos" / "fuentes"
DIR_LIMPIOS = RAIZ / "datos" / "limpios"
DIR_PROCESADOS = RAIZ / "datos" / "procesados"
DIR_RESULTADOS = RAIZ / "resultados"
URLS = DIR_FUENTES / "urls.txt"
PANEL = DIR_PROCESADOS / "panel.parquet"
RESUMEN_PREPROCESAMIENTO = DIR_PROCESADOS / "preprocesamiento.json"
RESULTADOS = DIR_RESULTADOS / "resultados.json"
FIGURAS = DIR_RESULTADOS / "figuras"

SEMILLA = 42

# Las 25 columnas del diccionario de datos de ENOSA, en orden.
DICCIONARIO = [
    "FECHA_CORTE",
    "NRO_SERVICIO",
    "NOMBRE_SERVICIO",
    "DIRECCION",
    "ID_TIPO_IDENTIDAD",
    "NRO_IDENTIDAD",
    "ID_EMPRESA",
    "NOMBRE_EMPRESA",
    "ID_UUNN",
    "UNIDAD_NEGOCIO",
    "PERIODO",
    "TARIFA",
    "DEPARTAMENTO",
    "PROVINCIA",
    "DISTRITO",
    "UBIGEO",
    "ID_TIPO_DOC_FAC",
    "NRO_DOC_FAC",
    "CARTERA",
    "FECHA_VENCIMIENTO",
    "FECHA_EMISION",
    "FECHA_CONSUMO_DESDE",
    "FECHA_CONSUMO_HASTA",
    "IMPORTE",
    "CONSUMO",
]

# Columnas 2 a 6 del diccionario: identifican a la persona. Nunca se conservan.
PERSONALES = DICCIONARIO[1:6]

# Las 19 últimas columnas (7 a 25) no son personales y no contienen comas libres, así que se
# pueden leer desde el final de la línea aunque un nombre o una dirección tenga comas sin comillas.
COLA = DICCIONARIO[6:]

# Nombres usados en los archivos de 2024 que traen encabezado (24 columnas, sin DIRECCION).
ALIAS_2024 = {
    "FECHACORTE": "FECHA_CORTE",
    "IDNROSERVICIO": "NRO_SERVICIO",
    "NOMBRENROSERVICIO": "NOMBRE_SERVICIO",
    "IDTIPOIDENTIDAD": "ID_TIPO_IDENTIDAD",
    "NROIDENTIDAD": "NRO_IDENTIDAD",
    "IDEMPRESA": "ID_EMPRESA",
    "NOMBREEMPRESA": "NOMBRE_EMPRESA",
    "IDUUNN": "ID_UUNN",
    "NOMBREUNIDADNEGOCIO": "UNIDAD_NEGOCIO",
    "IDTIPODOCUMENTO": "ID_TIPO_DOC_FAC",
    "NRODOCUMENTO": "NRO_DOC_FAC",
    "FECHAVENCIMIENTO": "FECHA_VENCIMIENTO",
    "FECHAEMISION": "FECHA_EMISION",
    "FECHACONSUMODESDE": "FECHA_CONSUMO_DESDE",
    "FECHACONSUMOHASTA": "FECHA_CONSUMO_HASTA",
    "CONSUMOSCALCULADOS": "CONSUMO",
}

# Columnas no personales que se guardan de cada mes.
CONSERVADAS = [
    "PERIODO",
    "FECHA_CORTE",
    "ID_UUNN",
    "UNIDAD_NEGOCIO",
    "TARIFA",
    "DEPARTAMENTO",
    "PROVINCIA",
    "DISTRITO",
    "UBIGEO",
    "CARTERA",
    "FECHA_EMISION",
    "FECHA_VENCIMIENTO",
    "FECHA_CONSUMO_DESDE",
    "FECHA_CONSUMO_HASTA",
    "IMPORTE",
    "CONSUMO",
]

MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Set", "Oct", "Nov", "Dic"]
