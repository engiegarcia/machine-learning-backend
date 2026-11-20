"""Lectura y descarga de los archivos mensuales de ENOSA sin conservar datos personales.

ENOSA publica un archivo por mes en la Plataforma Nacional de Datos Abiertos, en CSV, XLSX o
ZIP. Varios meses no traen fila de encabezado y seis se publicaron sin anonimizar (con nombres,
DNI y direcciones). Por eso:

* las columnas se identifican por **posición** (diccionario oficial de 25 columnas) cuando no
  hay encabezado, y por nombre cuando lo hay;
* de cada línea solo se copian las columnas no personales (``config.CONSERVADAS``); los campos
  personales nunca salen de la función que lee la línea, no se guardan ni se imprimen;
* si un nombre o una dirección trae comas sin comillas, la línea tiene más campos de los
  esperados: se leen entonces las 19 últimas columnas, que no son personales;
* la auditoría (``Lectura``) solo contiene conteos.
"""

from __future__ import annotations

import csv
import io
import re
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ml.config import ALIAS_2024, COLA, CONSERVADAS, DICCIONARIO

# El portal responde HTTP 418 al agente de usuario por defecto de Python.
AGENTE = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
_MASCARA = re.compile(r"^.?X{4,}.?.?$|^X+$|^0*$|^$")
_PERIODO = re.compile(r"\d{6}")
_POS_COLA = {c: i for i, c in enumerate(COLA)}


class FormatoNoReconocidoError(ValueError):
    """El encabezado del archivo no corresponde al diccionario de ENOSA."""


@dataclass
class Lectura:
    """Columnas no personales de un mes y conteos de la lectura (sin valores personales)."""

    datos: pd.DataFrame
    periodo: str
    con_encabezado: bool
    lineas_descartadas: int = 0
    campos_por_linea: dict[int, int] = field(default_factory=dict)
    nombres_sin_anonimizar: int = 0

    def auditoria(self) -> dict[str, object]:
        """Resumen serializable: solo conteos."""
        return {
            "periodo": self.periodo,
            "filas": len(self.datos),
            "con_encabezado": self.con_encabezado,
            "lineas_descartadas": self.lineas_descartadas,
            "campos_por_linea": {str(k): v for k, v in sorted(self.campos_por_linea.items())},
            "nombres_sin_anonimizar": self.nombres_sin_anonimizar,
        }


def _es_encabezado(fila: Sequence[str]) -> bool:
    return bool(fila) and str(fila[0]).strip().upper().startswith("FECHA")


def leer_filas(filas: Iterable[Sequence[str]]) -> Lectura:
    """Convierte las filas de un archivo mensual en un DataFrame de columnas no personales.

    Args:
        filas: Filas como secuencias de texto, con o sin encabezado en la primera.

    Raises:
        FormatoNoReconocidoError: Si el encabezado no trae las columnas del diccionario o si
            ninguna línea tiene un formato válido.
    """
    it = iter(filas)
    primera = next(it, None)
    if primera is None:
        raise FormatoNoReconocidoError("archivo vacío")

    if _es_encabezado(primera):
        nombres = [ALIAS_2024.get(n, n) for n in (str(c).strip().upper() for c in primera)]
        faltan = [c for c in CONSERVADAS if c not in nombres]
        if faltan:
            raise FormatoNoReconocidoError(f"faltan columnas: {faltan}")
        ancho = len(nombres)
        idx = [nombres.index(c) for c in CONSERVADAS]
        i_nombre = nombres.index("NOMBRE_SERVICIO") if "NOMBRE_SERVICIO" in nombres else None
        cola_valida = nombres[-len(COLA) :] == COLA
        resto: Iterator[Sequence[str]] = it
        con_encabezado = True
    else:
        ancho = len(DICCIONARIO)
        idx = [DICCIONARIO.index(c) for c in CONSERVADAS]
        i_nombre = DICCIONARIO.index("NOMBRE_SERVICIO")
        cola_valida = True
        resto = _encadenar(primera, it)
        con_encabezado = False

    idx_cola = [_POS_COLA.get(c) for c in CONSERVADAS]  # None solo para FECHA_CORTE
    filas_ok: list[list[str]] = []
    largos: dict[int, int] = {}
    descartadas = sin_mascara = 0
    for campos in resto:
        if not campos or _es_encabezado(campos):
            continue
        n = len(campos)
        largos[n] = largos.get(n, 0) + 1
        if n == ancho:
            filas_ok.append([campos[i] for i in idx])
            if i_nombre is not None and not _MASCARA.match(str(campos[i_nombre]).strip()):
                sin_mascara += 1
        elif n > ancho and cola_valida and _PERIODO.fullmatch(str(campos[-len(COLA) + 4]).strip()):
            cola = campos[-len(COLA) :]
            filas_ok.append([campos[0] if j is None else cola[j] for j in idx_cola])
            sin_mascara += 1  # una línea desplazada por comas trae nombre o dirección en claro
        else:
            descartadas += 1
    if not filas_ok:
        raise FormatoNoReconocidoError("ninguna línea con el formato del diccionario")

    df = pd.DataFrame(filas_ok, columns=CONSERVADAS)
    del filas_ok
    for c in ("IMPORTE", "CONSUMO"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    validos = df["PERIODO"].astype(str)
    validos = validos[validos.str.fullmatch(r"\d{6}", na=False)]
    periodo = str(validos.value_counts().idxmax()) if len(validos) else "desconocido"
    return Lectura(df, periodo, con_encabezado, descartadas, largos, sin_mascara)


def _encadenar(primera: Sequence[str], resto: Iterator[Sequence[str]]) -> Iterator[Sequence[str]]:
    yield primera
    yield from resto


def _texto(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def filas_csv(raw: bytes) -> Iterator[list[str]]:
    """Filas de un CSV (codificación y separador detectados)."""
    texto = _texto(raw)
    primera = texto[: texto.find("\n") if "\n" in texto else len(texto)]
    # El separador es el que parte la primera línea en más campos (24 o 25 en ENOSA).
    sep = max(",;|\t", key=lambda d: len(next(csv.reader([primera], delimiter=d), [])))
    yield from csv.reader(io.StringIO(texto), delimiter=sep)


def filas_xlsx(raw: bytes) -> Iterator[list[str]]:
    """Filas de la primera hoja de un XLSX, como texto."""
    hoja = pd.read_excel(io.BytesIO(raw), header=None, dtype=str)
    for fila in hoja.itertuples(index=False, name=None):
        yield ["" if pd.isna(v) else str(v) for v in fila]


def leer_archivo(ruta: Path) -> Lectura:
    """Lee un archivo mensual de ENOSA (CSV, XLSX o ZIP con uno de ellos dentro)."""
    nombre, raw = ruta.name.lower(), ruta.read_bytes()
    if nombre.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            interno = next(n for n in z.namelist() if n.lower().endswith((".csv", ".xlsx", ".txt")))
            nombre, raw = interno.lower(), z.read(interno)
    filas = filas_xlsx(raw) if nombre.endswith(".xlsx") else filas_csv(raw)
    return leer_filas(filas)


def urls_mensuales(archivo: Path) -> list[str]:
    """URLs de los archivos mensuales (se omite el diccionario de datos)."""
    lineas = archivo.read_text(encoding="utf8").splitlines()
    return [u.strip() for u in lineas if u.strip() and "diccionario" not in u.lower()]


def descargar_mes(url: str, destino: Path, timeout: int = 600) -> dict[str, object]:
    """Descarga un mes, guarda sus columnas no personales y borra el archivo original.

    Returns:
        La auditoría del mes (conteos), con el tamaño descargado y la duración.
    """
    t0 = time.time()
    nombre = urllib.parse.unquote(url.rsplit("/", 1)[1])
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / nombre
        req = urllib.request.Request(url, headers={"User-Agent": AGENTE})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(ruta, "wb") as f:
            while bloque := r.read(1 << 20):
                f.write(bloque)
        mb = ruta.stat().st_size / 1e6
        lectura = leer_archivo(ruta)
    destino.mkdir(parents=True, exist_ok=True)
    lectura.datos.to_parquet(destino / f"enosa_{lectura.periodo}.parquet", index=False)
    return {
        **lectura.auditoria(),
        "archivo": nombre,
        "mb": round(mb, 1),
        "segundos": round(time.time() - t0, 1),
    }
