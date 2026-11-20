"""Lectura de los archivos de ENOSA: columnas por posición y sin datos personales."""

from __future__ import annotations

import csv
import io
import zipfile

import pandas as pd
import pytest

from ml.config import CONSERVADAS, DICCIONARIO, PERSONALES
from ml.fuente import FormatoNoReconocidoError, leer_archivo, leer_filas, urls_mensuales

PERSONALES_FICTICIOS = ("99999999", "PERSONA DE PRUEBA", "CALLE FICTICIA", "00000001")


def _sin_datos_personales(df: pd.DataFrame) -> None:
    assert not set(PERSONALES) & set(df.columns)
    texto = df.astype(str).to_csv()
    for valor in PERSONALES_FICTICIOS:
        assert valor not in texto


def test_sin_encabezado_por_posicion(fila):
    lec = leer_filas([fila(), fila(CONSUMO="0", UBIGEO="200104")])
    assert list(lec.datos.columns) == CONSERVADAS
    assert len(lec.datos) == 2 and not lec.con_encabezado
    assert lec.periodo == "202501"
    assert lec.datos["CONSUMO"].tolist() == [85, 0]
    assert lec.datos["IMPORTE"].iloc[0] == pytest.approx(1234.5)
    assert lec.nombres_sin_anonimizar == 2
    _sin_datos_personales(lec.datos)


def test_comas_sin_comillas_se_leen_desde_el_final(fila):
    campos = fila()
    desplazada = campos[:3] + ["MZ A", "LOTE 4"] + campos[3:]  # dirección partida por comas
    lec = leer_filas([fila(), desplazada])
    assert len(lec.datos) == 2 and lec.lineas_descartadas == 0
    assert lec.campos_por_linea == {25: 1, 27: 1}
    pd.testing.assert_series_equal(lec.datos.iloc[0], lec.datos.iloc[1], check_names=False)
    _sin_datos_personales(lec.datos)


def test_linea_incompleta_o_desplazada_invalida_se_descarta(fila):
    corta = fila()[:20]
    larga_invalida = fila()[:-15] + ["X"] + fila()[-15:]
    larga_invalida[-15] = "PERIODO?"
    lec = leer_filas([fila(), corta, larga_invalida])
    assert len(lec.datos) == 1 and lec.lineas_descartadas == 2


def test_anonimizado_no_cuenta_como_dato_personal(fila):
    lec = leer_filas([fila(NOMBRE_SERVICIO="XXXXXXXX", NRO_IDENTIDAD="XXXXXXXX")])
    assert lec.nombres_sin_anonimizar == 0


def test_encabezado_2024_con_alias(fila):
    alias = {
        "FECHA_CORTE": "FECHACORTE",
        "NRO_SERVICIO": "IDNROSERVICIO",
        "NOMBRE_SERVICIO": "NOMBRENROSERVICIO",
        "ID_TIPO_IDENTIDAD": "IDTIPOIDENTIDAD",
        "NRO_IDENTIDAD": "NROIDENTIDAD",
        "ID_EMPRESA": "IDEMPRESA",
        "NOMBRE_EMPRESA": "NOMBREEMPRESA",
        "ID_UUNN": "IDUUNN",
        "UNIDAD_NEGOCIO": "NOMBREUNIDADNEGOCIO",
        "CONSUMO": "CONSUMOSCALCULADOS",
    }
    sin_dir = [i for i, c in enumerate(DICCIONARIO) if c != "DIRECCION"]
    encabezado = [alias.get(DICCIONARIO[i], DICCIONARIO[i]) for i in sin_dir]
    datos = [fila()[i] for i in sin_dir]
    lec = leer_filas([encabezado, datos])
    assert lec.con_encabezado and len(lec.datos) == 1
    assert lec.datos.at[0, "UBIGEO"] == "200101"
    assert lec.datos.at[0, "CONSUMO"] == 85
    _sin_datos_personales(lec.datos)


def test_encabezado_sin_columnas_del_diccionario():
    with pytest.raises(FormatoNoReconocidoError):
        leer_filas([["FECHA", "OTRA"], ["1", "2"]])


def test_archivo_vacio():
    with pytest.raises(FormatoNoReconocidoError):
        leer_filas([])


def _csv(filas, sep=","):
    buf = io.StringIO()
    csv.writer(buf, delimiter=sep).writerows(filas)
    return buf.getvalue().encode("latin-1")


def test_csv_zip_y_xlsx(tmp_path, fila):
    filas = [fila(), fila(DISTRITO="CASTILLA", UBIGEO="200104")]
    (tmp_path / "mes.csv").write_bytes(_csv(filas, sep=";"))
    with zipfile.ZipFile(tmp_path / "mes.zip", "w") as z:
        z.writestr("interno.csv", _csv(filas))
    pd.DataFrame(filas).to_excel(tmp_path / "mes.xlsx", header=False, index=False)
    lecturas = [leer_archivo(tmp_path / n) for n in ("mes.csv", "mes.zip", "mes.xlsx")]
    for lec in lecturas:
        assert lec.datos["UBIGEO"].tolist() == ["200101", "200104"]
        assert lec.datos["CONSUMO"].tolist() == [85, 85]
        _sin_datos_personales(lec.datos)


def test_urls_omiten_el_diccionario(tmp_path):
    f = tmp_path / "urls.txt"
    f.write_text("https://x/Diccionario_datos.xlsx\n\nhttps://x/Enero_2025.csv\n", encoding="utf8")
    assert urls_mensuales(f) == ["https://x/Enero_2025.csv"]


def test_lista_oficial_tiene_34_meses():
    from ml.config import URLS

    assert len(urls_mensuales(URLS)) == 34
