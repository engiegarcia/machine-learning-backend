"""Filtros y agregación del panel distrito-mes."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.config import CONSERVADAS
from ml.preprocesamiento import construir_panel, normalizar


def _recibos() -> pd.DataFrame:
    base = {
        "PERIODO": "202501",
        "FECHA_CORTE": "20250131",
        "ID_UUNN": "41",
        "UNIDAD_NEGOCIO": "Piura",
        "TARIFA": "BT5B",
        "DEPARTAMENTO": "PIURA",
        "PROVINCIA": "PIURA",
        "DISTRITO": "PIURA",
        "UBIGEO": "200101",
        "CARTERA": "C",
        "FECHA_EMISION": "20250201",
        "FECHA_VENCIMIENTO": "20250215",
        "FECHA_CONSUMO_DESDE": "20241231",
        "FECHA_CONSUMO_HASTA": "20250130",
        "IMPORTE": 50.0,
        "CONSUMO": 100.0,
    }
    variantes = [
        {},
        {"CONSUMO": 60.0},
        {"CONSUMO": 0.0},  # recibo en cero: cuenta para p_cero, no para el consumo medio
        {"CONSUMO": 1e7},  # atípico por encima
        {"CARTERA": "M"},  # no residencial
        {"TARIFA": "BT5C"},  # otra tarifa
        {"DEPARTAMENTO": "LIMA", "UBIGEO": "150101"},  # fuera de la concesión
        {"FECHA_CONSUMO_DESDE": "20250125"},  # 5 días: periodo inválido
        {"CONSUMO": float("nan")},
        {"DISTRITO": "Castilla", "UBIGEO": "200104", "CONSUMO": 80.0},
    ] + [{"CONSUMO": 70.0 + i} for i in range(20)]
    df = pd.DataFrame([{**base, **v} for v in variantes])
    return df[CONSERVADAS]


def test_normalizar():
    assert normalizar("  Sullána  del   Norte ") == "SULLANA DEL NORTE"


def test_conteos_de_cada_paso():
    panel, r = construir_panel(_recibos())
    assert r["filas_totales"] == 30
    assert r["fuera_de_formato_o_concesion"] == 1
    assert r["no_residenciales"] == 1
    assert r["residenciales_otra_tarifa"] == 1
    assert r["residenciales_bt5b"] == 27
    assert r["consumo_nulo"] == 1
    assert r["dias_invalidos"] == 1
    assert r["atipicos_sup"] == 1
    assert r["activos_limpios"] == 23
    assert r["distritos_total"] == 2 and r["celdas_total"] == 2


def test_agregacion_distrito_mes():
    panel, _ = construir_panel(_recibos())
    piura = panel.set_index("UBIGEO").loc["200101"]
    consumos = [100.0, 60.0] + [70.0 + i for i in range(20)]
    assert piura["n_sum"] == 24  # incluye el recibo en cero y el atípico
    assert piura["n_act"] == 22
    assert piura["p_cero"] == pytest.approx(1 / 24)
    assert piura["y"] == pytest.approx(sum(consumos) / len(consumos))
    assert piura["dias"] == pytest.approx(30)
    assert piura["mes"] == 1 and piura["DIST_N"] == "PIURA"
    assert panel.set_index("UBIGEO").loc["200104", "DIST_N"] == "CASTILLA"
