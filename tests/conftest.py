"""Fixtures compartidas. La configuración se fija por entorno para que las pruebas sean
herméticas (no dependen de un ``.env`` local)."""

from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ.update(
    {
        "LOG_LEVEL": "WARNING",
        "LOG_JSON": "false",
        "ENVIRONMENT": "dev",
        "CORS_ALLOWED_ORIGINS": '["http://localhost:3000"]',
        # Bosque pequeño: las pruebas de la API no necesitan los 500 árboles del modelo.
        "ARBOLES_PRONOSTICO": "20",
        "N_JOBS": "1",
    }
)


def _fila(**cambios: str) -> list[str]:
    """Una línea sintética de 25 campos con el orden del diccionario. No son datos reales."""
    base = {
        "FECHA_CORTE": "20250131",
        "NRO_SERVICIO": "99999999",
        "NOMBRE_SERVICIO": "PERSONA DE PRUEBA",
        "DIRECCION": "CALLE FICTICIA 123",
        "ID_TIPO_IDENTIDAD": "1",
        "NRO_IDENTIDAD": "00000001",
        "ID_EMPRESA": "4",
        "NOMBRE_EMPRESA": "ENOSA",
        "ID_UUNN": "41",
        "UNIDAD_NEGOCIO": "PIURA",
        "PERIODO": "202501",
        "TARIFA": "BT5B",
        "DEPARTAMENTO": "PIURA",
        "PROVINCIA": "PIURA",
        "DISTRITO": "PIURA",
        "UBIGEO": "200101",
        "ID_TIPO_DOC_FAC": "1",
        "NRO_DOC_FAC": "123",
        "CARTERA": "C",
        "FECHA_VENCIMIENTO": "20250215",
        "FECHA_EMISION": "20250201",
        "FECHA_CONSUMO_DESDE": "20241231",
        "FECHA_CONSUMO_HASTA": "20250130",
        "IMPORTE": "1,234.50",
        "CONSUMO": "85",
    }
    base.update(cambios)
    return list(base.values())


@pytest.fixture()
def fila():
    return _fila


@pytest.fixture()
def panel_sintetico() -> pd.DataFrame:
    """Panel de 4 distritos × 26 meses con estacionalidad y dos niveles de consumo."""
    rng = np.random.default_rng(0)
    filas = []
    fechas = pd.date_range("2024-01-01", periods=26, freq="MS")
    for i, (ubigeo, nivel) in enumerate(
        [("200101", 120), ("200104", 115), ("200201", 30), ("200202", 28)]
    ):
        for t, f in enumerate(fechas):
            y = nivel * (1 + 0.15 * np.sin(2 * np.pi * (f.month - 3) / 12)) * (1 + 0.002 * t)
            filas.append(
                {
                    "UBIGEO": ubigeo,
                    "PERIODO": f.strftime("%Y%m"),
                    "n_sum": 1000 + 10 * i + t,
                    "p_cero": 0.05,
                    "dias": 30.0 + rng.uniform(-1, 1),
                    "n_act": 950,
                    "y": y * (1 + rng.normal(0, 0.01)),
                    "mediana": y * 0.8,
                    "DIST_N": f"DISTRITO {i}",
                    "PROV_N": "PIURA" if i < 2 else "AYABACA",
                    "DEP_N": "PIURA",
                    "fecha": f,
                    "mes": f.month,
                }
            )
    return pd.DataFrame(filas)


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """``TestClient`` con el ``lifespan`` activo, sobre el panel y los resultados del repo."""
    from core.config import get_settings
    from main import create_app

    get_settings.cache_clear()
    with TestClient(create_app()) as cliente:
        yield cliente
