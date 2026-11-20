"""Variables de entrada a partir del panel distrito-mes: rezagos y las dos formulaciones.

* Corrida 1 (nivel): el objetivo es el consumo medio ``y`` (kWh).
* Corrida 2 (variación mensual): el objetivo es ``dl = ln(y_t / y_{t-1})``; el pronóstico en
  kWh se recupera como ``y_{t-1} · exp(dl)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FORMULACIONES: dict[int, dict[str, object]] = {
    1: {
        "num": ["lag1", "lag2", "lag12", "dias", "log_n", "p_cero_lag1"],
        "cat": ["mes", "PROV_N", "grupo"],
        "objetivo": "y",
    },
    2: {
        "num": ["dl_prev", "dl_seas", "dias_ratio", "log_n", "p_cero_lag1"],
        "cat": ["mes", "PROV_N", "grupo"],
        "objetivo": "dl",
    },
}


def numericas(f: int) -> list[str]:
    return list(FORMULACIONES[f]["num"])  # type: ignore[arg-type]


def categoricas(f: int) -> list[str]:
    return list(FORMULACIONES[f]["cat"])  # type: ignore[arg-type]


def entradas(f: int) -> list[str]:
    return numericas(f) + categoricas(f)


def objetivo(f: int) -> str:
    return str(FORMULACIONES[f]["objetivo"])


def ordenar(panel: pd.DataFrame) -> pd.DataFrame:
    """Panel ordenado por distrito y fecha, con la columna ``anio``."""
    pn = panel.sort_values(["UBIGEO", "fecha"]).reset_index(drop=True)
    pn["anio"] = pn["fecha"].dt.year
    return pn


def agregar_rezagos(pn: pd.DataFrame) -> pd.DataFrame:
    """Rezagos por distrito y variables derivadas de ambas formulaciones.

    El panel debe venir ordenado por distrito y fecha (``ordenar``) y sin meses faltantes.
    """
    pn = pn.copy()
    g = pn.groupby("UBIGEO")
    for k in (1, 2, 12, 13):
        pn[f"lag{k}"] = g["y"].shift(k)
    pn["p_cero_lag1"] = g["p_cero"].shift(1)
    pn["dias_lag1"] = g["dias"].shift(1)
    pn["log_n"] = np.log(pn["n_sum"])
    pn["dl"] = np.log(pn["y"] / pn["lag1"])
    pn["dl_prev"] = np.log(pn["lag1"] / pn["lag2"])
    pn["dl_seas"] = np.log(pn["lag12"] / pn["lag13"])
    pn["dias_ratio"] = pn["dias"] / pn["dias_lag1"]
    return pn


def supervisado(pn: pd.DataFrame) -> pd.DataFrame:
    """Filas con todos los rezagos disponibles (desde el mes 14 de cada distrito)."""
    return pn.dropna(subset=["lag13", "dias_lag1", "p_cero_lag1"]).reset_index(drop=True)


def a_kwh(f: int, pred: np.ndarray, filas: pd.DataFrame) -> np.ndarray:
    """Pronóstico en kWh: directo (corrida 1) o ``lag1 · exp(pred)`` (corrida 2)."""
    return pred if f == 1 else filas["lag1"].values * np.exp(pred)
