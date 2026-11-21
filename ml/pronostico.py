"""Pronóstico del mes siguiente con el modelo adoptado (corrida 2).

El modelo se reentrena con todos los distrito-meses que tienen rezagos completos y pronostica
el mes posterior al último del panel. Dos entradas del mes pronosticado todavía no se conocen:

* los **días facturados** (``dias_ratio``): por defecto se suponen iguales a los del último mes
  (razón 1); el usuario puede indicar otro valor;
* el **número de suministros** (``log_n``): se usa el del último mes.

Ambos son supuestos del pronóstico y la respuesta de la API los declara.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml import caracteristicas as car
from ml import modelos

F = 2


def entrenar(
    d: pd.DataFrame, params: dict, n_jobs: int = -1, nombre: str = "Random Forest"
) -> Pipeline:
    """Modelo adoptado de la corrida 2 con los hiperparámetros elegidos en la búsqueda."""
    m = modelos.crear(nombre, params)
    if "n_jobs" in m.get_params() and nombre != "XGBoost":
        m.set_params(n_jobs=n_jobs)
    return modelos.pipeline(F, m).fit(d[car.entradas(F)], d[car.objetivo(F)].values)


@dataclass
class Pronostico:
    ubigeo: str
    periodo: str
    ultimo_periodo: str
    ultimo_consumo: float
    dias_ultimo: float
    dias_supuestos: float
    consumo: float
    variacion: float


def siguiente_periodo(periodo: str) -> str:
    anio, mes = int(periodo[:4]), int(periodo[4:])
    return f"{anio + mes // 12}{mes % 12 + 1:02d}"


def filas_siguiente(pn: pd.DataFrame, dias: dict[str, float] | None = None) -> pd.DataFrame:
    """Variables de entrada del mes siguiente al último de cada distrito.

    Args:
        pn: Panel con rezagos (``caracteristicas.agregar_rezagos``) y grupo k-means.
        dias: Días facturados supuestos por ubigeo; si falta, los del último mes.
    """
    pn = pn.sort_values(["UBIGEO", "fecha"])
    ult = pn.groupby("UBIGEO").tail(1).set_index("UBIGEO")
    ant = pn.groupby("UBIGEO").nth(-2).set_index("UBIGEO")
    # mismo mes del año anterior al pronosticado (t + 1 − 12) y el previo (t − 12)
    hace11 = pn.groupby("UBIGEO").nth(-12).set_index("UBIGEO")
    hace12 = pn.groupby("UBIGEO").nth(-13).set_index("UBIGEO")
    dias = dias or {}
    x = pd.DataFrame(index=ult.index)
    x["periodo"] = [siguiente_periodo(p) for p in ult["PERIODO"]]
    x["mes"] = [int(p[4:]) for p in x["periodo"]]
    x["PROV_N"] = ult["PROV_N"]
    x["grupo"] = ult["grupo"]
    x["lag1"] = ult["y"]
    x["ultimo_periodo"] = ult["PERIODO"]
    x["dias_ultimo"] = ult["dias"]
    x["dias_supuestos"] = [dias.get(u, ult.at[u, "dias"]) for u in ult.index]
    x["dl_prev"] = np.log(ult["y"] / ant["y"])
    x["dl_seas"] = np.log(hace11["y"] / hace12["y"])
    x["dias_ratio"] = x["dias_supuestos"] / ult["dias"]
    x["log_n"] = np.log(ult["n_sum"])
    x["p_cero_lag1"] = ult["p_cero"]
    return x


def pronosticar(modelo: Pipeline, x: pd.DataFrame) -> list[Pronostico]:
    """Aplica el modelo a las filas de ``filas_siguiente``."""
    dl = modelo.predict(x[car.entradas(F)])
    consumo = x["lag1"].values * np.exp(dl)
    return [
        Pronostico(
            ubigeo=str(u),
            periodo=str(f.periodo),
            ultimo_periodo=str(f.ultimo_periodo),
            ultimo_consumo=float(f.lag1),
            dias_ultimo=float(f.dias_ultimo),
            dias_supuestos=float(f.dias_supuestos),
            consumo=float(c),
            variacion=float(np.exp(v) - 1),
        )
        for (u, f), c, v in zip(x.iterrows(), consumo, dl, strict=True)
    ]
