"""Preprocesamiento: de los recibos mensuales de ENOSA al panel distrito-mes.

Pasos (los conteos de cada uno quedan en el resumen):

1. registros con periodo, ubigeo y departamento válidos (Piura o Tumbes);
2. clientes residenciales (cartera común, ``C``) con tarifa BT5B;
3. consumo no nulo ni negativo y periodo de facturación de 20 a 40 días;
4. suministros activos (consumo > 0) sin valores atípicos extremos, con las cercas de Tukey
   ``Q1 − 3·RIC`` y ``Q3 + 3·RIC`` sobre el logaritmo del consumo;
5. agregación por distrito (ubigeo) y mes: suministros, proporción de recibos en cero, días
   facturados y consumo medio de los suministros activos (``y``, variable objetivo).
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RIC_TUKEY = 3.0


def normalizar(texto: object) -> str:
    """Mayúsculas sin tildes y con espacios simples (los nombres varían entre meses)."""
    s = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return " ".join(s.upper().split())


def cargar_limpios(directorio: Path) -> pd.DataFrame:
    """Une los archivos ``enosa_AAAAMM.parquet`` (solo columnas no personales)."""
    archivos = sorted(directorio.glob("enosa_*.parquet"))
    if not archivos:
        raise FileNotFoundError(f"no hay archivos enosa_*.parquet en {directorio}")
    return pd.concat([pd.read_parquet(f) for f in archivos], ignore_index=True)


def _describir(s: pd.Series, percentiles: list[float] | None = None) -> dict[str, float]:
    return {k: float(v) for k, v in s.describe(percentiles=percentiles).items()}


def construir_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filtra los recibos y los agrega en el panel distrito-mes.

    Returns:
        ``(panel, resumen)``: una fila por distrito y mes, y los conteos de cada paso.
    """
    r: dict[str, Any] = {"filas_totales": len(df)}
    per = df["PERIODO"]
    r["meses"] = sorted(per[per.str.fullmatch(r"20\d{4}", na=False)].unique().tolist())

    ok = (
        df["PERIODO"].str.fullmatch(r"202[4-6](0[1-9]|1[0-2])", na=False)
        & df["UBIGEO"].str.fullmatch(r"(20|24)\d{4}", na=False)
        & df["DEPARTAMENTO"].map(normalizar).isin(["PIURA", "TUMBES"])
    )
    r["fuera_de_formato_o_concesion"] = int((~ok).sum())
    df = df[ok]
    r["no_residenciales"] = int((df["CARTERA"] != "C").sum())
    df = df[df["CARTERA"] == "C"]
    r["residenciales_otra_tarifa"] = int((df["TARIFA"] != "BT5B").sum())
    df = df[df["TARIFA"] == "BT5B"]
    r["residenciales_bt5b"] = len(df)
    r["consumo_nulo"] = int(df["CONSUMO"].isna().sum())
    df = df[df["CONSUMO"].notna()]
    r["consumo_negativo"] = int((df["CONSUMO"] < 0).sum())
    df = df[df["CONSUMO"] >= 0]
    dias = (
        pd.to_datetime(df["FECHA_CONSUMO_HASTA"], format="%Y%m%d", errors="coerce")
        - pd.to_datetime(df["FECHA_CONSUMO_DESDE"], format="%Y%m%d", errors="coerce")
    ).dt.days
    df = df.assign(DIAS=dias)
    r["dias_invalidos"] = int((dias.isna() | (dias < 20) | (dias > 40)).sum())
    df = df[dias.between(20, 40)]
    r["dias_describe"] = _describir(df["DIAS"])

    cons = df["CONSUMO"]
    r["consumo_describe"] = _describir(cons, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
    r["ceros_proporcion"] = float((cons == 0).mean())
    r["hist_consumo"] = np.histogram(cons.clip(upper=600), bins=60, range=(0, 600))[0].tolist()

    act = df[df["CONSUMO"] > 0]
    lg = np.log(act["CONSUMO"])
    q1, q3 = lg.quantile([0.25, 0.75])
    sup, inf = q3 + RIC_TUKEY * (q3 - q1), q1 - RIC_TUKEY * (q3 - q1)
    r["tukey_log"] = {
        "q1": float(q1),
        "q3": float(q3),
        "limite_sup_kwh": float(np.exp(sup)),
        "limite_inf_kwh": float(np.exp(inf)),
    }
    r["atipicos_sup"] = int((lg > sup).sum())
    r["atipicos_inf"] = int((lg < inf).sum())
    act = act[(lg <= sup) & (lg >= inf)]
    r["activos_limpios"] = len(act)

    df = df.assign(
        DIST_N=df["DISTRITO"].map(normalizar),
        PROV_N=df["PROVINCIA"].map(normalizar),
        DEP_N=df["DEPARTAMENTO"].map(normalizar),
        UUNN_N=df["UNIDAD_NEGOCIO"].map(normalizar),
    )
    nombres = df.groupby("UBIGEO")[["DIST_N", "PROV_N", "DEP_N", "UUNN_N"]].agg(
        lambda s: s.value_counts().idxmax()
    )

    g_todos = df.groupby(["UBIGEO", "PERIODO"])
    base = pd.DataFrame(
        {
            "n_sum": g_todos.size(),
            "p_cero": g_todos["CONSUMO"].apply(lambda s: float((s == 0).mean())),
            "dias": g_todos["DIAS"].mean(),
        }
    )
    g = act.groupby(["UBIGEO", "PERIODO"])
    agg = pd.DataFrame(
        {
            "n_act": g.size(),
            "y": g["CONSUMO"].mean(),
            "mediana": g["CONSUMO"].median(),
            "kwh_total": g["CONSUMO"].sum(),
            "importe_medio": g["IMPORTE"].mean(),
        }
    )
    panel = (
        base.join(agg, how="left").reset_index().merge(nombres, left_on="UBIGEO", right_index=True)
    )
    panel["fecha"] = pd.to_datetime(panel["PERIODO"] + "01")
    panel["mes"] = panel["fecha"].dt.month
    r["distritos_total"] = int(panel["UBIGEO"].nunique())
    r["celdas_total"] = len(panel)
    cob = panel.groupby("UBIGEO").agg(meses=("PERIODO", "nunique"), n_min=("n_act", "min"))
    r["cobertura_meses"] = {
        int(k): int(v) for k, v in cob["meses"].value_counts().sort_index().items()
    }
    r["n_act_min_describe"] = _describir(cob["n_min"])

    reg = act.groupby("PERIODO")["CONSUMO"].agg(["mean", "median", "size", "sum"])
    r["serie_regional"] = {
        k: {
            "media": float(f["mean"]),
            "mediana": float(f["median"]),
            "n": int(f["size"]),
            "gwh": float(f["sum"]) / 1e6,
        }
        for k, f in reg.iterrows()
    }
    r["filas_por_mes"] = {k: int(v) for k, v in df.groupby("PERIODO").size().items()}
    return panel, r
