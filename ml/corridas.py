"""Corridas del informe sobre el panel distrito-mes (ENOSA, BT5B residencial).

* No supervisado: k-means de distritos y DBSCAN de distrito-meses atípicos.
* Supervisado: corrida 1 (nivel) y corrida 2 (variación logarítmica mensual), cada una con
  dos esquemas de validación:

  - A: división aleatoria 80/20 (semilla 42) con validación cruzada k = 5;
  - B: prueba temporal, entrenamiento feb-dic 2025 y prueba ene-oct 2026.

* Modelo adoptado: el de menor MAE, entre los que tienen búsqueda de hiperparámetros, en la
  corrida 2, esquema B; se analiza su error por grupo, por mes y la importancia de sus variables.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml import caracteristicas as car
from ml import modelos
from ml.agrupamiento import (
    dbscan_atipicos,
    jerarquico_distritos,
    kmeans_distritos,
    proyeccion_pca,
    resumen_grupos,
)
from ml.config import SEMILLA

logger = logging.getLogger(__name__)

CANDIDATOS = tuple(modelos.REJILLAS)
ANIO_PRUEBA = 2026


def preparar(panel: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    """Panel ordenado, con el grupo k-means de cada distrito y los rezagos."""
    pn = car.ordenar(panel)
    km = kmeans_distritos(pn)
    pn = pn.merge(km.grupo, left_on="UBIGEO", right_index=True)
    return car.agregar_rezagos(pn), km


def esquemas(d: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    tr_a, te_a = train_test_split(d, test_size=0.2, random_state=SEMILLA)
    m = d["anio"] < ANIO_PRUEBA
    return {"A": (tr_a, te_a), "B": (d[m], d[~m])}


def _no_supervisado(pn: pd.DataFrame, km: Any, r: dict[str, Any]) -> None:
    r["kmeans_codo"], r["kmeans_silueta"], r["kmeans_k"] = km.codo, km.silueta, km.k
    r["kmeans_grupos"] = resumen_grupos(km, pn)
    r["kmeans_asignacion"] = {u: int(g) for u, g in km.grupo.items()}
    r["kmeans_pca"] = proyeccion_pca(km)
    r["jerarquico"] = jerarquico_distritos(km)

    db = dbscan_atipicos(pn)
    d, ru = db.filas, db.ruido
    r["dbscan_kdist"] = [float(v) for v in db.kdist]
    r["dbscan"] = {
        "eps": db.eps,
        "min_samples": db.min_samples,
        "n": len(d),
        "periodos": [d.PERIODO.min(), d.PERIODO.max()],
        "clusters": db.n_clusters,
        "ruido": len(ru),
    }
    orden = ru.assign(a=(ru.y / ru.lag1 - 1).abs()).sort_values("a", ascending=False)
    r["dbscan_atipicos"] = [
        {
            "ubigeo": f.UBIGEO,
            "distrito": f.DIST_N,
            "provincia": f.PROV_N,
            "periodo": f.PERIODO,
            "y": float(f.y),
            "lag1": float(f.lag1),
            "lag12": float(f.lag12),
            "var_mes": float(f.y / f.lag1 - 1),
            "var_anio": float(f.y / f.lag12 - 1),
            "n_act": int(f.n_act),
            "grupo": int(f.grupo),
        }
        for f in orden.itertuples()
    ]
    r["dbscan_por_periodo"] = ru["PERIODO"].value_counts().sort_index().to_dict()
    r["dbscan_por_distrito"] = ru["DIST_N"].value_counts().to_dict()
    r["dbscan_puntos"] = {
        "x": [float(v) for v in db.z[:, 1]],
        "y": [float(v) for v in db.z[:, 2]],
        "lab": [int(v) for v in d["db"]],
        "ubigeo": d["UBIGEO"].tolist(),
        "periodo": d["PERIODO"].tolist(),
    }


def comparar_errores(
    err: dict[str, np.ndarray], base: str, grupos: np.ndarray, repeticiones: int = 2000
) -> dict[str, dict[str, float]]:
    """Diferencia de MAE de ``base`` frente a cada modelo, con bootstrap por distrito.

    Se remuestrean distritos completos (con sus diez meses de 2026) para respetar la
    dependencia entre los meses de un mismo distrito. Diferencias negativas: ``base`` tiene
    menor error. Devuelve la diferencia observada, el intervalo del 95 % y la proporción de
    réplicas en que ``base`` tiene menor MAE.
    """
    rng = np.random.default_rng(SEMILLA)
    ids = np.unique(grupos)
    pos = {u: np.flatnonzero(grupos == u) for u in ids}
    muestras = [
        np.concatenate([pos[u] for u in rng.choice(ids, len(ids))]) for _ in range(repeticiones)
    ]
    out = {}
    for nombre, e in err.items():
        if nombre == base:
            continue
        d = np.array([err[base][s].mean() - e[s].mean() for s in muestras])
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[nombre] = {
            "diferencia": float(err[base].mean() - e.mean()),
            "ic_inferior": float(lo),
            "ic_superior": float(hi),
            "p_menor": float(np.mean(d < 0)),
        }
    return out


def _modelo_adoptado(res_b: dict[str, Any], te: pd.DataFrame, r: dict[str, Any]) -> None:
    mejor = min(CANDIDATOS, key=lambda k: res_b[k]["test"]["MAE"])
    r["mejor"] = mejor
    y = te["y"].values
    err = {k: np.abs(res_b[k]["_pred"] - y) for k in CANDIDATOS}
    err["Ingenuo (mes anterior)"] = np.abs(te["lag1"].values - y)
    r["comparacion_adoptado"] = comparar_errores(err, mejor, te["UBIGEO"].values)
    r["importancia"] = modelos.importancia_permutacion(res_b[mejor]["_model"], 2, te)
    cols = ["UBIGEO", "DIST_N", "PROV_N", "PERIODO", "y", "lag1", "lag12", "n_act", "grupo"]
    t = te[cols].copy()
    t["pred"] = res_b[mejor]["_pred"]
    t["pred_ingenuo"] = t["lag1"]
    t["err"] = t["pred"] - t["y"]
    a, ai = t.err.abs(), (t.lag1 - t.y).abs()
    r["pred_B"] = t.to_dict(orient="records")
    r["mae_por_grupo"] = {int(k): float(v) for k, v in a.groupby(t.grupo).mean().items()}
    r["mae_ingenuo_por_grupo"] = {int(k): float(v) for k, v in ai.groupby(t.grupo).mean().items()}
    r["mae_por_periodo"] = a.groupby(t.PERIODO).mean().to_dict()
    r["mae_ingenuo_por_periodo"] = ai.groupby(t.PERIODO).mean().to_dict()
    r["mape_mejor"] = float((a / t.y).mean())
    r["mape_ingenuo"] = float((ai / t.y).mean())
    r["regional_B"] = {
        p: {
            "real": float(np.average(q["y"], weights=q["n_act"])),
            "pred": float(np.average(q["pred"], weights=q["n_act"])),
            "ingenuo": float(np.average(q["lag1"], weights=q["n_act"])),
        }
        for p, q in t.groupby("PERIODO")
    }


def ejecutar(panel: pd.DataFrame) -> dict[str, Any]:
    """Ejecuta todas las corridas y devuelve los resultados serializables."""
    r: dict[str, Any] = {}
    pn, km = preparar(panel)
    _no_supervisado(pn, km, r)

    d = car.supervisado(pn)
    r["n_supervisado"] = len(d)
    r["periodos_supervisado"] = [d["PERIODO"].min(), d["PERIODO"].max()]
    r["y_describe"] = {k: float(v) for k, v in d["y"].describe().items()}
    r["dl_describe"] = {k: float(v) for k, v in d["dl"].describe().items()}
    r["correlaciones"] = {
        f: {
            c: {
                "pearson": float(d[c].corr(d[car.objetivo(f)])),
                "spearman": float(d[c].corr(d[car.objetivo(f)], method="spearman")),
            }
            for c in car.numericas(f) + ["mes"]
        }
        for f in (1, 2)
    }
    r["corr_entradas"] = {
        "lag1_lag12": float(d.lag1.corr(d.lag12)),
        "lag1_lag2": float(d.lag1.corr(d.lag2)),
        "dl_prev_dl_seas": float(d.dl_prev.corr(d.dl_seas)),
    }

    esq = esquemas(d)
    (tr_a, te_a), (tr_b, te_b) = esq["A"], esq["B"]
    r["n_A"] = {"train": len(tr_a), "test": len(te_a)}
    r["n_B"] = {
        "train": len(tr_b),
        "test": len(te_b),
        "periodos_train": [tr_b.PERIODO.min(), tr_b.PERIODO.max()],
        "periodos_test": [te_b.PERIODO.min(), te_b.PERIODO.max()],
    }
    res = {}
    for f in (1, 2):
        for s, (tr, te) in esq.items():
            logger.info("Corrida %s, esquema %s", f, s)
            res[(f, s)] = modelos.evaluar(f, tr, te)
            r[f"c{f}{s}"] = {**modelos.ingenuos(te), **modelos.tabla(res[(f, s)])}

    _modelo_adoptado(res[(2, "B")], te_b, r)
    r["panel_series"] = pn[
        [
            "UBIGEO",
            "DIST_N",
            "PROV_N",
            "PERIODO",
            "y",
            "mediana",
            "n_sum",
            "n_act",
            "p_cero",
            "dias",
        ]
        + ["grupo"]
    ].to_dict(orient="records")
    return r
