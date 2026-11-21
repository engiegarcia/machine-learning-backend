"""Los artefactos del repositorio contienen las cifras publicadas en el informe."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ml import config


@pytest.fixture(scope="module")
def r():
    return json.loads(config.RESULTADOS.read_text(encoding="utf8"))


@pytest.fixture(scope="module")
def prep():
    return json.loads(config.RESUMEN_PREPROCESAMIENTO.read_text(encoding="utf8"))


def test_panel(prep):
    panel = pd.read_parquet(config.PANEL)
    assert panel.shape == (2652, 16)
    assert panel["UBIGEO"].nunique() == 78 and panel["PERIODO"].nunique() == 34
    assert prep["filas_totales"] == 19_713_482
    assert prep["residenciales_bt5b"] == 19_495_985
    assert prep["activos_limpios"] == 17_766_272
    assert (prep["atipicos_sup"], prep["atipicos_inf"]) == (628, 579)


def test_no_supervisado(r):
    assert r["kmeans_k"] == 3
    assert r["kmeans_silueta"]["3"] == pytest.approx(0.587, abs=5e-4)
    assert [g["n_distritos"] for g in r["kmeans_grupos"]] == [19, 41, 18]
    assert r["dbscan"]["ruido"] == 30 and r["dbscan"]["min_samples"] == 6
    assert r["dbscan"]["eps"] == pytest.approx(0.837, abs=5e-4)
    assert r["kmeans_pca"]["varianza"][0] == pytest.approx(0.994, abs=5e-4)
    assert len(r["kmeans_pca"]["puntos"]) == 78
    pt = r["dbscan_puntos"]
    assert len(pt["ubigeo"]) == len(pt["periodo"]) == len(pt["lab"]) == r["dbscan"]["n"]
    ruido = {
        (u, p)
        for u, p, lab in zip(pt["ubigeo"], pt["periodo"], pt["lab"], strict=True)
        if lab == -1
    }
    assert ruido == {(a["ubigeo"], a["periodo"]) for a in r["dbscan_atipicos"]}


def test_corridas(r):
    b1, b2 = r["c1B"], r["c2B"]
    assert b1["Ingenuo (mes anterior)"]["test"]["MAE"] == pytest.approx(4.29, abs=5e-3)
    assert min(b1[m]["test"]["MAE"] for m in ("Ridge", "Random Forest", "Gradient Boosting")) > 4.29
    assert r["mejor"] == "Random Forest"
    assert b2["Random Forest"]["test"]["MAE"] == pytest.approx(3.19, abs=5e-3)
    assert b2["Random Forest"]["params"] == {
        "max_depth": None,
        "min_samples_leaf": 3,
        "n_estimators": 500,
    }
    assert r["c1A"]["Gradient Boosting"]["test"]["MAE"] == pytest.approx(2.72, abs=5e-3)
