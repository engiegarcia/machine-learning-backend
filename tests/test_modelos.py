"""Rezagos, gradiente descendiente, agrupamiento, métricas y pronóstico."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml import caracteristicas as car
from ml import modelos, pronostico
from ml.agrupamiento import dbscan_atipicos, kmeans_distritos, rodilla
from ml.gradiente import GDRegressor, ecuacion_normal


@pytest.fixture()
def pn(panel_sintetico):
    p = car.ordenar(panel_sintetico)
    km = kmeans_distritos(p, ks=range(2, 4))
    return car.agregar_rezagos(p.merge(km.grupo, left_on="UBIGEO", right_index=True))


def test_rezagos_y_objetivo_de_la_corrida_2(pn):
    d = pn[pn["UBIGEO"] == "200101"].reset_index(drop=True)
    assert d.loc[13, "lag1"] == d.loc[12, "y"]
    assert d.loc[13, "lag12"] == d.loc[1, "y"]
    assert d.loc[13, "dl"] == pytest.approx(np.log(d.loc[13, "y"] / d.loc[12, "y"]))
    assert d.loc[13, "dl_seas"] == pytest.approx(np.log(d.loc[1, "y"] / d.loc[0, "y"]))
    assert d.loc[13, "dias_ratio"] == pytest.approx(d.loc[13, "dias"] / d.loc[12, "dias"])
    assert pd.isna(d.loc[0, "lag1"])
    sup = car.supervisado(pn)
    assert len(sup) == 4 * (26 - 13)


def test_a_kwh():
    filas = pd.DataFrame({"lag1": [100.0, 50.0]})
    assert car.a_kwh(1, np.array([1.0, 2.0]), filas).tolist() == [1.0, 2.0]
    np.testing.assert_allclose(car.a_kwh(2, np.log([1.1, 0.9]), filas), [110.0, 45.0])


def test_gradiente_converge_a_la_ecuacion_normal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 3))
    y = 2.0 + x @ np.array([1.5, -0.7, 0.3]) + rng.normal(0, 0.1, 300)
    theta = ecuacion_normal(x, y)
    gd = GDRegressor(lr=0.1, epochs=2000).fit(x, y)
    np.testing.assert_allclose(gd.theta, theta, atol=1e-6)
    assert gd.costo[0] > gd.costo[-1] and not gd.diverge
    np.testing.assert_allclose(gd.predict(x[:2]), np.c_[np.ones(2), x[:2]] @ theta, atol=1e-6)


def test_gradiente_diverge_con_tasa_alta():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 2)) * 10
    gd = GDRegressor(lr=1.0, epochs=500).fit(x, x[:, 0])
    assert gd.diverge and len(gd.costo) < 500


def test_rodilla():
    curva = np.array([0, 0.1, 0.2, 0.3, 0.4, 2.0, 5.0, 9.0])
    assert rodilla(curva) == pytest.approx(0.4)


def test_kmeans_separa_niveles(panel_sintetico):
    km = kmeans_distritos(car.ordenar(panel_sintetico), ks=range(2, 4))
    assert km.k == 2
    assert km.grupo["200201"] == km.grupo["200202"] == 1  # menor consumo → grupo 1
    assert km.grupo["200101"] == km.grupo["200104"] == 2


def test_dbscan_marca_un_mes_anomalo(pn):
    pn = pn.copy()
    i = pn.index[(pn["UBIGEO"] == "200201") & (pn["PERIODO"] == "202506")][0]
    pn.loc[i, "y"] *= 3
    pn = car.agregar_rezagos(pn)
    db = dbscan_atipicos(pn)
    assert db.min_samples == 6
    assert ("200201", "202506") in set(zip(db.ruido["UBIGEO"], db.ruido["PERIODO"], strict=True))


def test_metricas_e_ingenuos():
    m = modelos.metricas(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert m["MAE"] == pytest.approx(2 / 3) and m["RMSE"] == pytest.approx(np.sqrt(4 / 3))
    te = pd.DataFrame(
        {"y": [10.0, 20.0], "lag1": [10.0, 18.0], "lag12": [9.0, 19.0], "lag13": [9.0, 19.0]}
    )
    ing = modelos.ingenuos(te)
    assert ing["Ingenuo (mes anterior)"]["test"]["MAE"] == pytest.approx(1.0)
    assert ing["Ingenuo estacional ajustado"]["test"]["MAE"] == pytest.approx(1.0)


def test_pronostico_del_mes_siguiente(pn):
    sup = car.supervisado(pn)
    modelo = pronostico.entrenar(sup, {"n_estimators": 10, "min_samples_leaf": 3}, n_jobs=1)
    x = pronostico.filas_siguiente(pn, {"200101": 31.0})
    ult = pn[pn["UBIGEO"] == "200101"].iloc[-1]
    assert x.loc["200101", "periodo"] == "202603" and x.loc["200101", "mes"] == 3
    assert x.loc["200101", "dias_ratio"] == pytest.approx(31.0 / ult["dias"])
    assert x.loc["200104", "dias_ratio"] == pytest.approx(1.0)
    res = {p.ubigeo: p for p in pronostico.pronosticar(modelo, x)}
    assert set(res) == {"200101", "200104", "200201", "200202"}
    p = res["200101"]
    assert p.consumo == pytest.approx(p.ultimo_consumo * (1 + p.variacion))
    assert 0.5 < p.consumo / p.ultimo_consumo < 1.5


@pytest.mark.parametrize(
    ("p", "s"), [("202610", "202611"), ("202612", "202701"), ("202501", "202502")]
)
def test_siguiente_periodo(p, s):
    assert pronostico.siguiente_periodo(p) == s
