"""Endpoints de la API sobre el panel y los resultados del repositorio."""

from __future__ import annotations

import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_resumen(client):
    j = client.get("/api/resumen").json()
    assert j["distritos"] == 78 and len(j["periodos"]) == 34
    assert j["fuente"]["licencia"].endswith("(ODbL)")
    assert j["modelo_adoptado"] == "SVR"
    assert j["mejora_frente_ingenuo"] == pytest.approx(0.265, abs=1e-3)


def test_distritos(client):
    j = client.get("/api/distritos").json()
    assert len(j) == 78
    piura = next(d for d in j if d["ubigeo"] == "200101")
    assert piura["distrito"] == "PIURA" and piura["grupo"] == 3
    assert piura["ultimo_periodo"] == "202610"


def test_detalle_de_distrito(client):
    j = client.get("/api/distritos/200101").json()
    assert len(j["serie"]) == 34 and len(j["prueba_2026"]) == 10
    assert j["pronostico"]["periodo"] == "202611"


def test_distrito_desconocido_y_ubigeo_invalido(client):
    r = client.get("/api/distritos/999999")
    assert r.status_code == 404 and r.json()["error_code"] == "DISTRITO_NO_ENCONTRADO"
    assert client.get("/api/distritos/12ab").status_code == 422


def test_grupos_y_atipicos(client):
    g = client.get("/api/grupos").json()
    assert g["k"] == 3 and [x["n_distritos"] for x in g["grupos"]] == [19, 41, 18]
    a = client.get("/api/atipicos").json()
    assert len(a["atipicos"]) == 30
    sapillica = a["atipicos"][0]
    assert sapillica["distrito"] == "SAPILLICA" and sapillica["periodo"] == "202502"


def test_datos_de_los_graficos_de_dispersion(client):
    g = client.get("/api/grupos").json()
    j = g["jerarquico"]
    assert len(j["dendrograma"]["hojas"]) == len(j["dendrograma"]["nombres"]) == 78
    assert len(j["dendrograma"]["icoord"]) == 77
    pca = g["pca"]
    assert len(pca["puntos"]) == 78 and [c["grupo"] for c in pca["centroides"]] == [1, 2, 3]
    xs = [c["x"] for c in pca["centroides"]]
    assert xs == sorted(xs)  # la componente 1 ordena los grupos por consumo
    a = client.get("/api/atipicos").json()
    assert len(a["kdist"]) == len(a["puntos"]) == a["evaluados"]
    assert a["kdist"] == sorted(a["kdist"])
    assert sum(p["ruido"] for p in a["puntos"]) == 30
    m = client.get("/api/modelos").json()["real_vs_pronostico"]
    assert len(m) == 780 and {p["periodo"][:4] for p in m} == {"2026"}


def test_modelos_y_regional(client):
    m = client.get("/api/modelos").json()
    assert len(m["corridas"]) == 4 and m["adoptado"] == "SVR"
    c2b = next(c for c in m["corridas"] if c["corrida"] == 2 and c["esquema"] == "B")
    assert len(c2b["prueba_metricas"]) == 14 and len(c2b["validacion_cruzada"]) == 10
    svr = next(x for x in c2b["prueba_metricas"] if x["modelo"] == "SVR")
    assert svr["MAE"] == pytest.approx(3.155, abs=5e-4)
    comp = {c["modelo"]: c for c in m["comparacion_adoptado"]}
    assert comp["Ingenuo (mes anterior)"]["p_menor"] == 1.0
    assert len(c2b["gradiente"]["curvas"]["0.1"]) == 300


def test_predicciones_de_todos_los_modelos(client):
    p = client.get("/api/predicciones").json()
    assert p["modelos"][0] == "SVR" and len(p["modelos"]) == 11
    assert len(p["filas"]) == 780
    assert all(len(v) == 780 for v in p["predicciones"].values())
    real = [f["real"] for f in p["filas"]]
    mae = sum(abs(a - b) for a, b in zip(p["predicciones"]["SVR"], real, strict=True)) / 780
    assert mae == pytest.approx(3.155, abs=5e-4)
    reg = client.get("/api/regional").json()["serie"]
    assert len(reg) == 34 and reg[-1]["pronostico"] is not None and reg[0]["pronostico"] is None


def test_pronostico(client):
    todos = client.get("/api/pronostico").json()
    assert len(todos) == 78
    base = client.get("/api/pronostico/200101").json()
    mas_dias = client.get("/api/pronostico/200101", params={"dias": 33}).json()
    assert base["dias_supuestos"] == pytest.approx(base["dias_ultimo"])
    assert mas_dias["dias_supuestos"] == 33 and len(base["supuestos"]) == 2
    assert client.get("/api/pronostico/200101", params={"dias": 5}).status_code == 422
