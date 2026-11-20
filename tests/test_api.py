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
    assert j["modelo_adoptado"] == "Random Forest"
    assert j["mejora_frente_ingenuo"] == pytest.approx(0.256, abs=1e-3)


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


def test_modelos_y_regional(client):
    m = client.get("/api/modelos").json()
    assert len(m["corridas"]) == 4 and m["adoptado"] == "Random Forest"
    c2b = next(c for c in m["corridas"] if c["corrida"] == 2 and c["esquema"] == "B")
    rf = next(x for x in c2b["prueba_metricas"] if x["modelo"] == "Random Forest")
    assert rf["MAE"] == pytest.approx(3.19, abs=5e-3)
    assert "curvas" not in c2b["gradiente"]
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
