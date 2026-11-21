"""Servicio de la API: carga el panel y los resultados y entrena el modelo de pronóstico.

Se crea una sola vez al iniciar la aplicación. Los resultados de las corridas vienen de
``python cli.py corridas`` (``resultados/resultados.json``); el modelo adoptado de la corrida 2
(el de menor MAE en la prueba temporal) se reentrena con todos los distrito-meses para
pronosticar el mes siguiente.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import Settings
from core.exceptions import ArtefactoFaltanteError, DistritoNoEncontradoError
from ml import caracteristicas as car
from ml import pronostico
from schemas import ml as s

logger = logging.getLogger(__name__)

FUENTE = s.Fuente(
    conjunto="Consumo energético de clientes Enosa [Distriluz - DLZ]",
    publicador="Empresa Regional de Servicio Público de Electricidad Electronoroeste S.A. - ENOSA",
    licencia="Open Data Commons Open Database License (ODbL)",
    url="https://www.datosabiertos.gob.pe/dataset/consumo-energ%C3%A9tico-de-clientes-enosa-distriluz-dlz",
)
SUPUESTOS = [
    "Los días facturados del mes pronosticado son los indicados o, si no se indican, "
    "los del último mes.",
    "El número de suministros del mes pronosticado es el del último mes.",
]
DESCRIPCION = {
    ("1", "A"): "Nivel del consumo · división aleatoria 80/20",
    ("1", "B"): "Nivel del consumo · prueba temporal 2026",
    ("2", "A"): "Variación logarítmica mensual · división aleatoria 80/20",
    ("2", "B"): "Variación logarítmica mensual · prueba temporal 2026",
}


def _leer_json(ruta: Path) -> Any:
    if not ruta.exists():
        raise ArtefactoFaltanteError(f"no existe {ruta}")
    return json.loads(ruta.read_text(encoding="utf8"))


class MLService:
    """Consultas sobre el panel, los resultados y el pronóstico del mes siguiente."""

    def __init__(self, settings: Settings) -> None:
        if not settings.panel.exists():
            raise ArtefactoFaltanteError(f"no existe {settings.panel}")
        self.r: dict[str, Any] = _leer_json(settings.resultados)
        self.prep: dict[str, Any] = _leer_json(settings.preprocesamiento)
        grupo = pd.Series(self.r["kmeans_asignacion"], name="grupo")
        pn = car.ordenar(pd.read_parquet(settings.panel))
        self.pn = car.agregar_rezagos(pn.merge(grupo, left_on="UBIGEO", right_index=True))
        self.nombres = self.pn.drop_duplicates("UBIGEO").set_index("UBIGEO")
        params = dict(self.r["c2B"][self.r["mejor"]]["params"])
        if settings.arboles_pronostico and "n_estimators" in params:
            params["n_estimators"] = settings.arboles_pronostico
        self.modelo = pronostico.entrenar(
            car.supervisado(self.pn), params, n_jobs=settings.n_jobs, nombre=self.r["mejor"]
        )
        self._siguiente = pronostico.filas_siguiente(self.pn)
        self._atipicos = {(a["ubigeo"], a["periodo"]) for a in self.r["dbscan_atipicos"]}
        logger.info("Modelo de pronóstico entrenado", extra={"params": params})

    # ------------------------------------------------------------------ consultas
    def health_check(self) -> bool:
        return len(self._siguiente) == len(self.nombres) > 0

    def resumen(self) -> s.ResumenResponse:
        p, b = self.prep, self.r["c2B"]
        mae, mae_ing = b[self.r["mejor"]]["test"]["MAE"], b["Ingenuo (mes anterior)"]["test"]["MAE"]
        pasos = [
            ("Registros descargados", p["filas_totales"]),
            ("Fuera de formato o de la concesión", p["fuera_de_formato_o_concesion"]),
            ("No residenciales", p["no_residenciales"]),
            ("Residenciales con otra tarifa", p["residenciales_otra_tarifa"]),
            ("Residenciales BT5B", p["residenciales_bt5b"]),
            ("Atípicos por encima (Tukey)", p["atipicos_sup"]),
            ("Atípicos por debajo (Tukey)", p["atipicos_inf"]),
            ("Recibos activos limpios", p["activos_limpios"]),
        ]
        return s.ResumenResponse(
            fuente=FUENTE,
            periodos=p["meses"],
            registros_descargados=p["filas_totales"],
            filtrado=[s.PasoFiltrado(paso=a, registros=n) for a, n in pasos],
            recibos_activos_limpios=p["activos_limpios"],
            proporcion_recibos_cero=p["ceros_proporcion"],
            limites_tukey_kwh={
                "inferior": p["tukey_log"]["limite_inf_kwh"],
                "superior": p["tukey_log"]["limite_sup_kwh"],
            },
            distritos=p["distritos_total"],
            celdas=p["celdas_total"],
            modelo_adoptado=self.r["mejor"],
            mae_modelo=mae,
            mae_ingenuo=mae_ing,
            mejora_frente_ingenuo=1 - mae / mae_ing,
        )

    def _distrito(self, ubigeo: str) -> s.Distrito:
        n = self.nombres.loc[ubigeo]
        ult = self._siguiente.loc[ubigeo]
        serie = self.pn[self.pn["UBIGEO"] == ubigeo]
        return s.Distrito(
            ubigeo=ubigeo,
            distrito=n["DIST_N"],
            provincia=n["PROV_N"],
            departamento=n["DEP_N"],
            grupo=int(n["grupo"]),
            ultimo_periodo=str(ult["ultimo_periodo"]),
            ultimo_consumo=float(ult["lag1"]),
            suministros=int(serie["n_sum"].iloc[-1]),
        )

    def distritos(self) -> list[s.Distrito]:
        return [self._distrito(u) for u in sorted(self.nombres.index)]

    def _validar(self, ubigeo: str) -> None:
        if ubigeo not in self.nombres.index:
            raise DistritoNoEncontradoError(f"ubigeo desconocido: {ubigeo}")

    def distrito(self, ubigeo: str) -> s.DistritoDetalle:
        self._validar(ubigeo)
        serie = self.pn[self.pn["UBIGEO"] == ubigeo]
        puntos = [
            s.PuntoSerie(
                periodo=f.PERIODO,
                consumo=None if pd.isna(f.y) else float(f.y),
                mediana=None if pd.isna(f.mediana) else float(f.mediana),
                suministros=int(f.n_sum),
                activos=None if pd.isna(f.n_act) else int(f.n_act),
                proporcion_cero=float(f.p_cero),
                dias=float(f.dias),
                atipico=(ubigeo, f.PERIODO) in self._atipicos,
            )
            for f in serie.itertuples()
        ]
        prueba = [
            s.PuntoPrueba(
                periodo=q["PERIODO"], real=q["y"], pronostico=q["pred"], ingenuo=q["pred_ingenuo"]
            )
            for q in self.r["pred_B"]
            if q["UBIGEO"] == ubigeo
        ]
        return s.DistritoDetalle(
            **self._distrito(ubigeo).model_dump(),
            serie=puntos,
            prueba_2026=prueba,
            pronostico=self.pronostico(ubigeo),
        )

    def grupos(self) -> s.GruposResponse:
        r, pc = self.r, self.r["kmeans_pca"]
        n = self.nombres
        return s.GruposResponse(
            k=r["kmeans_k"],
            silueta={int(k): v for k, v in r["kmeans_silueta"].items()},
            codo={int(k): v for k, v in r["kmeans_codo"].items()},
            grupos=[s.Grupo(**g) for g in r["kmeans_grupos"]],
            pca=s.ProyeccionPCA(
                varianza=pc["varianza"],
                puntos=[
                    s.PuntoPCA(
                        ubigeo=u,
                        distrito=n.at[u, "DIST_N"],
                        provincia=n.at[u, "PROV_N"],
                        grupo=int(r["kmeans_asignacion"][u]),
                        x=x,
                        y=y,
                    )
                    for u, (x, y) in sorted(pc["puntos"].items())
                ],
                centroides=[
                    s.CentroidePCA(grupo=int(g), x=x, y=y)
                    for g, (x, y) in sorted(pc["centroides"].items())
                ],
            ),
            jerarquico=s.Jerarquico(
                **{k: v for k, v in r["jerarquico"].items() if k != "dendrograma"},
                dendrograma=s.Dendrograma(
                    **r["jerarquico"]["dendrograma"],
                    nombres=[n.at[u, "DIST_N"] for u in r["jerarquico"]["dendrograma"]["hojas"]],
                ),
            ),
        )

    def atipicos(self) -> s.AtipicosResponse:
        db = self.r["dbscan"]
        return s.AtipicosResponse(
            eps=db["eps"],
            min_samples=db["min_samples"],
            evaluados=db["n"],
            clusters=db["clusters"],
            atipicos=[s.Atipico(**a) for a in self.r["dbscan_atipicos"]],
            por_periodo=self.r["dbscan_por_periodo"],
            kdist=[round(v, 4) for v in self.r["dbscan_kdist"]],
            puntos=self._puntos_dbscan(),
        )

    def _puntos_dbscan(self) -> list[s.PuntoDBSCAN]:
        pt = self.r["dbscan_puntos"]
        return [
            s.PuntoDBSCAN(
                ubigeo=u,
                distrito=self.nombres.at[u, "DIST_N"],
                periodo=p,
                log_mes=round(x, 5),
                log_anio=round(y, 5),
                ruido=lab == -1,
            )
            for u, p, x, y, lab in zip(
                pt["ubigeo"], pt["periodo"], pt["x"], pt["y"], pt["lab"], strict=True
            )
        ]

    def modelos(self) -> s.ModelosResponse:
        r = self.r
        corridas = []
        for f in ("1", "2"):
            for e in ("A", "B"):
                c = r[f"c{f}{e}"]
                n = r[f"n_{e}"]
                corridas.append(
                    s.Corrida(
                        corrida=int(f),
                        esquema=e,
                        descripcion=DESCRIPCION[(f, e)],
                        entrenamiento=n["train"],
                        prueba=n["test"],
                        prueba_metricas=[
                            s.Metrica(modelo=k, **v["test"], params=v.get("params"))
                            for k, v in c.items()
                            if not k.startswith("_")
                        ],
                        validacion_cruzada=[
                            s.Metrica(modelo=k, MAE=v["MAE"], RMSE=v["RMSE"], R2=v["R2"])
                            for k, v in c["_cv"].items()
                        ],
                        gradiente={k: v for k, v in c["_gd"].items() if k != "curvas"},
                    )
                )
        return s.ModelosResponse(
            corridas=corridas,
            adoptado=r["mejor"],
            importancia=r["importancia"],
            mae_por_grupo={
                int(g): {"modelo": v, "ingenuo": r["mae_ingenuo_por_grupo"][g]}
                for g, v in r["mae_por_grupo"].items()
            },
            mae_por_periodo={
                p: {"modelo": v, "ingenuo": r["mae_ingenuo_por_periodo"][p]}
                for p, v in r["mae_por_periodo"].items()
            },
            mape={"modelo": r["mape_mejor"], "ingenuo": r["mape_ingenuo"]},
            real_vs_pronostico=[
                s.PuntoDispersion(
                    ubigeo=q["UBIGEO"],
                    distrito=q["DIST_N"],
                    periodo=q["PERIODO"],
                    grupo=int(q["grupo"]),
                    real=q["y"],
                    pronostico=q["pred"],
                )
                for q in r["pred_B"]
            ],
            comparacion_adoptado=[
                s.ComparacionAdoptado(modelo=k, **v)
                for k, v in sorted(
                    r["comparacion_adoptado"].items(), key=lambda kv: kv[1]["diferencia"]
                )
            ],
        )

    def regional(self) -> s.RegionalResponse:
        rb = self.r["regional_B"]
        return s.RegionalResponse(
            serie=[
                s.PuntoRegional(
                    periodo=p,
                    media=v["media"],
                    mediana=v["mediana"],
                    recibos_activos=v["n"],
                    gwh=v["gwh"],
                    pronostico=rb.get(p, {}).get("pred"),
                    ingenuo=rb.get(p, {}).get("ingenuo"),
                )
                for p, v in sorted(self.prep["serie_regional"].items())
            ]
        )

    def pronostico(self, ubigeo: str, dias: float | None = None) -> s.PronosticoResponse:
        self._validar(ubigeo)
        x = pronostico.filas_siguiente(self.pn, {ubigeo: dias}) if dias else self._siguiente
        p = pronostico.pronosticar(self.modelo, x.loc[[ubigeo]])[0]
        return s.PronosticoResponse(
            **p.__dict__, distrito=self.nombres.at[ubigeo, "DIST_N"], supuestos=SUPUESTOS
        )

    def pronosticos(self) -> list[s.PronosticoResponse]:
        return [
            s.PronosticoResponse(
                **p.__dict__, distrito=self.nombres.at[p.ubigeo, "DIST_N"], supuestos=SUPUESTOS
            )
            for p in pronostico.pronosticar(self.modelo, self._siguiente)
        ]
