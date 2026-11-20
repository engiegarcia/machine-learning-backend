"""Línea de comandos del proyecto de aprendizaje automático.

Flujo completo (los pasos 1 y 2 solo hacen falta para reconstruir el panel desde cero)::

    python cli.py descargar      # 1. 34 meses de ENOSA → datos/limpios (~3,7 GB de descarga)
    python cli.py preparar       # 2. panel distrito-mes → datos/procesados
    python cli.py corridas       # 3. k-means, DBSCAN y corridas 1 y 2 → resultados/resultados.json
    python cli.py figuras        # 4. figuras del informe → resultados/figuras
    python cli.py pronosticar    # pronóstico del mes siguiente por distrito
    python cli.py pronosticar --ubigeo 200101 --dias 31

El panel ya procesado viene en el repositorio (``datos/procesados/panel.parquet``), así que los
pasos 3 y 4 se pueden ejecutar sin descargar nada.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ml import config


def _escribir_json(ruta: Path, datos: Any) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=1, default=str), encoding="utf8")


def _leer_json(ruta: Path) -> Any:
    return json.loads(ruta.read_text(encoding="utf8"))


def descargar(args: argparse.Namespace) -> int:
    from urllib.parse import unquote

    from ml.fuente import descargar_mes, urls_mensuales

    registro = args.destino / "auditoria.json"
    auditoria = _leer_json(registro) if registro.exists() else {}
    hechos = {v.get("archivo") for v in auditoria.values()}
    errores = 0
    for url in urls_mensuales(args.urls):
        nombre = unquote(url.rsplit("/", 1)[1])
        if nombre in hechos and not args.forzar:
            continue
        try:
            a = descargar_mes(url, args.destino)
        except Exception as e:  # noqa: BLE001 - se informa y se sigue con el mes siguiente
            errores += 1
            print(f"ERROR {nombre}: {e}", file=sys.stderr)
            continue
        auditoria[a["periodo"]] = a
        _escribir_json(registro, dict(sorted(auditoria.items())))
        print(
            f"{a['periodo']}: {a['filas']:,} filas, "
            f"{a['lineas_descartadas']} descartadas, {a['mb']} MB"
        )
    print(f"{len(auditoria)} meses en {args.destino}")
    return 1 if errores else 0


def preparar(args: argparse.Namespace) -> int:
    from ml.preprocesamiento import cargar_limpios, construir_panel

    panel, resumen = construir_panel(cargar_limpios(args.origen))
    config.DIR_PROCESADOS.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(config.PANEL, index=False)
    _escribir_json(config.RESUMEN_PREPROCESAMIENTO, resumen)
    print(
        f"{resumen['filas_totales']:,} registros → {resumen['activos_limpios']:,} recibos activos "
        f"limpios → {resumen['distritos_total']} distritos × {len(resumen['meses'])} meses"
    )
    return 0


def corridas(args: argparse.Namespace) -> int:
    from ml.corridas import ejecutar

    r = ejecutar(pd.read_parquet(config.PANEL))
    _escribir_json(config.RESULTADOS, r)
    print(f"k-means: k = {r['kmeans_k']}, silueta {r['kmeans_silueta'][r['kmeans_k']]:.3f}")
    print(f"DBSCAN: ε = {r['dbscan']['eps']:.3f}, {r['dbscan']['ruido']} distrito-meses atípicos")
    for clave in ("c1A", "c1B", "c2A", "c2B"):
        print(f"\nCorrida {clave[1]}, esquema {clave[2]}")
        for nombre, v in r[clave].items():
            if not nombre.startswith("_"):
                m = v["test"]
                print(
                    f"  {nombre:48s} MAE {m['MAE']:6.2f}  RMSE {m['RMSE']:6.2f}  R² {m['R2']:.4f}"
                )
    print(f"\nModelo adoptado: {r['mejor']} (corrida 2) → {config.RESULTADOS}")
    return 0


def figuras(args: argparse.Namespace) -> int:
    from ml.figuras import generar

    nombres = generar(
        _leer_json(config.RESULTADOS), _leer_json(config.RESUMEN_PREPROCESAMIENTO), config.FIGURAS
    )
    print(f"{len(nombres)} figuras en {config.FIGURAS}")
    return 0


def pronosticar(args: argparse.Namespace) -> int:
    from ml import caracteristicas as car
    from ml import pronostico

    r = _leer_json(config.RESULTADOS)
    pn = car.ordenar(pd.read_parquet(config.PANEL))
    grupo = pd.Series(r["kmeans_asignacion"], name="grupo")
    pn = car.agregar_rezagos(pn.merge(grupo, left_on="UBIGEO", right_index=True))
    modelo = pronostico.entrenar(car.supervisado(pn), r["c2B"]["Random Forest"]["params"])
    dias = {args.ubigeo: args.dias} if args.ubigeo and args.dias else None
    x = pronostico.filas_siguiente(pn, dias)
    if args.ubigeo:
        if args.ubigeo not in x.index:
            print(f"ubigeo desconocido: {args.ubigeo}", file=sys.stderr)
            return 2
        x = x.loc[[args.ubigeo]]
    res = pronostico.pronosticar(modelo, x)
    if args.json:
        print(json.dumps([p.__dict__ for p in res], ensure_ascii=False, indent=1))
        return 0
    nombres = pn.drop_duplicates("UBIGEO").set_index("UBIGEO")["DIST_N"]
    for p in res:
        print(
            f"{p.ubigeo} {nombres[p.ubigeo]:28s} {p.periodo}: {p.consumo:7.1f} kWh "
            f"({p.variacion:+.1%} frente a {p.ultimo_periodo}, {p.dias_supuestos:.1f} días)"
        )
    return 0


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="comando", required=True)

    d = sub.add_parser("descargar", help="descarga los archivos mensuales de ENOSA")
    d.add_argument("--urls", type=Path, default=config.URLS)
    d.add_argument("--destino", type=Path, default=config.DIR_LIMPIOS)
    d.add_argument("--forzar", action="store_true", help="vuelve a descargar los meses ya leídos")
    d.set_defaults(func=descargar)

    pr = sub.add_parser("preparar", help="construye el panel distrito-mes")
    pr.add_argument("--origen", type=Path, default=config.DIR_LIMPIOS)
    pr.set_defaults(func=preparar)

    sub.add_parser("corridas", help="ejecuta las corridas del informe").set_defaults(func=corridas)
    sub.add_parser("figuras", help="dibuja las figuras del informe").set_defaults(func=figuras)

    pn = sub.add_parser("pronosticar", help="pronostica el mes siguiente por distrito")
    pn.add_argument("--ubigeo", help="código de 6 dígitos del distrito")
    pn.add_argument("--dias", type=float, help="días facturados supuestos del mes pronosticado")
    pn.add_argument("--json", action="store_true")
    pn.set_defaults(func=pronosticar)
    return p


def main(argv: list[str] | None = None) -> int:
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = construir_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
