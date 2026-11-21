"""Figuras del informe (fuente serif y coma decimal, como los informes anteriores del grupo)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from ml.config import MESES  # noqa: E402

AZUL, RELLENO, GRIS, GRIS_CLARO = "#2f6db5", "#dce9f6", "#6b7785", "#eef1f5"
ROJO, NARANJA, VERDE = "#c8453b", "#e8862a", "#3c8d5a"
COLOR_GRUPO = {1: VERDE, 2: AZUL, 3: NARANJA}
NOMBRE_VARIABLE = {
    "dias_ratio": "Razón de días facturados",
    "dl_seas": "Variación del mismo mes, año anterior",
    "dl_prev": "Variación del mes anterior",
    "mes": "Mes del año",
    "p_cero_lag1": "Proporción de recibos en cero (t − 1)",
    "grupo": "Grupo k-means",
    "log_n": "Número de suministros (log)",
    "PROV_N": "Provincia",
}
ESTILO = {
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.axisbelow": True,
    "savefig.dpi": 220,
    "axes.unicode_minus": False,
}


def coma(v: float, d: int = 0) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def _etiqueta(p: str) -> str:
    return f"{MESES[int(p[4:]) - 1]}\n{p[:4]}"


class _Figuras:
    def __init__(self, r: dict[str, Any], prep: dict[str, Any], destino: Path) -> None:
        self.r, self.p, self.destino = r, prep, destino
        destino.mkdir(parents=True, exist_ok=True)

    def guardar(self, fig: plt.Figure, nombre: str) -> None:
        fig.tight_layout()
        fig.savefig(self.destino / nombre)
        plt.close(fig)

    def flujo(self) -> None:
        fig, ax = plt.subplots(figsize=(11, 3.3))
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 3.3)
        ax.axis("off")
        meses, distritos = len(self.p["meses"]), self.p["distritos_total"]
        pasos = [
            (f"Datos abiertos\nENOSA\n{meses} meses", GRIS_CLARO, GRIS),
            ("Filtrado y\nlimpieza\n(BT5B residencial)", RELLENO, AZUL),
            (f"Panel\ndistrito-mes\n{distritos} × {meses}", RELLENO, AZUL),
            ("Rezagos y\nvariables", RELLENO, AZUL),
            ("Modelos\nsupervisados y\nno supervisados", RELLENO, AZUL),
            ("Validación\nMAE, RMSE, R²\nsilueta", GRIS_CLARO, GRIS),
        ]
        for k, (t, fc, ec) in enumerate(pasos):
            x = 0.2 + k * 1.8
            ax.add_patch(
                FancyBboxPatch(
                    (x, 1.0),
                    1.5,
                    1.4,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    fc=fc,
                    ec=ec,
                    lw=1.3,
                )
            )
            ax.text(x + 0.75, 1.7, t, ha="center", va="center", fontsize=9.5)
            if k < len(pasos) - 1:
                ax.annotate(
                    "",
                    xy=(x + 1.78, 1.7),
                    xytext=(x + 1.52, 1.7),
                    arrowprops={"arrowstyle": "-|>", "color": "#444", "lw": 1.5},
                )
        ax.text(
            5.5,
            0.45,
            "Esquema A: división aleatoria 80/20 con validación cruzada k = 5     ·     "
            "Esquema B: prueba temporal (2026)",
            ha="center",
            fontsize=9.5,
            color=GRIS,
        )
        self.guardar(fig, "flujo.png")

    def histograma(self) -> None:
        h, bordes = np.array(self.p["hist_consumo"]), np.linspace(0, 600, 61)
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.bar(bordes[:-1], h / 1e6, width=10, align="edge", color=RELLENO, edgecolor=AZUL, lw=0.6)
        med, media = self.p["consumo_describe"]["50%"], self.p["consumo_describe"]["mean"]
        ax.axvline(med, color=AZUL, lw=1.6, ls="--")
        ax.axvline(media, color=ROJO, lw=1.6, ls=":")
        tope = ax.get_ylim()[1]
        ax.text(med + 6, tope * 0.9, f"mediana {coma(med)} kWh", color=AZUL, fontsize=9.5)
        ax.text(media + 6, tope * 0.75, f"media {coma(media, 1)} kWh", color=ROJO, fontsize=9.5)
        ax.set_xlabel("Consumo del recibo (kWh/mes); la última barra agrupa 590 kWh o más")
        ax.set_ylabel("Recibos (millones)")
        self.guardar(fig, "hist.png")

    def serie(self) -> None:
        sr = self.p["serie_regional"]
        per = sorted(sr)
        x = range(len(per))
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.plot(
            x, [sr[p]["media"] for p in per], color=AZUL, marker="o", ms=3.5, lw=1.8, label="Media"
        )
        ax.plot(
            x,
            [sr[p]["mediana"] for p in per],
            color=GRIS,
            marker="s",
            ms=3,
            lw=1.4,
            ls="--",
            label="Mediana",
        )
        ticks = range(0, len(per), 3)
        ax.set_xticks(ticks, [_etiqueta(per[i]) for i in ticks], fontsize=9)
        ax.set_ylabel("kWh por suministro activo")
        ax.legend(loc="upper right", fontsize=9.5)
        self.guardar(fig, "serie.png")

    def codo(self) -> None:
        r = self.r
        ks = sorted(int(k) for k in r["kmeans_codo"])
        codo = {int(k): v for k, v in r["kmeans_codo"].items()}
        sil = {int(k): v for k, v in r["kmeans_silueta"].items()}
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
        a1.plot(ks, [codo[k] for k in ks], color=AZUL, marker="o")
        a1.set_xlabel("k")
        a1.set_ylabel("SSE (inercia)")
        a1.set_title("a) Método del codo", loc="left", fontsize=11)
        a2.plot(ks, [sil[k] for k in ks], color=AZUL, marker="o")
        a2.plot([r["kmeans_k"]], [sil[int(r["kmeans_k"])]], "o", color=ROJO, ms=9)
        a2.set_xlabel("k")
        a2.set_ylabel("Coeficiente de silueta")
        a2.set_title("b) Silueta media", loc="left", fontsize=11)
        for a in (a1, a2):
            a.set_xticks(ks)
        self.guardar(fig, "codo.png")

    def perfiles(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 3.9))
        for g in self.r["kmeans_grupos"]:
            ax.plot(
                range(12),
                g["perfil"],
                color=COLOR_GRUPO.get(g["grupo"], GRIS),
                marker="o",
                lw=2,
                label=f"Grupo {g['grupo']} ({g['n_distritos']} distritos)",
            )
        ax.set_xticks(range(12), MESES)
        ax.set_ylabel("kWh por suministro activo")
        ax.set_yscale("log")
        ticks = [20, 30, 50, 80, 120, 150]
        ax.set_yticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
        self.guardar(fig, "perfiles.png")

    def pca(self) -> None:
        pc = self.r["kmeans_pca"]
        g = self.r["kmeans_asignacion"]
        v = pc["varianza"]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for gk in sorted({int(x) for x in g.values()}):
            pts = np.array([pc["puntos"][u] for u, k in g.items() if int(k) == gk])
            col = COLOR_GRUPO.get(gk, GRIS)
            ax.scatter(pts[:, 0], pts[:, 1], s=22, color=col, alpha=0.85, label=f"Grupo {gk}")
            cx, cy = pc["centroides"][str(gk)]
            ax.scatter([cx], [cy], marker="X", s=170, color=col, edgecolor="#222", lw=1)
        ax.set_xlabel(f"Componente principal 1 ({coma(v[0] * 100, 1)} % de la varianza)")
        ax.set_ylabel(f"Componente principal 2 ({coma(v[1] * 100, 1)} %)")
        ax.xaxis.set_major_formatter(lambda t, _: coma(t, 0).replace("-", "−"))
        ax.yaxis.set_major_formatter(lambda t, _: coma(t, 1).replace("-", "−"))
        ax.legend(fontsize=9.5)
        self.guardar(fig, "pca.png")

    def dbscan(self) -> None:
        r = self.r
        kd, eps = np.array(r["dbscan_kdist"]), r["dbscan"]["eps"]
        pt = r["dbscan_puntos"]
        lab, x, y = np.array(pt["lab"]), np.array(pt["x"]) * 100, np.array(pt["y"]) * 100
        fig, (a1, a2) = plt.subplots(
            1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1, 1.25]}
        )
        a1.plot(range(len(kd)), kd, color=AZUL)
        a1.axhline(eps, color=ROJO, ls="--", lw=1.2)
        a1.text(20, eps + 0.12, f"ε = {coma(eps, 2)}", color=ROJO, fontsize=9.5)
        a1.set_xlabel("Distrito-meses ordenados")
        a1.set_ylabel(f"Distancia al {r['dbscan']['min_samples']}.º vecino")
        a1.set_title("a) Curva k-distancia", loc="left", fontsize=11)
        a2.scatter(
            x[lab != -1],
            y[lab != -1],
            s=9,
            color=RELLENO,
            edgecolor=AZUL,
            lw=0.4,
            label="Núcleo y borde",
        )
        a2.scatter(x[lab == -1], y[lab == -1], s=26, color=ROJO, label="Ruido (atípicos)")
        a2.set_xlabel("Variación respecto del mes anterior (log × 100)")
        a2.set_ylabel("Variación interanual (log × 100)")
        a2.set_title("b) Distrito-meses según DBSCAN", loc="left", fontsize=11)
        a2.legend(fontsize=9)
        self.guardar(fig, "dbscan.png")

    def gradiente(self) -> None:
        fig, ejes = plt.subplots(1, 2, figsize=(11, 3.7))
        titulos = {"c1B": "a) Corrida 1 (nivel)", "c2B": "b) Corrida 2 (variación mensual)"}
        for ax, (clave, titulo) in zip(ejes, titulos.items(), strict=True):
            gd = self.r[clave]["_gd"]
            for lr, col in (("0.01", GRIS), ("0.1", AZUL), ("0.5", ROJO)):
                c = [v for v in gd["curvas"][lr] if v is not None]
                ax.plot(
                    np.arange(len(c)) * 10,
                    c,
                    color=col,
                    lw=1.8,
                    label=f"α = {lr.replace('.', ',')}",
                )
            ax.set_yscale("log")
            ax.set_xlabel("Iteración")
            ax.set_ylabel("Costo J(θ) = MSE")
            ax.axhline(gd["mse_ne"], color="#333", ls=":", lw=1)
            ax.legend(fontsize=9)
            ax.set_title(titulo, loc="left", fontsize=11)
        ejes[0].set_ylim(5, 1e6)
        self.guardar(fig, "gd.png")

    def comparacion(self) -> None:
        r = self.r
        nombres = [
            "Ingenuo (mes anterior)",
            "Ingenuo estacional ajustado",
            "Regresión lineal (gradiente descendiente)",
            "Ridge",
            "Random Forest",
            "Gradient Boosting",
        ]
        cortos = [
            "Ingenuo",
            "Ingenuo\nestac. ajust.",
            "Regresión\nlineal (GD)",
            "Ridge",
            "Random\nForest",
            "Gradient\nBoosting",
        ]
        fig, ax = plt.subplots(figsize=(10.5, 3.9))
        xx, w = np.arange(len(nombres)), 0.38
        v1 = [r["c1B"][n]["test"]["MAE"] for n in nombres]
        v2 = [r["c2B"][n]["test"]["MAE"] for n in nombres]
        b1 = ax.bar(xx - w / 2, v1, w, color=GRIS_CLARO, edgecolor=GRIS, label="Corrida 1: nivel")
        b2 = ax.bar(
            xx + w / 2, v2, w, color=RELLENO, edgecolor=AZUL, label="Corrida 2: variación mensual"
        )
        for b in list(b1) + list(b2):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.08,
                coma(b.get_height(), 2),
                ha="center",
                fontsize=8.5,
            )
        ax.axhline(r["c2B"]["Ingenuo (mes anterior)"]["test"]["MAE"], color=ROJO, ls="--", lw=1)
        ax.set_xticks(xx, cortos, fontsize=9.5)
        ax.set_ylabel("MAE (kWh)")
        ax.legend(fontsize=9.5, loc="upper left")
        self.guardar(fig, "comparacion.png")

    def regional(self) -> None:
        rg = self.r["regional_B"]
        pp = sorted(rg)
        x = range(len(pp))
        fig, ax = plt.subplots(figsize=(10.5, 3.7))
        ax.plot(x, [rg[p]["real"] for p in pp], color="#222", marker="o", lw=2, label="Real")
        ax.plot(
            x,
            [rg[p]["pred"] for p in pp],
            color=AZUL,
            marker="s",
            lw=1.8,
            label=f"{self.r['mejor']} (corrida 2)",
        )
        ax.plot(
            x,
            [rg[p]["ingenuo"] for p in pp],
            color=ROJO,
            marker="^",
            lw=1.4,
            ls="--",
            label="Ingenuo (mes anterior)",
        )
        ax.set_xticks(x, [_etiqueta(p) for p in pp], fontsize=9)
        ax.set_ylabel("kWh por suministro activo")
        ax.legend(fontsize=9.5)
        self.guardar(fig, "regional.png")

    def dispersion(self) -> None:
        te = pd.DataFrame(self.r["pred_B"])
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        for gk in sorted(te["grupo"].unique()):
            q = te[te.grupo == gk]
            ax.scatter(
                q["y"],
                q["pred"],
                s=12,
                color=COLOR_GRUPO.get(gk, GRIS),
                alpha=0.7,
                label=f"Grupo {gk}",
            )
        lim = [te[["y", "pred"]].min().min() * 0.9, te[["y", "pred"]].max().max() * 1.05]
        ax.plot(lim, lim, color="#333", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xscale("log")
        ax.set_yscale("log")
        tk = [20, 30, 50, 80, 120, 200]
        ax.set_xticks(tk, [str(t) for t in tk])
        ax.set_yticks(tk, [str(t) for t in tk])
        ax.minorticks_off()
        ax.set_xlabel("Consumo medio real (kWh)")
        ax.set_ylabel("Consumo medio pronosticado (kWh)")
        ax.legend(fontsize=9)
        self.guardar(fig, "dispersion.png")

    def importancia(self) -> None:
        imp = self.r["importancia"][::-1]
        fig, ax = plt.subplots(figsize=(9.5, 3.9))
        ax.barh(
            [NOMBRE_VARIABLE.get(v["variable"], v["variable"]) for v in imp],
            [v["media"] for v in imp],
            xerr=[v["sd"] for v in imp],
            color=RELLENO,
            edgecolor=AZUL,
        )
        ax.set_xlabel("Aumento del MAE al permutar la variable (kWh)")
        self.guardar(fig, "importancia.png")

    def mae_mes(self) -> None:
        mp, mi = self.r["mae_por_periodo"], self.r["mae_ingenuo_por_periodo"]
        pp = sorted(mp)
        xx, w = np.arange(len(pp)), 0.38
        fig, ax = plt.subplots(figsize=(10.5, 3.5))
        ax.bar(
            xx - w / 2,
            [mi[p] for p in pp],
            w,
            color="#f6dcd8",
            edgecolor=ROJO,
            label="Ingenuo (mes anterior)",
        )
        ax.bar(
            xx + w / 2, [mp[p] for p in pp], w, color=RELLENO, edgecolor=AZUL, label=self.r["mejor"]
        )
        ax.set_xticks(xx, [_etiqueta(p) for p in pp], fontsize=9)
        ax.set_ylabel("MAE (kWh)")
        ax.legend(fontsize=9.5)
        self.guardar(fig, "mae_mes.png")


def generar(
    resultados: dict[str, Any], preprocesamiento: dict[str, Any], destino: Path
) -> list[str]:
    """Dibuja las 13 figuras del informe en ``destino`` y devuelve sus nombres."""
    plt.rcParams.update(ESTILO)
    f = _Figuras(resultados, preprocesamiento, destino)
    for paso in (
        f.flujo,
        f.histograma,
        f.serie,
        f.codo,
        f.perfiles,
        f.pca,
        f.dbscan,
        f.gradiente,
        f.comparacion,
        f.regional,
        f.dispersion,
        f.importancia,
        f.mae_mes,
    ):
        paso()
    return sorted(p.name for p in destino.glob("*.png"))
