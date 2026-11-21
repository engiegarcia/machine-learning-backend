"""Aprendizaje no supervisado.

* k-means agrupa los distritos por su perfil mensual de consumo (2024-2025, en logaritmo y
  estandarizado); k se elige con el coeficiente de silueta y se reporta el codo (SSE). Para
  verlos en un plano, los perfiles se proyectan en sus dos primeras componentes principales.
* DBSCAN busca distrito-meses atípicos en ``[ln y, ln(y/y_{t-1}), ln(y/y_{t-12})]``
  estandarizado, con ``min_samples = 2·dimensión`` y ε en la rodilla de la curva k-distancia.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ml.config import SEMILLA


@dataclass
class ResultadoKMeans:
    k: int
    codo: dict[int, float]
    silueta: dict[int, float]
    grupo: pd.Series  # grupo (1..k, de menor a mayor consumo) por ubigeo
    perfiles: pd.DataFrame  # consumo medio por mes (columnas 1..12) por ubigeo


def perfiles_mensuales(pn: pd.DataFrame, hasta_anio: int = 2025) -> pd.DataFrame:
    """Consumo medio de cada distrito en cada mes del año; 2026 se reserva para la prueba."""
    hist = pn[pn["fecha"].dt.year <= hasta_anio]
    return hist.pivot_table(index="UBIGEO", columns="mes", values="y", aggfunc="mean")


def kmeans_distritos(pn: pd.DataFrame, ks: range = range(2, 11)) -> ResultadoKMeans:
    """k-means de los perfiles; los grupos se numeran de menor a mayor consumo medio."""
    prof = perfiles_mensuales(pn)
    x = StandardScaler().fit_transform(np.log(prof.values))
    codo, silueta = {}, {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEMILLA).fit(x)
        codo[k] = float(km.inertia_)
        silueta[k] = float(silhouette_score(x, km.labels_))
    k_sel = max(silueta, key=silueta.get)
    km = KMeans(n_clusters=k_sel, n_init=20, random_state=SEMILLA).fit(x)
    nivel = (
        pd.Series(prof.mean(axis=1).values, index=prof.index)
        .groupby(km.labels_)
        .mean()
        .sort_values()
    )
    orden = {viejo: nuevo + 1 for nuevo, viejo in enumerate(nivel.index)}
    grupo = pd.Series([orden[e] for e in km.labels_], index=prof.index, name="grupo")
    return ResultadoKMeans(int(k_sel), codo, silueta, grupo, prof)


def proyeccion_pca(res: ResultadoKMeans) -> dict[str, object]:
    """Perfiles de k-means (mismo logaritmo y estandarización) en sus dos primeras componentes.

    El signo de cada componente es arbitrario; se fija para que la componente 1 crezca con el
    consumo medio y la carga de mayor magnitud de la componente 2 sea positiva. Los centroides
    son la media de los puntos proyectados de cada grupo.
    """
    x = StandardScaler().fit_transform(np.log(res.perfiles.values))
    pca = PCA(n_components=2, random_state=SEMILLA).fit(x)
    signo = np.ones(2)
    if np.corrcoef(pca.transform(x)[:, 0], x.mean(axis=1))[0, 1] < 0:
        signo[0] = -1
    if pca.components_[1][np.argmax(np.abs(pca.components_[1]))] < 0:
        signo[1] = -1
    z = pca.transform(x) * signo
    g = res.grupo.loc[res.perfiles.index].values
    return {
        "varianza": [float(v) for v in pca.explained_variance_ratio_],
        "puntos": {
            u: [float(a), float(b)] for u, (a, b) in zip(res.perfiles.index, z, strict=True)
        },
        "centroides": {
            int(k): [float(v) for v in z[g == k].mean(axis=0)] for k in range(1, res.k + 1)
        },
    }


def resumen_grupos(res: ResultadoKMeans, pn: pd.DataFrame) -> list[dict[str, object]]:
    """Tamaño, nivel, perfil y distritos de cada grupo."""
    nom = pn.drop_duplicates("UBIGEO").set_index("UBIGEO")
    out = []
    for gk in range(1, res.k + 1):
        ids = res.grupo[res.grupo == gk].index
        p = res.perfiles.loc[ids].mean(axis=0)
        sel = pn["UBIGEO"].isin(ids) & (pn["PERIODO"] == "202512")
        out.append(
            {
                "grupo": gk,
                "n_distritos": int(len(ids)),
                "consumo_medio": float(res.perfiles.loc[ids].values.mean()),
                "perfil": [float(v) for v in p.values],
                "max_mes": int(p.idxmax()),
                "min_mes": int(p.idxmin()),
                "amplitud": float(p.max() / p.min()),
                "suministros_dic2025": int(pn.loc[sel, "n_sum"].sum()),
                "distritos": sorted(nom.loc[ids, "DIST_N"].tolist()),
                "provincias": nom.loc[ids, "PROV_N"].value_counts().to_dict(),
            }
        )
    return out


@dataclass
class ResultadoDBSCAN:
    eps: float
    min_samples: int
    kdist: np.ndarray
    filas: pd.DataFrame  # distrito-meses evaluados, con la etiqueta ``db`` (-1 = ruido)
    z: np.ndarray = field(repr=False)  # variables sin estandarizar

    @property
    def n_clusters(self) -> int:
        etiquetas = set(self.filas["db"])
        return len(etiquetas) - (1 if -1 in etiquetas else 0)

    @property
    def ruido(self) -> pd.DataFrame:
        return self.filas[self.filas["db"] == -1]


def rodilla(curva: np.ndarray) -> float:
    """Valor de la curva ordenada en el punto más alejado de la recta entre sus extremos."""
    xs = np.arange(len(curva))
    a, b, n = curva[0], curva[-1], len(curva) - 1
    dist = np.abs((b - a) * xs - n * curva + n * a) / np.hypot(b - a, n)
    return float(curva[int(np.argmax(dist))])


def dbscan_atipicos(pn: pd.DataFrame) -> ResultadoDBSCAN:
    """DBSCAN sobre los distrito-meses que tienen rezagos de 1 y 12 meses."""
    d = pn.dropna(subset=["lag1", "lag12"]).copy()
    z = np.column_stack([np.log(d["y"]), np.log(d["y"] / d["lag1"]), np.log(d["y"] / d["lag12"])])
    zs = StandardScaler().fit_transform(z)
    ms = 2 * z.shape[1]
    kd = np.sort(NearestNeighbors(n_neighbors=ms).fit(zs).kneighbors(zs)[0][:, -1])
    eps = rodilla(kd)
    d["db"] = DBSCAN(eps=eps, min_samples=ms).fit(zs).labels_
    return ResultadoDBSCAN(eps, ms, kd, d, z)
