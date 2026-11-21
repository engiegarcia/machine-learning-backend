"""Aprendizaje supervisado: pipelines, búsqueda de hiperparámetros, métricas y referencias.

Cada modelo es un ``Pipeline`` de scikit-learn: ``ColumnTransformer`` (``StandardScaler`` para
las numéricas y ``OneHotEncoder`` para las categóricas) seguido del regresor. SVR y la red
neuronal estandarizan además el objetivo (``TransformedTargetRegressor``), porque sus
hiperparámetros (ε, tasa de aprendizaje) dependen de su escala. Las métricas se calculan siempre
en kWh, también en la corrida 2, cuyo objetivo es la variación logarítmica.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from ml import caracteristicas as car
from ml.config import SEMILLA
from ml.gradiente import GDRegressor, con_intercepto, ecuacion_normal

TASAS_GD = (0.01, 0.1, 0.5)
EPOCAS_GD = 3000
TASA_GD = 0.1


def _escalado(modelo: RegressorMixin) -> TransformedTargetRegressor:
    """Estandariza el objetivo antes de ajustar ``modelo`` y deshace la escala al predecir."""
    return TransformedTargetRegressor(regressor=modelo, transformer=StandardScaler())


def _svr() -> RegressorMixin:
    return _escalado(SVR(kernel="rbf"))


def _red() -> RegressorMixin:
    return _escalado(
        MLPRegressor(max_iter=2000, early_stopping=True, n_iter_no_change=20, random_state=SEMILLA)
    )


def _xgb(**kw: Any) -> RegressorMixin:
    return XGBRegressor(random_state=SEMILLA, n_jobs=1, tree_method="hist", **kw)


# Modelos supervisados con su rejilla de hiperparámetros (en el orden de las tablas del informe).
REJILLAS: dict[str, tuple[Callable[[], RegressorMixin], dict[str, list[Any]]]] = {
    "Ridge": (lambda: Ridge(), {"m__alpha": [0.01, 0.1, 1, 10, 100]}),
    "Lasso": (
        lambda: Lasso(max_iter=50_000),
        {"m__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]},
    ),
    "Árbol de decisión": (
        lambda: DecisionTreeRegressor(random_state=SEMILLA),
        {"m__max_depth": [3, 5, 10, None], "m__min_samples_leaf": [1, 5, 10]},
    ),
    "KNN": (
        lambda: KNeighborsRegressor(),
        {"m__n_neighbors": [3, 5, 10, 20, 40], "m__weights": ["uniform", "distance"]},
    ),
    "SVR": (
        _svr,
        {"m__regressor__C": [0.1, 1, 10, 100], "m__regressor__epsilon": [0.01, 0.1, 0.5]},
    ),
    "Red neuronal (MLP)": (
        _red,
        {
            "m__regressor__hidden_layer_sizes": [(32,), (64, 32)],
            "m__regressor__alpha": [0.0001, 0.01, 1],
        },
    ),
    "Random Forest": (
        lambda: RandomForestRegressor(random_state=SEMILLA, n_jobs=-1),
        {
            "m__n_estimators": [200, 500],
            "m__max_depth": [None, 10, 20],
            "m__min_samples_leaf": [1, 3, 5],
        },
    ),
    "Gradient Boosting": (
        lambda: GradientBoostingRegressor(random_state=SEMILLA),
        {
            "m__n_estimators": [200, 500],
            "m__learning_rate": [0.03, 0.1],
            "m__max_depth": [2, 3, 4],
        },
    ),
    "XGBoost": (
        _xgb,
        {
            "m__n_estimators": [200, 500],
            "m__learning_rate": [0.03, 0.1],
            "m__max_depth": [3, 6],
            "m__subsample": [0.8, 1.0],
        },
    ),
}

# Validación cruzada k = 5 con los valores por defecto de cada modelo; Ridge y Lasso usan el α
# elegido en la búsqueda, porque su valor por defecto (α = 1) no tiene sentido en ambas escalas.
POR_DEFECTO: dict[str, Callable[[dict[str, Any]], RegressorMixin]] = {
    "Regresión lineal": lambda e: LinearRegression(),
    "Ridge": lambda e: Ridge(alpha=e["Ridge"]["alpha"]),
    "Lasso": lambda e: Lasso(alpha=e["Lasso"]["alpha"], max_iter=50_000),
    "Árbol de decisión": lambda e: DecisionTreeRegressor(random_state=SEMILLA),
    "KNN": lambda e: KNeighborsRegressor(),
    "SVR": lambda e: _svr(),
    "Red neuronal (MLP)": lambda e: _red(),
    "Random Forest": lambda e: RandomForestRegressor(random_state=SEMILLA, n_jobs=-1),
    "Gradient Boosting": lambda e: GradientBoostingRegressor(random_state=SEMILLA),
    "XGBoost": lambda e: _xgb(),
}


def nombre_parametro(clave: str) -> str:
    """``m__regressor__C`` → ``C``."""
    return clave.replace("m__", "").replace("regressor__", "")


def crear(nombre: str, params: dict[str, Any]) -> RegressorMixin:
    """Modelo ``nombre`` con los hiperparámetros elegidos (claves sin prefijos)."""
    modelo = REJILLAS[nombre][0]()
    if isinstance(modelo, TransformedTargetRegressor):
        modelo.regressor.set_params(**params)
    else:
        modelo.set_params(**params)
    return modelo


def preprocesador(f: int) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), car.numericas(f)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                car.categoricas(f),
            ),
        ]
    )


def pipeline(f: int, modelo: RegressorMixin) -> Pipeline:
    return Pipeline([("p", preprocesador(f)), ("m", modelo)])


def metricas(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
    }


def ingenuos(te: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Referencias: mes anterior, mismo mes del año anterior y estacional ajustado."""
    y = te["y"].values
    return {
        "Ingenuo (mes anterior)": {"test": metricas(y, te["lag1"])},
        "Ingenuo estacional (mismo mes del año anterior)": {"test": metricas(y, te["lag12"])},
        "Ingenuo estacional ajustado": {
            "test": metricas(y, te["lag1"] * te["lag12"] / te["lag13"])
        },
    }


def _gradiente(f: int, a: np.ndarray, b: np.ndarray, t: np.ndarray, te: pd.DataFrame) -> dict:
    """Ecuación normal frente a gradiente descendiente con varias tasas de aprendizaje."""
    yte = te["y"].values
    out: dict[str, Any] = {}
    t0 = time.time()
    theta = ecuacion_normal(a, t)
    out["Regresión lineal (ecuación normal)"] = {
        "test": metricas(yte, car.a_kwh(f, con_intercepto(b) @ theta, te)),
        "segundos": time.time() - t0,
    }
    curvas = {str(lr): GDRegressor(lr=lr, epochs=EPOCAS_GD).fit(a, t).costo for lr in TASAS_GD}
    t0 = time.time()
    gd = GDRegressor(lr=TASA_GD, epochs=EPOCAS_GD).fit(a, t)
    out["Regresión lineal (gradiente descendiente)"] = {
        "test": metricas(yte, car.a_kwh(f, gd.predict(b), te)),
        "params": {"lr": TASA_GD, "epochs": EPOCAS_GD},
        "segundos": time.time() - t0,
    }
    mse_ne = float(np.mean((con_intercepto(a) @ theta - t) ** 2))
    conv = next(
        (i for i, c in enumerate(curvas[str(TASA_GD)]) if c is not None and c <= mse_ne * 1.0001),
        None,
    )
    out["_gd"] = {
        "curvas": {k: v[:EPOCAS_GD:10] for k, v in curvas.items()},
        "final": {k: v[-1] for k, v in curvas.items()},
        "iter_diverge": {k: (len(v) if v[-1] is None else None) for k, v in curvas.items()},
        "mse_ne": mse_ne,
        "dif_max_theta": float(np.max(np.abs(gd.theta - theta))),
        f"iter_converge_{TASA_GD}": conv,
        "n_parametros": int(len(theta)),
    }
    return out


def evaluar(f: int, tr: pd.DataFrame, te: pd.DataFrame) -> dict[str, Any]:
    """Entrena todos los modelos de la formulación ``f`` y los evalúa en ``te`` (kWh).

    Las claves que empiezan con ``_`` guardan objetos internos (modelo, predicciones, curvas,
    validación cruzada) que no forman parte de la tabla de resultados.
    """
    entradas, t = car.entradas(f), car.objetivo(f)
    xtr, xte, ttr = tr[entradas], te[entradas], tr[t].values
    yte = te["y"].values
    pre = preprocesador(f).fit(xtr)
    out = _gradiente(f, pre.transform(xtr), pre.transform(xte), ttr, te)

    cv = KFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    for nombre, (crear, rejilla) in REJILLAS.items():
        t0 = time.time()
        gs = GridSearchCV(
            pipeline(f, crear()), rejilla, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
        ).fit(xtr, ttr)
        p = car.a_kwh(f, gs.predict(xte), te)
        out[nombre] = {
            "test": metricas(yte, p),
            "params": {nombre_parametro(k): v for k, v in gs.best_params_.items()},
            "n_combinaciones": len(gs.cv_results_["params"]),
            "segundos": time.time() - t0,
            "_model": gs.best_estimator_,
            "_pred": p,
        }

    elegidos = {k: v["params"] for k, v in out.items() if k in REJILLAS}
    cvres = {}
    for nombre, por_defecto in POR_DEFECTO.items():
        ms = []
        for itr, iva in cv.split(xtr):
            m = pipeline(f, por_defecto(elegidos)).fit(xtr.iloc[itr], ttr[itr])
            va = tr.iloc[iva]
            ms.append(metricas(va["y"].values, car.a_kwh(f, m.predict(xtr.iloc[iva]), va)))
        cvres[nombre] = {k: float(np.mean([m_[k] for m_ in ms])) for k in ("MAE", "RMSE", "R2")}
        cvres[nombre].update(
            {k + "_sd": float(np.std([m_[k] for m_ in ms])) for k in ("MAE", "R2")}
        )
    out["_cv"] = cvres
    return out


def tabla(res: dict[str, Any]) -> dict[str, Any]:
    """Resultados serializables: quita los objetos internos de cada modelo."""
    return {
        k: (v if k.startswith("_") else {kk: vv for kk, vv in v.items() if not kk.startswith("_")})
        for k, v in res.items()
    }


def importancia_permutacion(
    modelo: Pipeline, f: int, te: pd.DataFrame, repeticiones: int = 20, semilla: int = SEMILLA
) -> list[dict[str, float | str]]:
    """Aumento del MAE (kWh) al permutar cada variable de entrada en el conjunto de prueba."""
    entradas = car.entradas(f)
    rng = np.random.default_rng(semilla)
    base = mean_absolute_error(te["y"], car.a_kwh(f, modelo.predict(te[entradas]), te))
    imp = []
    for c in entradas:
        deltas = []
        for _ in range(repeticiones):
            xp = te[entradas].copy()
            xp[c] = rng.permutation(xp[c].values)
            deltas.append(mean_absolute_error(te["y"], car.a_kwh(f, modelo.predict(xp), te)) - base)
        imp.append({"variable": c, "media": float(np.mean(deltas)), "sd": float(np.std(deltas))})
    return sorted(imp, key=lambda r: -r["media"])
