"""Regresión lineal propia: gradiente descendiente de lote completo y ecuación normal.

Ambos minimizan el error cuadrático medio ``J(θ) = (1/m)·‖Xθ − y‖²`` con un término
independiente. La ecuación normal da la solución exacta ``θ = (XᵀX)⁺ Xᵀ y`` (pseudoinversa,
porque las variables indicadoras de la codificación one-hot son colineales con el intercepto).
"""

from __future__ import annotations

import numpy as np


def con_intercepto(x: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(x)), x]


def ecuacion_normal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Parámetros ``θ`` (el primero es el intercepto)."""
    a = con_intercepto(x)
    return np.linalg.pinv(a.T @ a) @ a.T @ y


class GDRegressor:
    """Regresión lineal por gradiente descendiente de lote completo sobre el MSE.

    Args:
        lr: Tasa de aprendizaje ``α``.
        epochs: Número de iteraciones.

    Attributes:
        theta: Parámetros aprendidos (el primero es el intercepto).
        costo: ``J(θ)`` en cada iteración; ``None`` si diverge (y entonces se detiene).
    """

    def __init__(self, lr: float = 0.1, epochs: int = 3000) -> None:
        self.lr, self.epochs = lr, epochs

    def fit(self, x: np.ndarray, y: np.ndarray) -> GDRegressor:
        a = con_intercepto(x)
        m = len(y)
        self.theta = np.zeros(a.shape[1])
        self.costo: list[float | None] = []
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(self.epochs):
                err = a @ self.theta - y
                c = float((err @ err) / m)
                if not np.isfinite(c):
                    self.costo.append(None)
                    break
                self.costo.append(c)
                self.theta = self.theta - self.lr * (2 / m) * (a.T @ err)
        return self

    @property
    def diverge(self) -> bool:
        return bool(self.costo) and self.costo[-1] is None

    def predict(self, x: np.ndarray) -> np.ndarray:
        return con_intercepto(x) @ self.theta
