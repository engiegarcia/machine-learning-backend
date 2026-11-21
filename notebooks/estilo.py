"""Estilo común de los gráficos de los notebooks (paleta Océano de la presentación del grupo)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

# Raíz del repositorio en sys.path para importar el paquete ``ml`` desde la carpeta notebooks/
RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

NAVY, DEEP, BLUE, TEAL, CYAN = "#03045e", "#023e8a", "#0077b6", "#0096c7", "#00b4d8"
AQUA, SKY, ICE, MUTED, GRIS = "#48cae4", "#90e0ef", "#caf0f8", "#5c6f86", "#9aa9b9"
COLOR_GRUPO = {1: AQUA, 2: BLUE, 3: NAVY}
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Set", "Oct", "Nov", "Dic"]


def _registrar_poppins() -> None:
    """Matplotlib no busca en la carpeta de fuentes del usuario de Windows: se registra a mano."""
    carpetas = [Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"]
    for carpeta in carpetas:
        for f in carpeta.glob("Poppins-*.ttf"):
            font_manager.fontManager.addfont(str(f))
    # sin Poppins se usa DejaVu Sans sin avisos
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def aplicar() -> None:
    """Fuente, rejilla suave y bordes mínimos (Poppins si está instalada)."""
    _registrar_poppins()
    plt.rcParams.update(
        {
            "font.family": ["Poppins", "DejaVu Sans"],
            "font.size": 10,
            "axes.edgecolor": "#d9eaf2",
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": "#e3eef4",
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "figure.dpi": 110,
            "figure.figsize": (10, 3.6),
        }
    )


def coma(v: float, d: int = 0) -> str:
    """Coma decimal y espacio de miles desde 10 000 (criterio del informe: 2652, 19 713 482)."""
    t = f"{v:,.{d}f}" if abs(v) >= 10_000 else f"{v:.{d}f}"
    return t.replace(",", " ").replace(".", ",")


EJE_COMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
