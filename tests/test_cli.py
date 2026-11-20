"""Línea de comandos."""

from __future__ import annotations

import pytest

from cli import construir_parser


@pytest.mark.parametrize("comando", ["descargar", "preparar", "corridas", "figuras", "pronosticar"])
def test_comandos(comando):
    args = construir_parser().parse_args([comando])
    assert args.comando == comando and callable(args.func)


def test_pronosticar_con_opciones():
    args = construir_parser().parse_args(["pronosticar", "--ubigeo", "200101", "--dias", "31"])
    assert args.ubigeo == "200101" and args.dias == 31.0


def test_comando_obligatorio():
    with pytest.raises(SystemExit):
        construir_parser().parse_args([])
