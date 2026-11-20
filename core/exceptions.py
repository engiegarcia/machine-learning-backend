"""Excepciones de dominio.

Todas heredan de :class:`AprendizajeError` para que la API las capture con una sola clase
base. Cada una expone un ``codigo`` estable (``error_code`` HTTP) y su estado HTTP.
"""

from __future__ import annotations


class AprendizajeError(Exception):
    """Error base del proyecto."""

    codigo: str = "ERROR_APRENDIZAJE"
    estado: int = 500


class DistritoNoEncontradoError(AprendizajeError):
    """El ubigeo no está en el panel distrito-mes."""

    codigo = "DISTRITO_NO_ENCONTRADO"
    estado = 404


class ArtefactoFaltanteError(AprendizajeError):
    """Falta el panel o los resultados (``python cli.py preparar`` / ``corridas``)."""

    codigo = "ARTEFACTO_FALTANTE"
    estado = 503
