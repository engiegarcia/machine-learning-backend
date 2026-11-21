"""Endpoints REST bajo el prefijo ``/api``.

Los endpoints son ``def`` (no ``async def``): la predicción es CPU-bound y FastAPI los ejecuta
en su threadpool.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Request

from schemas.ml import (
    AtipicosResponse,
    Distrito,
    DistritoDetalle,
    ErrorResponse,
    GruposResponse,
    ModelosResponse,
    PrediccionesResponse,
    PronosticoResponse,
    RegionalResponse,
    ResumenResponse,
)
from services.ml_service import MLService

router = APIRouter(prefix="/api", tags=["aprendizaje automático"])

_NO_ENCONTRADO = {404: {"model": ErrorResponse, "description": "Ubigeo desconocido"}}
Ubigeo = Annotated[str, Path(pattern=r"^\d{6}$", examples=["200101"])]


def get_servicio(request: Request) -> MLService:
    """Dependencia: servicio creado una sola vez en el ``lifespan``."""
    servicio: MLService = request.app.state.servicio
    return servicio


Servicio = Annotated[MLService, Depends(get_servicio)]


@router.get("/resumen", response_model=ResumenResponse)
def resumen(servicio: Servicio) -> ResumenResponse:
    """Fuente de datos, conteos del preprocesamiento y desempeño del modelo adoptado."""
    return servicio.resumen()


@router.get("/distritos", response_model=list[Distrito])
def distritos(servicio: Servicio) -> list[Distrito]:
    """Los distritos del panel con su grupo k-means y su último consumo medio."""
    return servicio.distritos()


@router.get("/distritos/{ubigeo}", response_model=DistritoDetalle, responses=_NO_ENCONTRADO)
def distrito(ubigeo: Ubigeo, servicio: Servicio) -> DistritoDetalle:
    """Serie mensual, pronósticos de la prueba temporal 2026 y pronóstico del mes siguiente."""
    return servicio.distrito(ubigeo)


@router.get("/grupos", response_model=GruposResponse)
def grupos(servicio: Servicio) -> GruposResponse:
    """k-means: codo, silueta y perfil mensual de cada grupo de distritos."""
    return servicio.grupos()


@router.get("/atipicos", response_model=AtipicosResponse)
def atipicos(servicio: Servicio) -> AtipicosResponse:
    """DBSCAN: parámetros y distrito-meses marcados como ruido."""
    return servicio.atipicos()


@router.get("/modelos", response_model=ModelosResponse)
def modelos(servicio: Servicio) -> ModelosResponse:
    """Métricas de las corridas 1 y 2 en ambos esquemas y análisis del modelo adoptado."""
    return servicio.modelos()


@router.get("/predicciones", response_model=PrediccionesResponse)
def predicciones(servicio: Servicio) -> PrediccionesResponse:
    """Pronósticos de cada modelo en la prueba temporal 2026 (corrida 2)."""
    return servicio.predicciones()


@router.get("/regional", response_model=RegionalResponse)
def regional(servicio: Servicio) -> RegionalResponse:
    """Serie regional mensual y pronóstico agregado de la prueba temporal 2026."""
    return servicio.regional()


@router.get("/pronostico", response_model=list[PronosticoResponse])
def pronosticos(servicio: Servicio) -> list[PronosticoResponse]:
    """Pronóstico del mes siguiente para todos los distritos."""
    return servicio.pronosticos()


@router.get("/pronostico/{ubigeo}", response_model=PronosticoResponse, responses=_NO_ENCONTRADO)
def pronostico(
    ubigeo: Ubigeo,
    servicio: Servicio,
    dias: Annotated[
        float | None, Query(ge=20, le=40, description="Días facturados supuestos del mes")
    ] = None,
) -> PronosticoResponse:
    """Pronóstico del mes siguiente para un distrito, con días facturados opcionales."""
    return servicio.pronostico(ubigeo, dias)
