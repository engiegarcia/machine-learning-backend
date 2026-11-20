"""Aplicación FastAPI del pronóstico del consumo eléctrico residencial por distrito.

Ejecución: ``uvicorn main:app --reload`` (dev) o ``uvicorn main:app --host 0.0.0.0`` (prod).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import APP_NAME, APP_VERSION, get_settings
from core.exceptions import AprendizajeError
from core.logging_config import configure_logging
from routers.api import router as api_router
from schemas.ml import ErrorResponse
from services.ml_service import MLService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configura logging y crea ``MLService`` una sola vez (en ``app.state``)."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    logger.info("Iniciando backend", extra={"environment": settings.environment})
    app.state.servicio = MLService(settings)
    yield
    logger.info("Backend detenido")


def _formatear_loc(loc: tuple[Any, ...]) -> str:
    partes = [str(p) for p in loc if p not in ("body", "query", "path")]
    return ".".join(partes) or "(cuerpo)"


async def _validacion(_: Request, exc: RequestValidationError) -> JSONResponse:
    """422 con formato ``ErrorResponse``."""
    errores = [
        {"campo": _formatear_loc(tuple(e["loc"])), "mensaje": str(e["msg"]), "tipo": str(e["type"])}
        for e in exc.errors()
    ]
    detalle = "; ".join(f"{e['campo']}: {e['mensaje']}" for e in errores)
    cuerpo = ErrorResponse(
        detail=f"Entrada inválida: {detalle}", error_code="VALIDATION_ERROR", errors=errores
    )
    return JSONResponse(status_code=422, content=cuerpo.model_dump(exclude_none=True))


async def _error_dominio(_: Request, exc: Exception) -> JSONResponse:
    """Errores del dominio con su estado HTTP (404 ubigeo desconocido, 503 sin artefactos)."""
    codigo = getattr(exc, "codigo", AprendizajeError.codigo)
    estado = getattr(exc, "estado", AprendizajeError.estado)
    cuerpo = ErrorResponse(detail=str(exc), error_code=codigo)
    return JSONResponse(status_code=estado, content=cuerpo.model_dump(exclude_none=True))


async def _error_inesperado(_: Request, exc: Exception) -> JSONResponse:
    """Cualquier otra excepción → 500 genérico (sin filtrar detalles internos)."""
    logger.error("Excepción no controlada", exc_info=exc)
    cuerpo = ErrorResponse(detail="Error interno del servidor", error_code="ERROR_INTERNO")
    return JSONResponse(status_code=500, content=cuerpo.model_dump(exclude_none=True))


def create_app() -> FastAPI:
    """Construye la aplicación FastAPI."""
    settings = get_settings()
    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description=(
            "Pronóstico del consumo eléctrico medio residencial (tarifa BT5B) por distrito de "
            "Piura y Tumbes con los datos abiertos de ENOSA, más k-means de distritos y DBSCAN "
            "de meses atípicos (ODS 7). Proyecto UNMSM · Sistemas Inteligentes · Grupo 6."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_exception_handler(RequestValidationError, _validacion)  # type: ignore[arg-type]
    app.add_exception_handler(AprendizajeError, _error_dominio)
    app.add_exception_handler(Exception, _error_inesperado)
    app.include_router(api_router)

    @app.get("/health", tags=["salud"])
    def health(request: Request) -> JSONResponse:
        """200 si el panel, los resultados y el modelo de pronóstico están cargados."""
        servicio: MLService = request.app.state.servicio
        sano = servicio.health_check()
        return JSONResponse(
            status_code=200 if sano else 503,
            content={"status": "ok" if sano else "degraded", "version": APP_VERSION},
        )

    return app


app = create_app()
