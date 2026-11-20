"""Configuración de la aplicación vía variables de entorno / ``.env``."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from ml import config as ml_config

APP_NAME = "Pronóstico del Consumo Eléctrico Residencial por Distrito (ENOSA)"
APP_VERSION = "1.0.0"


class Settings(BaseSettings):
    """Parámetros de ejecución del backend.

    Attributes:
        cors_allowed_origins: Orígenes permitidos por CORS (lista JSON o separada por comas).
        log_level: Nivel de logging.
        log_json: Si es ``True`` los logs se emiten como una línea JSON.
        environment: ``dev`` o ``prod``.
        panel: Ruta del panel distrito-mes.
        resultados: Ruta de los resultados de las corridas.
        preprocesamiento: Ruta del resumen del preprocesamiento.
        arboles_pronostico: Si se indica, reemplaza el número de árboles del modelo adoptado
            (útil en pruebas o servidores pequeños; por defecto, el elegido en la búsqueda).
        n_jobs: Núcleos para entrenar el modelo al iniciar.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    cors_allowed_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]
    log_level: str = "INFO"
    log_json: bool = False
    environment: Literal["dev", "prod"] = "dev"
    panel: Path = ml_config.PANEL
    resultados: Path = ml_config.RESULTADOS
    preprocesamiento: Path = ml_config.RESUMEN_PREPROCESAMIENTO
    arboles_pronostico: int | None = None
    n_jobs: int = 1

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _parsear_origenes(cls, valor: Any) -> Any:
        """Acepta lista JSON o cadena separada por comas y quita la barra final."""
        if isinstance(valor, str):
            texto = valor.strip()
            valor = json.loads(texto) if texto.startswith("[") else texto.split(",")
        if isinstance(valor, list):
            return [str(item).strip().rstrip("/") for item in valor if str(item).strip()]
        return valor

    @field_validator("log_level")
    @classmethod
    def _validar_nivel(cls, valor: str) -> str:
        nivel = valor.upper()
        if nivel not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError(f"log_level inválido: {valor!r}")
        return nivel

    @field_validator("arboles_pronostico", "n_jobs")
    @classmethod
    def _validar_positivo(cls, valor: int | None) -> int | None:
        if valor is not None and valor < 1 and valor != -1:
            raise ValueError("debe ser >= 1 (o -1 para n_jobs)")
        return valor


@lru_cache
def get_settings() -> Settings:
    """Instancia única de :class:`Settings` (``get_settings.cache_clear()`` en pruebas)."""
    return Settings()
