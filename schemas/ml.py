"""Esquemas Pydantic de las respuestas de la API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    detail: str
    error_code: str
    errors: list[dict[str, str]] | None = None


class Fuente(BaseModel):
    conjunto: str
    publicador: str
    licencia: str
    url: str


class PasoFiltrado(BaseModel):
    paso: str
    registros: int


class ResumenResponse(BaseModel):
    """Datos, preprocesamiento y modelo adoptado."""

    fuente: Fuente
    periodos: list[str]
    registros_descargados: int
    filtrado: list[PasoFiltrado]
    recibos_activos_limpios: int
    proporcion_recibos_cero: float
    limites_tukey_kwh: dict[str, float]
    distritos: int
    celdas: int
    modelo_adoptado: str
    mae_modelo: float
    mae_ingenuo: float
    mejora_frente_ingenuo: float


class Distrito(BaseModel):
    ubigeo: str = Field(examples=["200101"])
    distrito: str
    provincia: str
    departamento: str
    grupo: int = Field(description="Grupo k-means (1 = menor consumo)")
    ultimo_periodo: str
    ultimo_consumo: float = Field(description="kWh por suministro activo en el último mes")
    suministros: int


class PuntoSerie(BaseModel):
    periodo: str
    consumo: float | None
    mediana: float | None
    suministros: int
    activos: int | None
    proporcion_cero: float
    dias: float
    atipico: bool = Field(description="Ruido según DBSCAN")


class PuntoPrueba(BaseModel):
    periodo: str
    real: float
    pronostico: float
    ingenuo: float


class PronosticoResponse(BaseModel):
    ubigeo: str
    distrito: str
    periodo: str
    ultimo_periodo: str
    ultimo_consumo: float
    consumo: float = Field(description="kWh por suministro activo pronosticados")
    variacion: float = Field(description="Variación relativa frente al último mes")
    dias_ultimo: float
    dias_supuestos: float
    supuestos: list[str]


class DistritoDetalle(Distrito):
    serie: list[PuntoSerie]
    prueba_2026: list[PuntoPrueba]
    pronostico: PronosticoResponse


class Grupo(BaseModel):
    grupo: int
    n_distritos: int
    consumo_medio: float
    perfil: list[float] = Field(description="kWh medios de enero a diciembre (2024-2025)")
    max_mes: int
    min_mes: int
    amplitud: float
    suministros_dic2025: int
    distritos: list[str]
    provincias: dict[str, int]


class PuntoPCA(BaseModel):
    ubigeo: str
    distrito: str
    provincia: str
    grupo: int
    x: float = Field(description="Componente principal 1")
    y: float = Field(description="Componente principal 2")


class CentroidePCA(BaseModel):
    grupo: int
    x: float
    y: float


class ProyeccionPCA(BaseModel):
    varianza: list[float] = Field(description="Proporción de varianza de cada componente")
    puntos: list[PuntoPCA]
    centroides: list[CentroidePCA]


class GruposResponse(BaseModel):
    k: int
    silueta: dict[int, float]
    codo: dict[int, float]
    grupos: list[Grupo]
    pca: ProyeccionPCA


class Atipico(BaseModel):
    ubigeo: str
    distrito: str
    provincia: str
    periodo: str
    y: float
    lag1: float
    lag12: float
    var_mes: float
    var_anio: float
    n_act: int
    grupo: int


class PuntoDBSCAN(BaseModel):
    ubigeo: str
    distrito: str
    periodo: str
    log_mes: float = Field(description="ln(y / y del mes anterior)")
    log_anio: float = Field(description="ln(y / y del mismo mes del año anterior)")
    ruido: bool


class AtipicosResponse(BaseModel):
    eps: float
    min_samples: int
    evaluados: int
    clusters: int
    atipicos: list[Atipico]
    por_periodo: dict[str, int]
    kdist: list[float] = Field(description="Distancia al min_samples-ésimo vecino, ordenada")
    puntos: list[PuntoDBSCAN]


class Metrica(BaseModel):
    modelo: str
    MAE: float
    RMSE: float
    R2: float
    params: dict[str, Any] | None = None


class Corrida(BaseModel):
    corrida: int
    esquema: str
    descripcion: str
    entrenamiento: int
    prueba: int
    prueba_metricas: list[Metrica]
    validacion_cruzada: list[Metrica]
    gradiente: dict[str, Any]


class PuntoDispersion(BaseModel):
    ubigeo: str
    distrito: str
    periodo: str
    grupo: int
    real: float
    pronostico: float


class ModelosResponse(BaseModel):
    corridas: list[Corrida]
    adoptado: str
    importancia: list[dict[str, Any]]
    mae_por_grupo: dict[int, dict[str, float]]
    mae_por_periodo: dict[str, dict[str, float]]
    mape: dict[str, float]
    real_vs_pronostico: list[PuntoDispersion] = Field(
        description="Prueba temporal 2026 del modelo adoptado (corrida 2)"
    )


class PuntoRegional(BaseModel):
    periodo: str
    media: float
    mediana: float
    recibos_activos: int
    gwh: float
    pronostico: float | None = None
    ingenuo: float | None = None


class RegionalResponse(BaseModel):
    serie: list[PuntoRegional]
