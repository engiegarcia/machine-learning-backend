FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# NumPy, pandas, pyarrow y scikit-learn publican wheels para python:3.11-slim:
# no hacen falta compiladores.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Solo lo que usa la API: el panel procesado y los resultados vienen en el repositorio.
COPY main.py cli.py ./
COPY core ./core
COPY ml ./ml
COPY routers ./routers
COPY schemas ./schemas
COPY services ./services
COPY datos/procesados ./datos/procesados
COPY resultados/resultados.json ./resultados/resultados.json

# Usuario no root
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 10000

# El modelo se entrena al iniciar (unos segundos): start-period holgado.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, sys, urllib.request; port = os.environ.get('PORT', '10000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=4).status == 200 else 1)"

# Render inyecta PORT (10000 por defecto).
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
