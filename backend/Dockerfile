# Backend image: FastAPI app + migrations + embedding/reranking models.
#
# Deliberately CPU-only torch (see the --extra-index-url below). The
# default PyPI torch wheel bundles CUDA and adds several GB to the image
# for hardware this project never uses.

FROM python:3.11-slim

# PYTHONUNBUFFERED so container logs appear immediately rather than being
# held in a buffer — otherwise `docker compose logs` looks hung.
# PYTHONDONTWRITEBYTECODE because .pyc files in an image are dead weight.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    # Where sentence-transformers caches BAAI/bge-m3 and the reranker.
    # Mounted as a named volume in docker-compose.yml so the ~2.3GB
    # download survives container recreation.
    HF_HOME=/models

# Requirements are copied and installed BEFORE the application code, so
# editing a source file doesn't invalidate the (very slow) dependency
# layer.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

WORKDIR /app
COPY . .

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

# Runs migrations, then the app — see entrypoint.sh.
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
