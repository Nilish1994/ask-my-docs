# API image. The FastAPI docs index and the ONNX embedding model are built
# into the image, so the container starts without network access to GitHub or
# Hugging Face. ANTHROPIC_API_KEY is supplied at runtime via .env, never baked in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    FASTEMBED_CACHE_PATH=/app/models

RUN pip install --no-cache-dir uv==0.13.8 \
    && useradd --create-home --uid 10001 app

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY ingest ./ingest
COPY retrieval ./retrieval
COPY answer ./answer
COPY app ./app
COPY eval ./eval

# Fetch the docs at the pinned tag, chunk, embed, write data/index.
RUN python -m ingest.build_index && rm -rf data/raw && chown -R app:app /app

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
