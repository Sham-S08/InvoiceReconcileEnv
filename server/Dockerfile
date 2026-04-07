FROM ghcr.io/meta-pytorch/openenv-base:latest AS builder

WORKDIR /app

COPY . /app/env

WORKDIR /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        pip install --no-cache-dir \
        "openenv-core[core]>=0.2.2" \
        "fastapi>=0.115.0" \
        "uvicorn>=0.24.0" \
        "pydantic>=2.0.0" \
        "requests>=2.31.0" \
        "openai>=1.0.0"; \
    fi

FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

COPY --from=builder /app/env /app/env

ENV PYTHONPATH="/app/env:/app/env/server:${PYTHONPATH}"

ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

EXPOSE 7860

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 7860"]