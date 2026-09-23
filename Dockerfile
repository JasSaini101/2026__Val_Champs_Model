FROM python:3.11-slim AS base

RUN pip install --no-cache-dir "uv==0.8.17"

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first so they cache independently of source changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src ./src
COPY configs ./configs
RUN uv sync --locked --no-dev

RUN useradd --create-home app && mkdir -p /app/data && chown app /app/data
USER app

ENV VALCHAMPS_DB_URL=sqlite:////app/data/valchamps.db \
    VALCHAMPS_CACHE_DIR=/app/data/raw/vlr
VOLUME ["/app/data"]

ENTRYPOINT ["valchamps"]
CMD ["--help"]
