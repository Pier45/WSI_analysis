# syntax=docker/dockerfile:1.7

# ── Builder: install deps into a clean venv ────────────────────
FROM python:3.10-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Cache-mount decisive wins: wheels are downloaded ONCE per host, not per build.
# Layer order: deps before code → code-only edits never invalidate the wheel layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ── Runtime ────────────────────────────────────────────────────
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QT_X11_NO_MITSHM=1 \
    QT_QPA_PLATFORM=xcb \
    QT_DEBUG_PLUGINS=0 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    PATH="/app/.venv/bin:$PATH"

# Qt/X11 runtime libs — unchanged, but placed in one layer w/ clean cache.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 libx11-xcb1 libxcb1 libxcb-util1 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render0 libxcb-render-util0 \
    libxcb-shape0 libxcb-shm0 libxcb-sync1 \
    libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 \
    libxrender1 libxrandr2 libxinerama1 \
    libxcursor1 libxfixes3 libxtst6 \
    libfontconfig1 libfreetype6 libgl1 libglu1-mesa libegl1 \
    libxkbcommon0 libxkbcommon-x11-0 \
    libdbus-1-3 libglib2.0-0 \
    ca-certificates x11-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the prebuilt venv from builder (only deps, no source).
COPY --from=builder /app/.venv /app/.venv

WORKDIR /app

# Copy source LAST — code edits never invalidate the .venv layer above.
COPY . .

# Run as non-root.
RUN useradd --create-home --uid 1001 wsi && chown -R wsi:wsi /app
USER wsi

# Default to the dataclean GUI; override per-service in docker-compose.
CMD ["python", "ui_dataclean.py"]