FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime image ──────────────────────────────────────────────
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QT_X11_NO_MITSHM=1 \
    QT_QPA_PLATFORM=xcb \
    QT_DEBUG_PLUGINS=0 \
    LIBGL_ALWAYS_SOFTWARE=1

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

COPY --from=builder /install /usr/local

WORKDIR /app
COPY . .

CMD ["python", "ui_dataclean.py"]