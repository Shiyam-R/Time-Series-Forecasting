# ═══════════════════════════════════════════════════════════════════════════
# Dockerfile — Rossmann Forecasting API
#
# Multi-stage build:
#   1. "builder"  — installs dependencies into an isolated virtualenv
#   2. "runtime"  — copies only the finished venv + app code into a clean
#                   base image, runs as a non-root user
#
# Build:  docker build -t rossmann-forecasting-api .
# Run:    docker run -p 8000:8000 rossmann-forecasting-api
# Docs:   http://localhost:8000/docs
# ═══════════════════════════════════════════════════════════════════════════


# ── Stage 1: builder ──────────────────────────────────────────────────────────
# python:3.13-slim — matches your local interpreter. Every pin in
# requirements.txt was verified (by actually resolving against PyPI, not
# just reading changelogs) to have a working cp313 wheel, so this installs
# via prebuilt wheels rather than falling back to slow/fragile source builds.
FROM python:3.13-slim AS builder

# PYTHONDONTWRITEBYTECODE   — skip writing .pyc files; irrelevant in a
#                             throwaway build stage and just adds noise.
# PYTHONUNBUFFERED          — flush stdout/stderr immediately so build logs
#                             stream in real time instead of being buffered.
# PIP_NO_CACHE_DIR          — don't let pip cache wheels on disk; the whole
#                             stage is discarded later anyway.
# PIP_DISABLE_PIP_VERSION_CHECK — skip pip's self-update network check,
#                             which only slows down the build.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# build-essential covers the rare case where a dependency (or one of its
# transitive dependencies) has no prebuilt wheel for this platform and needs
# to compile from source. This entire layer — and the compiler itself —
# is discarded once the builder stage ends, so it never bloats the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install into an isolated virtual environment rather than the system
# site-packages. This lets stage 2 copy ONE clean directory (/opt/venv)
# without dragging along pip's cache, apt's package lists, or anything
# else that accumulated in this stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy ONLY the dependency manifest first. Docker caches each layer by its
# inputs — as long as requirements.txt is unchanged, this (slow) layer is
# reused on every rebuild, even after you edit app/*.py. Copying the full
# source tree before this step would invalidate the cache on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Running as root inside a container is a standard security-audit finding —
# a compromised process shouldn't have root privileges. Create a dedicated,
# unprivileged user for the app to run as.
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Bring in the pre-built virtual environment from stage 1. No compiler,
# no apt cache, no pip cache — just the installed packages.
COPY --from=builder /opt/venv /opt/venv

# Application source code.
COPY app/ ./app/

# Only artifacts actually read by app/model_loader.py and app/config.py:
#   artifacts/models/{near,mid,far,extended}_model.pkl   (~8.4 MB total)
#   artifacts/feature_columns.json
#   artifacts/store_metadata.json
#   artifacts/lag_defaults.json
# The ~230 MB of superseded experiment .pkl files at the artifacts/ root
# (xgb_horizon_far.pkl, xgb_fair_eval_mid.pkl, etc.) are excluded via
# .dockerignore and never reach the build context.
COPY artifacts/ ./artifacts/

# app/config.py creates LOGS_DIR at import time. Pre-create it here and
# hand ownership to appuser so the app isn't left trying to write logs
# as root, or failing to write them at all once USER is switched below.
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Uses Python's own stdlib (urllib) instead of installing curl, keeping the
# final image free of an extra package just for health probing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# Must be run as a module path (app.main:app) from /app, matching how the
# project's imports resolve — see app/main.py's own docstring. This is the
# same command you'd run locally with `uvicorn app.main:app --reload`.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
