# LEXe API - Legal Tools Service
# Multi-stage build for smaller production image

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.12-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install dependencies
RUN uv pip install --system --no-cache .

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.12-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash lexe

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=lexe:lexe src/ /app/src/

# Switch to non-root user
USER lexe

# Environment
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    LEXE_API_HOST=0.0.0.0 \
    LEXE_API_PORT=8020

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8020/health/live || exit 1

# Expose port
EXPOSE 8020

# Run the application
CMD ["python", "-m", "uvicorn", "lexe_api.main:app", "--host", "0.0.0.0", "--port", "8020"]
