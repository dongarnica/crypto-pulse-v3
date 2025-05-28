# Multi-stage Dockerfile for Crypto Pulse V3 Trading System
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Create non-root user
RUN groupadd -r cryptopulse && useradd -r -g cryptopulse cryptopulse

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/backtest_cache /app/data /app/scripts \
    && chown -R cryptopulse:cryptopulse /app

# Set proper permissions
RUN chmod +x /app/scripts/*.py 2>/dev/null || true
RUN chmod +x /app/*.py

# Switch to non-root user
USER cryptopulse

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app'); from src.monitoring.health_check import check_health; exit(0 if check_health() else 1)" || exit 1

# Default command
CMD ["python", "main.py"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio black isort flake8 mypy jupyter

# Copy application code
COPY . .

# Create directories with proper permissions
RUN mkdir -p /app/logs /app/backtest_cache /app/data /app/scripts

# Expose ports
EXPOSE 8000 8001 8888

# Default command for development
CMD ["python", "main.py"]

# Backtesting stage
FROM base as backtesting

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/logs /app/backtest_cache /app/data /app/scripts

# Set entrypoint for backtesting
ENTRYPOINT ["python"]
CMD ["working_advanced_cli.py", "--help"]
