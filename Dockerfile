# Multi-stage Dockerfile for MCP XGBoost Server
# Stage 1: Build stage with all dependencies
FROM python:3.12-slim AS builder

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy lock files for the target platform
COPY environments/ ./environments/

# Install conda with automatic platform detection
ARG TARGETPLATFORM
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        CONDA_ARCH="aarch64"; \
    else \
        CONDA_ARCH="x86_64"; \
    fi && \
    curl -o ~/miniconda.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${CONDA_ARCH}.sh" && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH

# Create conda environment using platform-specific lock file
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        CONDA_PLATFORM="linux-aarch64"; \
    else \
        CONDA_PLATFORM="linux-64"; \
    fi && \
    if [ -f "environments/development.lock-${CONDA_PLATFORM}.yml" ]; then \
        echo "Using platform-specific lock file: development.lock-${CONDA_PLATFORM}.yml"; \
        conda env create -f "environments/development.lock-${CONDA_PLATFORM}.yml" -p /opt/conda/envs/mcp-xgboost; \
    else \
        echo "Platform-specific lock file not found, using generic environment file"; \
        conda env create -f environments/development.yml -p /opt/conda/envs/mcp-xgboost; \
    fi

# Stage 2: Production stage
FROM python:3.12-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy conda environment from builder
COPY --from=builder /opt/conda /opt/conda

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH
ENV PATH=/opt/conda/envs/mcp-xgboost/bin:$PATH

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY pyproject.toml ./

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Environment variables
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "-m", "src.app"] 