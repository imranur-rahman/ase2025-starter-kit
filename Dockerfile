FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Configure Poetry: Don't create virtual env (we're in container)
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set model cache directories
ENV HF_HOME=/app/models

# Create app directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-root

# Create models directory
RUN mkdir -p /app/models
# Copy the rest of the application
COPY . .

# Download HuggingFace models at build time
RUN poetry run python download_models.py
# # Download MLX model
# RUN poetry run huggingface-cli download mlx-community/Qwen2.5-Coder-3B-Instruct-bf16 --cache-dir /app/models
