FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required directories
RUN mkdir -p \
    data/corpus \
    data/train \
    data/validation \
    data/test \
    models \
    results \
    demo/web \
    demo/sample_docs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TOKENIZERS_PARALLELISM=false

# Expose API port
EXPOSE 8000

# Default command — runs the FastAPI demo app
CMD ["uvicorn", "demo.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]