# STAGE 1: Builder - Install build-time dependencies
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS builder

WORKDIR /app

# Install system dependencies needed for compiling some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install requirements to the user site-packages with cache mounting
RUN pip install --user --no-cache-dir --upgrade pip setuptools wheel
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements.txt

# STAGE 2: Runner - Lightweight production image
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install only runtime system dependencies (FFmpeg, TTS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and setup directories with correct permissions BEFORE copying code
RUN useradd -m -u 1000 matcha && \
    mkdir -p models uploads && \
    chown -R matcha:matcha /app

# Copy the installed python packages from the builder stage
COPY --from=builder /root/.local /home/matcha/.local

# Ensure the scripts from .local/bin are in the PATH
ENV PATH=/home/matcha/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application code with correct ownership immediately to avoid heavy chown layers
COPY --chown=matcha:matcha app/ app/
COPY --chown=matcha:matcha main.py .

USER matcha
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
