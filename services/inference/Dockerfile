# STAGE 1: Builder - Install build-time dependencies
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS builder

WORKDIR /app

# Install system dependencies needed for compiling some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install requirements to the user site-packages
RUN pip install --user --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --user --no-cache-dir -r requirements.txt

# STAGE 2: Runner - Lightweight production image
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install only runtime system dependencies (OpenCV, FFmpeg, TTS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Copy the installed python packages from the builder stage
COPY --from=builder /root/.local /root/.local
# Ensure the scripts from .local/bin are in the PATH
ENV PATH=/root/.local/bin:$PATH
# Prevent python from writing pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy the application source code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
