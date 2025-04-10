# Use Python 3.11 slim base image
FROM python:3.11.11-slim
# Set working directory
WORKDIR /app
# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libstdc++6 \
    git \
    cmake \
    build-essential \
    libx11-dev \
    libatlas-base-dev \
    python3-pip \
    protobuf-compiler \
    libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*
# Copy requirements first for better caching
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt

COPY requirements.txt .
# Install Python dependencies

RUN pip install --no-cache-dir -r requirements.txt
# Copy models first
COPY . .
# Environment variables
ENV PYTHONPATH=/app

# Expose port
EXPOSE 50051
# Command to run the server
CMD ["python", "server.py"]
