FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RUNPOD_VOLUME_PATH=/runpod-volume \
    LOCAL_CACHE_PATH=/workspace/kid-studio-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "8000"]
