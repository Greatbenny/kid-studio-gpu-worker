FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RUNPOD_VOLUME_PATH=/runpod-volume \
    LOCAL_CACHE_PATH=/workspace/kid-studio-cache \
    LTX_ROOT=/opt/LTX-Video \
    LTX_CONFIG=/opt/LTX-Video/configs/ltxv-13b-0.9.8-distilled.yaml

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Pin the last LTX-Video codebase revision before the repository became an
# LTX-2 redirect notice. This keeps the 0.9.8 13B distilled runtime stable.
RUN git clone https://github.com/Lightricks/LTX-Video.git ${LTX_ROOT} && \
    cd ${LTX_ROOT} && \
    git checkout 5260738e171955b66c0827f9af7e84d68fd1d919 && \
    python3 -m pip install -e '.[inference]'

COPY . .

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "8000"]
