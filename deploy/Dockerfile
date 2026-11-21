# GPU server image. The CUDA runtime comes with the PyTorch wheels ; the host needs an NVIDIA driver
# and the NVIDIA Container Toolkit (docker run --gpus all).
#   docker build -f deploy/Dockerfile -t realtime-translation .
#   docker run --gpus all -p 127.0.0.1:8000:8000 -e WLK_API_TOKEN=... -v rt-models:/models realtime-translation
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl && rm -rf /var/lib/apt/lists/*
ENV HF_HOME=/models PYTHONUNBUFFERED=1 HOST=0.0.0.0

WORKDIR /app
COPY pyproject.toml README.md ./
COPY realtime_translation ./realtime_translation
COPY deploy/server.sh ./deploy/server.sh
RUN pip install --no-cache-dir uv \
 && uv venv --python 3.12 .venv \
 && uv pip install --python .venv/bin/python --no-cache -e . "whisperlivekit[translation,voxtral-hf]>=0.2.26"

VOLUME /models
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["./deploy/server.sh"]
