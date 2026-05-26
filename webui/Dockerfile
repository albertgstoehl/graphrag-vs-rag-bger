# Multi-stage build for kg-rag-control
# Stage 1: Python base + dependencies
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install OS-level deps required by datasets/qdrant_client/networkx
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps: webui + pipeline in one venv
COPY webui/requirements.txt /tmp/webui-req.txt
COPY requirements-poc.txt /tmp/poc-req.txt

RUN pip install --no-cache-dir -r /tmp/webui-req.txt \
 && pip install --no-cache-dir \
      "qdrant-client>=1.13,<2.0" \
      "networkx>=3.2" \
      "requests>=2.31" \
      "tqdm>=4.66" \
      "datasets>=3.0" \
      "huggingface-hub>=0.24"

# Copy pipeline source + webui source
COPY scripts/   /app/scripts/
COPY webui/     /app/webui/

# Writable mount points (will be backed by PVCs in k3s)
RUN mkdir -p /app/data/eval /app/webui/state /root/.cache/huggingface

# Non-root user (optional — for now we stay root so PVC mounts are simple)
EXPOSE 8000

ENV WEBUI_STATE_DIR=/app/webui/state \
    EVAL_DIR=/app/data/eval \
    HF_HOME=/root/.cache/huggingface

CMD ["uvicorn", "webui.app:app", "--host", "0.0.0.0", "--port", "8000"]
