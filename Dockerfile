FROM python:3.11-slim-bookworm
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG ORT_PACKAGE=onnxruntime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install torch==2.8.0 --index-url "$TORCH_INDEX" \
    && pip install numpy==1.26.4 onnx==1.18.0 "$ORT_PACKAGE==1.22.0"
RUN pip install soundfile==0.13.1 tensorboard==2.19.0 museval==0.4.1
WORKDIR /app
COPY model.py run.py dataset.py train.py evaluate.py ./
ENV OUTPUT_DIR=/output MODEL_DIR=/models
ENTRYPOINT ["python", "/app/run.py"]
