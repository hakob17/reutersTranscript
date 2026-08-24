# GPU worker image for AWS Batch: WhisperX + pyannote + OpenCV detection.
# Build from the repo root:
#   docker build -f services/gpu_task/Dockerfile -t <ecr>/reuters-transcript-gpu:v1 .
#
# Bake model weights at build time so production has no runtime dependency on
# Hugging Face (pass --build-arg HF_TOKEN=hf_... ; the token is NOT kept in the
# final layer environment).
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt boto3 \
    && python3.11 -m pip install --no-cache-dir --force-reinstall \
        torch torchaudio --index-url https://download.pytorch.org/whl/cu126

# --- bake models (optional but recommended) -------------------------------
ARG HF_TOKEN=""
RUN if [ -n "$HF_TOKEN" ]; then \
      HF_TOKEN=$HF_TOKEN python3.11 - <<'EOF'; fi
import os
from huggingface_hub import snapshot_download
snapshot_download("Systran/faster-whisper-large-v3")
snapshot_download("pyannote/speaker-diarization-community-1",
                  token=os.environ["HF_TOKEN"])
EOF
ENV HF_HUB_OFFLINE=0
# Set HF_HUB_OFFLINE=1 in the job definition once weights are baked.

COPY speaker_attribution/ speaker_attribution/
COPY services/gpu_task/entrypoint.py .

ENTRYPOINT ["python3.11", "entrypoint.py"]
