# syntax=docker/dockerfile:1
# footless + OrcaSAQ-2-27B for the Tesla P100 (sm_60).
# CUDA 12.6 is the newest toolkit that still compiles for sm_60.
ARG CUDA_VERSION=12.6.3

# --- 1. compile the kernels (no GPU needed at build time) ---------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS kernels
WORKDIR /build
COPY model/footless/cuda_sm60/kernels.cu model/footless/cuda_sm60/kernels_exl3.cu ./
# Same flags as the runtime's own build (cuda_sm60/bridge.py, Device.compile).
RUN nvcc -cubin -arch=sm_60 -O3 --use_fast_math -o kernels.cubin kernels.cu \
 && nvcc -cubin -arch=sm_60 -O3 --use_fast_math -o kernels_exl3.cubin kernels_exl3.cu

# --- 2. the server: Python + the driver library from the host -----------------
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HOME=/tmp \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_DISABLE_REQUIRE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
    HF_HUB_DISABLE_PROGRESS_BARS=1
# NVIDIA_DISABLE_REQUIRE: let docker/preflight.py explain an old driver instead
# of the container toolkit refusing to start with a terse "unsatisfied condition".

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
 && pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

WORKDIR /app
# The engine, and the model package that docker/prepare_model.py copies next to
# the downloaded weights (models/OrcaSAQ-2-27B/footless/).
COPY footless/ /app/footless/
COPY model/footless/ /opt/footless-package/footless/
COPY --from=kernels /build/kernels.cubin /build/kernels_exl3.cubin /opt/footless-package/footless/cuda_sm60/
COPY docker/ /app/docker/
RUN chmod -R a+rX /app /opt/footless-package \
 && chmod a+x /app/docker/entrypoint.sh \
 && mkdir -p /app/models && chmod a+rwx /app/models

EXPOSE 8080
ENTRYPOINT ["/app/docker/entrypoint.sh"]
