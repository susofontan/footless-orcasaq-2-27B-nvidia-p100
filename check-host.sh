#!/usr/bin/env bash
# Check this host can run the container, and find the Tesla P100 for .env.
#   ./check-host.sh          every check, including a GPU test inside a container
#   ./check-host.sh --quick  skip the container test (no image pull)
# Exit status: 0 when nothing failed.
set -uo pipefail
cd "$(dirname "$0")"

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1
CUDA_IMAGE="nvidia/cuda:12.6.3-base-ubuntu24.04"   # the base of the server image
MIN_DRIVER=560     # CUDA 12.6
MAX_DRIVER=580     # the last driver branch that supports Pascal (P100)
CHECKPOINT_GB=13

if [[ -t 1 ]]; then G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; B=$'\e[1m'; N=$'\e[0m'; else G= Y= R= B= N=; fi
FAILS=0; WARNS=0
ok()   { echo "  ${G}OK${N}    $*"; }
warn() { echo "  ${Y}WARN${N}  $*"; WARNS=$((WARNS + 1)); }
fail() { echo "  ${R}FAIL${N}  $*"; FAILS=$((FAILS + 1)); }
section() { echo; echo "${B}$*${N}"; }

# KEY=value from .env without executing it; default when absent or empty.
env_get() {
  local v=""
  [[ -f .env ]] && v=$(grep -E "^[[:space:]]*$1=" .env | tail -n1 | cut -d= -f2- | sed -e 's/[[:space:]]*#.*$//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/" -e 's/[[:space:]]*$//')
  echo "${v:-$2}"
}

section "Settings"
if [[ ! -f .env ]]; then
  # the primary group from passwd: `id -g` is the effective one, which
  # `newgrp docker` / `sg docker` change
  MY_GID=$(getent passwd "$(id -u)" | cut -d: -f4); MY_GID=${MY_GID:-$(id -g)}
  sed -e "s/^PUID=.*/PUID=$(id -u)/" -e "s/^PGID=.*/PGID=$MY_GID/" .env.example > .env
  ok "created .env from .env.example, PUID/PGID set to your user (edit it to change GPU, port, token)"
else
  ok ".env found"
fi
GPU_DEVICE=$(env_get GPU_DEVICE 0)
PORT=$(env_get PORT 8080)
MODELS_DIR=$(env_get MODELS_DIR ./models)
PUID=$(env_get PUID 1000)

section "Operating system"
if [[ "$(uname -s)" == "Linux" ]]; then ok "Linux $(uname -r)"; else fail "$(uname -s): the NVIDIA container stack needs a Linux host"; fi

section "Docker"
if ! command -v docker >/dev/null; then
  fail "docker is not installed: https://docs.docker.com/engine/install/"
else
  if out=$(docker info --format '{{.ServerVersion}}' 2>&1); then
    ok "Docker Engine $out"
  elif grep -qi "permission denied" <<<"$out"; then
    fail "your user cannot reach the Docker daemon: run  sudo usermod -aG docker \$USER  and log in again (or run the docker commands with sudo)"
  else
    fail "the Docker daemon is not running: sudo systemctl start docker"
  fi
  if v=$(docker compose version --short 2>/dev/null); then ok "Docker Compose $v"; else fail "the Docker Compose plugin is missing: https://docs.docker.com/compose/install/linux/"; fi
fi

section "NVIDIA driver"
if ! command -v nvidia-smi >/dev/null; then
  fail "nvidia-smi not found: install the NVIDIA driver (560 to 580 series), then reboot"
  DRIVER=""
else
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
  if [[ -z "$DRIVER" ]]; then
    fail "nvidia-smi cannot talk to the driver (is the module loaded? reboot after installing)"
  else
    major=${DRIVER%%.*}
    if (( major < MIN_DRIVER )); then
      fail "driver $DRIVER is too old: the kernels need CUDA 12.6, driver $MIN_DRIVER or newer (up to the $MAX_DRIVER series)"
    elif (( major > MAX_DRIVER )); then
      fail "driver $DRIVER: drivers after the $MAX_DRIVER series do not support the P100 (Pascal); install a $MAX_DRIVER-series driver"
    else
      ok "driver $DRIVER"
    fi
  fi
fi

section "NVIDIA Container Toolkit"
if command -v nvidia-ctk >/dev/null || command -v nvidia-container-runtime-hook >/dev/null; then
  ok "installed ($(nvidia-ctk --version 2>/dev/null | head -n1 || echo nvidia-container-runtime-hook))"
else
  fail "not installed: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html  (then: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker)"
fi

section "GPUs"
P100=()
SELECTED=""
if [[ -n "$DRIVER" ]]; then
  printf "  %-5s %-28s %-42s %9s %6s\n" index name uuid memory cc
  while IFS=, read -r idx name uuid mem cc used; do
    idx=${idx// /}; uuid=${uuid// /}; mem=${mem// /}; cc=${cc// /}; used=${used// /}; name=${name# }
    mark=""
    if [[ "$cc" == "6.0" && "$mem" -ge 15000 ]]; then P100+=("$idx|$uuid|$used"); mark="  <- P100 16 GB"; fi
    if [[ "$GPU_DEVICE" == "$idx" || "$GPU_DEVICE" == "$uuid" ]]; then SELECTED="$idx|$uuid|$cc|$mem|$used|$name"; mark="$mark  (GPU_DEVICE)"; fi
    printf "  %-5s %-28s %-42s %6s MiB %6s%s\n" "$idx" "$name" "$uuid" "$mem" "$cc" "$mark"
  done < <(nvidia-smi --query-gpu=index,name,uuid,memory.total,compute_cap,memory.used --format=csv,noheader,nounits 2>/dev/null)

  if (( ${#P100[@]} == 0 )); then
    fail "no 16 GB Tesla P100 (compute capability 6.0) found; this image runs only on that card"
  elif [[ -z "$SELECTED" ]]; then
    p=${P100[0]#*|}; fail "GPU_DEVICE=$GPU_DEVICE matches no GPU. Set GPU_DEVICE=${p%%|*} in .env"
  else
    IFS='|' read -r s_idx s_uuid s_cc s_mem s_used s_name <<<"$SELECTED"
    if [[ "$s_cc" != "6.0" || "$s_mem" -lt 15000 ]]; then
      p=${P100[0]#*|}; fail "GPU_DEVICE=$GPU_DEVICE is $s_name, not a 16 GB P100. Set GPU_DEVICE=${p%%|*} in .env"
    else
      ok "GPU_DEVICE=$GPU_DEVICE is the P100 ($s_uuid)"
      if (( s_used > 500 )); then
        warn "$s_used MiB of the P100 already in use (by this server, if it is running). The model needs ~15.8 GB of its 16: stop any other user of the card (nvidia-smi lists them)"
      fi
      gpus=$(nvidia-smi -L | wc -l)
      if (( gpus > 1 )) && [[ "$GPU_DEVICE" =~ ^[0-9]+$ ]]; then
        warn "several GPUs: indexes can change between boots, the UUID cannot. Prefer GPU_DEVICE=$s_uuid"
      fi
    fi
  fi
fi

section "Model folder and port"
mkdir -p "$MODELS_DIR" 2>/dev/null
if [[ ! -d "$MODELS_DIR" ]]; then
  fail "MODELS_DIR=$MODELS_DIR does not exist and cannot be created"
else
  owner=$(stat -c %u "$MODELS_DIR")
  if [[ "$owner" != "$PUID" && "$PUID" != "0" ]]; then
    warn "$MODELS_DIR belongs to UID $owner but the container runs as PUID=$PUID; set PUID/PGID in .env to $(id -u)/$(getent passwd "$(id -u)" | cut -d: -f4) or chown the folder"
  fi
  if [[ -f "$MODELS_DIR/OrcaSAQ-2-27B/model-00004-of-00004.safetensors" ]]; then
    ok "checkpoint present in $MODELS_DIR/OrcaSAQ-2-27B (checked again at start)"
  else
    free=$(df -BG --output=avail "$MODELS_DIR" | tail -n1 | tr -dc 0-9)
    if (( free < CHECKPOINT_GB )); then
      fail "$MODELS_DIR has ${free} GB free; the checkpoint needs ~12.3 GB"
    else
      ok "${free} GB free in $MODELS_DIR for the first-start download (~12.3 GB)"
    fi
  fi
fi
if command -v ss >/dev/null && ss -ltnH "sport = :$PORT" 2>/dev/null | grep -q .; then
  warn "port $PORT is already in use on this host (by a running copy of this server?); change PORT in .env if not"
else
  ok "port $PORT is free"
fi

if (( QUICK == 0 && FAILS == 0 )); then
  section "GPU inside a container"
  echo "  running $CUDA_IMAGE with GPU_DEVICE=$GPU_DEVICE (first time: pulls ~100 MB)"
  if out=$(docker run --rm --gpus "\"device=$GPU_DEVICE\"" "$CUDA_IMAGE" nvidia-smi -L 2>&1); then
    ok "the container sees: $(grep -m1 GPU <<<"$out")"
  else
    fail "a container could not use the GPU:"
    sed 's/^/          /' <<<"$out" | tail -n 5
  fi
fi

echo
if (( FAILS > 0 )); then
  echo "${R}${B}$FAILS check(s) failed${N}, $WARNS warning(s). Fix the failures above, then run this again."
  exit 1
fi
echo "${G}${B}Ready${N} ($WARNS warning(s)). Start with:  docker compose up -d --build   and follow with:  docker compose logs -f"
