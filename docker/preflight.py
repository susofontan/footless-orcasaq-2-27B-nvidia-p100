"""Check, inside the container, that this GPU can run the model. Before anything else.

Fails (exit 1) with the reason and what to do when:
  * no NVIDIA driver library reached the container (toolkit / --gpus missing),
  * the driver is too old for the prebuilt kernels (needs CUDA 12.6, driver >= 560),
  * no GPU, or the GPU is not a 16 GB Tesla P100 (compute capability 6.0),
  * too little GPU memory is free (another process is using the card),
  * the models folder lacks room for the checkpoint (only when it must download).

Warnings (the server still starts): several GPUs visible, a hot or throttled card.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

MIN_DRIVER_CUDA = 12060          # cuDriverGetVersion() of a CUDA 12.6 driver (>= 560)
NEED_CC = (6, 0)                 # Tesla P100 (GP100), the only target of the kernels
MIN_TOTAL_GIB = 15.5             # the 16 GB P100; the 12 GB model cannot hold the weights
MIN_FREE_GIB = 14.0              # weights (~11.5 GiB) + MTP + a minimal cache
WARN_FREE_GIB = 15.3             # below this the maximum context shrinks
CHECKPOINT_GB = 12.4

errors: list[str] = []
warnings: list[str] = []


def line(status: str, text: str) -> None:
    print(f"[preflight] {status:5s} {text}", flush=True)


def error(text: str) -> None:
    errors.append(text)
    line("FAIL", text)


def warn(text: str) -> None:
    warnings.append(text)
    line("WARN", text)


def ok(text: str) -> None:
    line("OK", text)


def load_driver():
    try:
        return ctypes.CDLL("libcuda.so.1")
    except OSError:
        error("the NVIDIA driver library (libcuda.so.1) is not visible inside the container. "
              "Install the NVIDIA Container Toolkit on the host, restart Docker, and start "
              "the service with `docker compose up` (it requests the GPU).")
        return None


def check_driver(cuda) -> bool:
    rc = cuda.cuInit(0)
    if rc != 0:
        error(f"cuInit failed with CUDA error {rc}. The driver cannot see a usable GPU "
              "(is the card in use by a VM passthrough, or the driver not loaded?).")
        return False
    ver = ctypes.c_int()
    cuda.cuDriverGetVersion(ctypes.byref(ver))
    major, minor = ver.value // 1000, (ver.value % 1000) // 10
    smi = driver_version_from_smi()
    shown = f"driver {smi}, " if smi else ""
    if ver.value < MIN_DRIVER_CUDA:
        error(f"{shown}CUDA {major}.{minor}: the kernels need a driver for CUDA 12.6 or newer "
              "(560 or later; the 580 branch is the last one that supports the P100).")
        return False
    ok(f"{shown}supports CUDA {major}.{minor}")
    return True


def driver_version_from_smi() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return None


def check_device(cuda) -> None:
    count = ctypes.c_int()
    cuda.cuDeviceGetCount(ctypes.byref(count))
    if count.value == 0:
        error("no GPU visible in the container. Check GPU_DEVICE in .env (see `nvidia-smi -L` "
              "on the host).")
        return
    if count.value > 1:
        warn(f"{count.value} GPUs visible; the server uses the first one. Set GPU_DEVICE in "
             ".env to the P100's index or UUID so only it is passed in.")

    dev = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(dev), 0)
    name = ctypes.create_string_buffer(256)
    cuda.cuDeviceGetName(name, 256, dev)
    name = name.value.decode(errors="replace")
    cc = []
    for attr in (75, 76):  # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR
        out = ctypes.c_int()
        cuda.cuDeviceGetAttribute(ctypes.byref(out), attr, dev)
        cc.append(out.value)
    total = ctypes.c_size_t()
    cuda.cuDeviceTotalMem_v2(ctypes.byref(total), dev)
    total_gib = total.value / 2**30

    if tuple(cc) != NEED_CC:
        error(f"GPU 0 is {name} (compute capability {cc[0]}.{cc[1]}). These kernels are built "
              "for the Tesla P100 only (compute capability 6.0). If the host has a P100 too, "
              "set GPU_DEVICE in .env to it.")
        return
    if total_gib < MIN_TOTAL_GIB:
        error(f"{name} has {total_gib:.1f} GiB; the model needs the 16 GB P100 "
              "(the 12 GB version cannot hold the weights).")
        return
    ok(f"{name}, compute capability {cc[0]}.{cc[1]}, {total_gib:.1f} GiB")

    ctx = ctypes.c_void_p()
    rc = cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    if rc != 0:
        error(f"cannot open a context on {name} (CUDA error {rc}); is it in exclusive mode "
              "and used by another process?")
        return
    free, tot = ctypes.c_size_t(), ctypes.c_size_t()
    cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(tot))
    cuda.cuCtxDestroy_v2(ctx)
    free_gib = free.value / 2**30
    if free_gib < MIN_FREE_GIB:
        error(f"only {free_gib:.1f} GiB of GPU memory free; the model needs {MIN_FREE_GIB} GiB. "
              "Stop whatever else uses the card (a desktop session, another container: "
              "`nvidia-smi` on the host lists them).")
    elif free_gib < WARN_FREE_GIB:
        warn(f"{free_gib:.1f} GiB of GPU memory free: it will run, with a shorter maximum "
             "context. A card that only runs this server has ~15.6 GiB free.")
    else:
        ok(f"{free_gib:.1f} GiB of GPU memory free")


def check_thermals() -> None:
    if not shutil.which("nvidia-smi"):
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", "0", "--query-gpu=temperature.gpu,clocks.sm,clocks.max.sm",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        temp, sm, sm_max = (int(x) for x in out.stdout.strip().split(","))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return
    if temp >= 75:
        warn(f"the GPU is at {temp} C while idle: with poor cooling it will throttle under "
             "load and lose ~10% or more of its speed (see README, Performance).")
    else:
        ok(f"GPU temperature {temp} C (SM clock {sm} of {sm_max} MHz)")


def check_disk(model_dir: Path) -> None:
    if (model_dir / "model-00004-of-00004.safetensors").is_file():
        return  # the checkpoint is (mostly) there; prepare_model.py checks the rest
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(model_dir.parent).free / 1e9
    if free_gb < CHECKPOINT_GB + 0.5:
        error(f"the models folder has {free_gb:.1f} GB free and the checkpoint needs "
              f"~{CHECKPOINT_GB} GB. Point MODELS_DIR in .env to a larger disk.")
    else:
        ok(f"{free_gb:.0f} GB free in the models folder for the {CHECKPOINT_GB} GB download")


def main() -> int:
    print("[preflight] checking the GPU and the driver", flush=True)
    cuda = load_driver()
    if cuda is not None and check_driver(cuda):
        check_device(cuda)
        if not errors:
            check_thermals()
    check_disk(Path(os.environ.get("MODEL_DIR", "/app/models/OrcaSAQ-2-27B")))
    if errors:
        what = "" if "--check-only" in sys.argv else "; the server was not started"
        print(f"\n[preflight] {len(errors)} requirement(s) not met{what}. "
              "See README.md, Prerequisites.\n", file=sys.stderr, flush=True)
        return 1
    print(f"[preflight] all requirements met ({len(warnings)} warning(s))", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
