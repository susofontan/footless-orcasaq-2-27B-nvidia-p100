"""Make /app/models/OrcaSAQ-2-27B a complete footless model package.

1. The checkpoint: every file of the pinned Hugging Face revision must be
   present with its exact size. Missing or partial files are downloaded
   (finished files are kept across restarts; a file cut off mid-way starts
   over). A fresh download is then checked against the published SHA-256 of
   each large file, once.
2. The footless/ folder (runtime + kernels, built into the image) is copied
   into the checkpoint directory, replacing any older copy, so the weights and
   the code that runs them always match this image.

Exit status 0 when the package is ready; 1 with a message otherwise.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import threading
import time
from pathlib import Path

REPO_ID = "orcarouter/OrcaSAQ-2-27B"
# The revision this project was measured with. HF_REVISION overrides it; the
# size table below then only applies if that revision has the same files.
PINNED_REVISION = "de9948b02a8c6ea8afb35cfced5b474dff42bdb2"

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models/OrcaSAQ-2-27B"))
PACKAGE_SRC = Path(os.environ.get("PACKAGE_SRC", "/opt/footless-package/footless"))
MARKER = ".footless-verified"
INSTALLED = ".installed-by-container"

# name -> (size in bytes, sha256 or None for small files checked by size only)
FILES = {
    "chat_template.jinja": (8952, None),
    "config.json": (6151, None),
    "generation_config.json": (202, None),
    "merges.txt": (3353259, None),
    "model-00001-of-00004.safetensors":
        (3992179840, "25353d254848dba68557344a6f484a49b21e005669efd41927fe14b2a877d00d"),
    "model-00002-of-00004.safetensors":
        (3986581108, "155c4d6ecda09f1864f5aec92afd49d6c343613a54724bb7cfecb7d818d8f61c"),
    "model-00003-of-00004.safetensors":
        (3125485096, "1c1fc0805f18fc7284f958df99b92bf72307b27bc358f8979d58179aae88b9aa"),
    "model-00004-of-00004.safetensors":
        (1166189360, "36ddf6c9dc0ca408f92a415a9d910a93b1805c2d5ae6da1e1316d3bc9764174f"),
    "model.safetensors.index.json": (197638, None),
    "preprocessor_config.json": (390, None),
    "quantization_config.json": (629308, None),
    "tokenizer.json":
        (12809320, "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"),
    "tokenizer_config.json": (17928, None),
    "video_preprocessor_config.json": (385, None),
    "vocab.json": (6722759, None),
}
TOTAL_BYTES = sum(size for size, _ in FILES.values())


def say(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"\n[prepare] ERROR: {msg}\n", file=sys.stderr, flush=True)
    sys.exit(1)


def missing_files(revision: str) -> list[str]:
    """Files absent or with the wrong size (sizes known for the pinned revision)."""
    out = []
    for name, (size, _) in FILES.items():
        path = MODEL_DIR / name
        if not path.is_file():
            out.append(name)
        elif revision == PINNED_REVISION and path.stat().st_size != size:
            out.append(name)
    return out


PARTIALS = Path(".cache/huggingface/download")


def bytes_on_disk() -> int:
    """Checkpoint bytes written so far: finished files plus this run's partial ones."""
    total = 0
    paths = [MODEL_DIR / name for name in FILES]
    paths += list((MODEL_DIR / PARTIALS).glob("*.incomplete"))
    for path in paths:
        try:
            total += path.stat().st_size
        except OSError:
            pass  # absent, or renamed while we looked
    return total


def drop_stale_partials() -> None:
    """An interrupted file starts over under a new name; the old part is dead weight."""
    for part in (MODEL_DIR / PARTIALS).glob("*.incomplete"):
        part.unlink(missing_ok=True)


def report_progress(done: threading.Event, t0: float) -> None:
    """One log line every 30 s: Docker logs cannot show progress bars."""
    start = bytes_on_disk()
    while not done.wait(30):
        now = min(bytes_on_disk(), TOTAL_BYTES)
        rate = (now - start) / (time.time() - t0) / 1e6
        say(f"downloaded {now / 1e9:.1f} of {TOTAL_BYTES / 1e9:.1f} GB ({rate:.0f} MB/s)")


def download(revision: str) -> None:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError as exc:  # the image always has it; a bare checkout may not
        fail(f"huggingface_hub is not installed: {exc}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    drop_stale_partials()
    free = shutil.disk_usage(MODEL_DIR).free
    have = sum((MODEL_DIR / n).stat().st_size for n in FILES if (MODEL_DIR / n).is_file())
    need = TOTAL_BYTES - have + (512 << 20)
    if free < need:
        fail(f"not enough disk space in the models folder: {free / 1e9:.1f} GB free, "
             f"{need / 1e9:.1f} GB needed. Free some space or point MODELS_DIR in .env "
             "to a larger disk.")

    token = os.environ.get("HF_TOKEN") or None
    say(f"downloading {REPO_ID}@{revision[:12]} into {MODEL_DIR} "
        f"(~{TOTAL_BYTES / 1e9:.1f} GB, {'with' if token else 'without'} HF_TOKEN)")
    t0 = time.time()
    done = threading.Event()
    threading.Thread(target=report_progress, args=(done, t0), daemon=True).start()
    try:
        snapshot_download(
            repo_id=REPO_ID,
            revision=revision,
            local_dir=str(MODEL_DIR),
            allow_patterns=list(FILES),
            token=token,
        )
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        fail(f"Hugging Face refused the download ({exc.__class__.__name__}). "
             "Check HF_TOKEN in .env, or that the repository is reachable.")
    except Exception as exc:  # network, disk, auth: report and stop
        fail(f"download failed: {exc}\nThe next start downloads what is still missing "
             "(finished files are kept).")
    finally:
        done.set()
    say(f"download finished in {(time.time() - t0) / 60:.1f} min")


def verify_hashes() -> None:
    for name, (size, sha) in FILES.items():
        if sha is None:
            continue
        say(f"checking SHA-256 of {name}")
        digest = hashlib.sha256()
        with open(MODEL_DIR / name, "rb") as fh:
            while chunk := fh.read(16 << 20):
                digest.update(chunk)
        if digest.hexdigest() != sha:
            (MODEL_DIR / name).unlink()
            fail(f"{name} is corrupt (SHA-256 mismatch); it was deleted. "
                 "Restart the container to download it again.")


def install_package() -> None:
    """Replace MODEL_DIR/footless with the image's copy, cubins kept newest."""
    if not PACKAGE_SRC.is_dir():
        fail(f"the image has no footless package at {PACKAGE_SRC}")
    target = MODEL_DIR / "footless"
    tmp = MODEL_DIR / ".footless.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(PACKAGE_SRC, tmp, copy_function=shutil.copy2)
    # The runtime recompiles a kernel source whose .cubin is older than it,
    # and this image has no compiler: make the prebuilt cubins the newest files.
    now = time.time() + 1
    for cubin in tmp.rglob("*.cubin"):
        os.utime(cubin, (now, now))
    (tmp / INSTALLED).write_text("installed by the container; replaced on every start\n")
    if target.is_dir() and not (target / INSTALLED).is_file():
        # someone else's footless/ (a development copy): keep it, never delete it
        keep = MODEL_DIR / f"footless.backup-{time.strftime('%Y%m%d-%H%M%S')}"
        target.rename(keep)
        say(f"an existing footless/ folder not installed by this container was moved to {keep}")
    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    say(f"installed the footless runtime and kernels into {target}")


def main() -> int:
    revision = os.environ.get("HF_REVISION") or PINNED_REVISION
    if revision != PINNED_REVISION:
        say(f"HF_REVISION={revision} differs from the tested {PINNED_REVISION[:12]}; "
            "file sizes and hashes are not checked")

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        probe = MODEL_DIR / ".write-test"
        probe.touch()
        probe.unlink()
    except OSError as exc:
        fail(f"cannot write to {MODEL_DIR} ({exc}). The models folder on the host must be "
             f"writable by UID {os.getuid()} (set PUID/PGID in .env, or chown the folder).")

    marker = MODEL_DIR / MARKER
    missing = missing_files(revision)
    if missing:
        say(f"{len(missing)} of {len(FILES)} checkpoint files missing or incomplete")
        marker.unlink(missing_ok=True)
        download(revision)
        missing = missing_files(revision)
        if missing:
            fail(f"after the download these files are still missing or incomplete: {missing}")
    else:
        say(f"checkpoint found in {MODEL_DIR}, no download needed")

    if revision == PINNED_REVISION and not marker.is_file():
        verify_hashes()
        marker.write_text(revision + "\n")

    install_package()
    return 0


if __name__ == "__main__":
    sys.exit(main())
