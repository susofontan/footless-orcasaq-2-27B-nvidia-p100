"""Reading and verifying footless/manifest — the truth before load (rule 3)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .sdk import ContractError, Facts

# Directories a package may carry beside its backends. A package keeps its own
# tests in footless/tests/ (AGENTS.md), which is not a backend; neither is a
# private (`_build`, `_reference`) or tooling (`.pytest_cache`) directory.
_RESERVED_DIRS = frozenset({"tests"})
_IGNORED_PREFIXES = ("_", ".")

_KEY_TYPES = {
    "name": str,
    "backends": list,
    "max_batch": int,
    "max_context": int,
    "cache_bytes_per_token": int,
    "capabilities": list,
}

# keys a package may carry; a package without them offers no such thing.
# `runtime` is the package's own settings table: its runtime reads it after
# load, and the engine checks only that it is a table.
_OPTIONAL_TYPES = {
    "thinking_levels": list,
    "runtime": dict,
}


@dataclass(frozen=True)
class Manifest:
    name: str
    backends: list[str]
    max_batch: int
    max_context: int
    cache_bytes_per_token: int
    capabilities: list[str]
    thinking_levels: list[str] = field(default_factory=list)


def read(footless_dir: Path) -> Manifest:
    path = footless_dir / "manifest"
    try:
        raw = tomllib.loads(path.read_text())
    except FileNotFoundError:
        raise ContractError(f"missing manifest: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ContractError(f"unreadable manifest {path}: {exc}") from None

    missing = [key for key in _KEY_TYPES if key not in raw]
    if missing:
        raise ContractError(f"manifest is missing keys: {missing}")
    unknown = [key for key in raw if key not in _KEY_TYPES and key not in _OPTIONAL_TYPES]
    if unknown:
        raise ContractError(f"manifest has unknown keys: {unknown}")
    for key, typ in (_KEY_TYPES | _OPTIONAL_TYPES).items():
        if key not in raw:
            continue
        value = raw[key]
        if not isinstance(value, typ) or (typ is int and isinstance(value, bool)):
            raise ContractError(f"manifest {key} must be {typ.__name__}")
    if not all(isinstance(item, str) for item in raw["backends"]):
        raise ContractError("manifest backends must be strings")
    if not all(isinstance(item, str) for item in raw["capabilities"]):
        raise ContractError("manifest capabilities must be strings")
    if not all(isinstance(item, str) for item in raw.get("thinking_levels", [])):
        raise ContractError("manifest thinking_levels must be strings")

    return Manifest(
        name=raw["name"],
        backends=list(raw["backends"]),
        max_batch=raw["max_batch"],
        max_context=raw["max_context"],
        cache_bytes_per_token=raw["cache_bytes_per_token"],
        capabilities=list(raw["capabilities"]),
        thinking_levels=list(raw.get("thinking_levels", [])),
    )


def verify_backends(manifest: Manifest, footless_dir: Path) -> None:
    """The manifest declares which backends exist; the disk must match."""
    present = sorted(
        p.name
        for p in footless_dir.iterdir()
        if p.is_dir()
        and not p.name.startswith(_IGNORED_PREFIXES)
        and p.name not in _RESERVED_DIRS
    )
    declared = sorted(manifest.backends)
    if present != declared:
        raise ContractError(
            f"manifest backends {declared} do not match the disk {present}"
        )
    for backend in declared:
        if not (footless_dir / backend / "runtime.py").is_file():
            raise ContractError(f"backend {backend!r} has no runtime.py")


def verify(manifest: Manifest, facts: Facts) -> None:
    """Contradictions between the manifest and the running code are hard errors."""
    for field in ("cache_bytes_per_token", "max_batch", "max_context"):
        declared = getattr(manifest, field)
        reported = getattr(facts, field)
        if declared != reported:
            raise ContractError(
                f"manifest {field} = {declared}, runtime reports {reported}"
            )
    missing = [c for c in manifest.capabilities if c not in facts.capabilities]
    if missing:
        raise ContractError(
            f"manifest declares capabilities the runtime does not offer: {missing}"
        )
    declared, reported = set(manifest.thinking_levels), set(facts.thinking_levels)
    if declared != reported:
        raise ContractError(
            f"manifest thinking_levels = {manifest.thinking_levels}, runtime "
            f"reports {facts.thinking_levels}"
        )
