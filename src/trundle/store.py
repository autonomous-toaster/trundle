"""Model storage: a plain directory per model, filled over HTTPS.

Deliberately no huggingface_hub dependency: this needs to work inside a
distroless image with a single volume mount and no cache semantics. The layout
is `$TRUNDLE_MODEL_DIR/<model name>/<file>` so it can be inspected, copied and
backed up with `ls` and `cp`.

Downloads are resumable (HTTP Range) and atomic (`.part` file, then rename), so
an interrupted pull never leaves a model that looks present but is truncated.
"""

from __future__ import annotations

import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .models import ModelSpec

DEFAULT_HF_ENDPOINT = "https://huggingface.co"
CHUNK = 1 << 20

Progress = Callable[[str, int, int], None]


class StoreError(RuntimeError):
    """Anything that stops a model from being usable on disk."""


def model_root(model_dir: Path, spec: ModelSpec) -> Path:
    return Path(model_dir) / spec.name


def missing_files(root: Path, spec: ModelSpec) -> list[str]:
    return [name for name in spec.files if not (root / name).is_file() or (root / name).stat().st_size == 0]


def is_present(root: Path, spec: ModelSpec) -> bool:
    return not missing_files(root, spec)


def hf_endpoint() -> str:
    """Read the endpoint per call: mirrors are configured after import in tests and CLIs."""
    return os.environ.get("TRUNDLE_HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/")


def file_url(spec: ModelSpec, name: str) -> str:
    return f"{hf_endpoint()}/{spec.repo}/resolve/{spec.revision}/{name}"


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".trundle-write-test"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def download_file(
    url: str,
    dest: Path,
    *,
    token: str | None = None,
    progress: Progress | None = None,
) -> None:
    """Fetch `url` into `dest`, resuming a previous `.part` if there is one."""
    partial = dest.with_name(dest.name + ".part")
    have = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "trundle-embed"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if have:
        headers["Range"] = f"bytes={have}-"

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        if error.code == 416 and have:
            # The partial file is already complete or larger; start over.
            partial.unlink(missing_ok=True)
            return download_file(url, dest, token=token, progress=progress)
        if error.code in (401, 403):
            raise StoreError(
                f"{url} -> HTTP {error.code}. The repository is gated or private: set HF_TOKEN."
            ) from None
        raise StoreError(f"{url} -> HTTP {error.code} {error.reason}") from None
    except urllib.error.URLError as error:
        raise StoreError(f"{url} -> {error.reason}. Offline? Set TRUNDLE_OFFLINE=1 to skip downloads.") from None

    resumed = response.status == 206
    received = have if resumed else 0
    declared = response.headers.get("Content-Length")
    total = received + int(declared) if declared is not None else 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(partial, "ab" if resumed else "wb") as handle:
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            handle.write(block)
            received += len(block)
            if progress:
                progress(dest.name, received, total)
    response.close()

    if total and received != total:
        raise StoreError(f"{dest.name}: got {received} of {total} bytes; retry the download")
    os.replace(partial, dest)


def pull(
    spec: ModelSpec,
    model_dir: Path,
    *,
    progress: Progress | None = None,
    force: bool = False,
) -> Path:
    """Download every file of `spec` into the model directory."""
    root = model_root(model_dir, spec)
    if not _writable(Path(model_dir)):
        raise StoreError(
            f"{model_dir} is not writable. A named volume inherits the image ownership, "
            "a bind mount does not: `chown -R 65532:65532 <host dir>` or pre-pull the model."
        )
    todo = spec.files if force else tuple(missing_files(root, spec))
    if not todo:
        return root
    token = _token()
    for name in todo:
        download_file(file_url(spec, name), root / name, token=token, progress=progress)
    return root


def ensure(
    spec: ModelSpec,
    model_dir: Path,
    *,
    allow_download: bool,
    progress: Progress | None = None,
) -> Path:
    """Make sure the model is on disk; download it if allowed, else explain."""
    root = model_root(model_dir, spec)
    absent = missing_files(root, spec)
    if not absent:
        return root
    if not allow_download:
        raise StoreError(
            f"model {spec.name!r} is incomplete in {root} (missing: {', '.join(absent)}) "
            f"and downloading is disabled. Run `trundle pull --model {spec.name}` "
            "with TRUNDLE_OFFLINE unset."
        )
    return pull(spec, model_dir, progress=progress)


def disk_usage(model_dir: Path) -> int:
    """Bytes used by everything under the model directory."""
    total = 0
    root = Path(model_dir)
    if not root.is_dir():
        return 0
    for entry in root.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def remove(spec: ModelSpec, model_dir: Path) -> None:
    shutil.rmtree(model_root(model_dir, spec), ignore_errors=True)
