"""The model directory: what counts as present, and downloads that can be retried."""

from __future__ import annotations

import functools
import http.server
import threading
from pathlib import Path

import pytest

from trundle.models import resolve
from trundle.store import (
    StoreError,
    disk_usage,
    download_file,
    ensure,
    is_present,
    missing_files,
    model_root,
    pull,
)


@pytest.fixture
def served(tmp_path: Path):
    """Serve a directory over HTTP on a random port for the download tests."""
    content = tmp_path / "served"
    content.mkdir()
    (content / "model.onnx").write_bytes(b"x" * 4096)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(content))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", content
    finally:
        server.shutdown()
        server.server_close()


def test_missing_files_and_presence(tmp_path: Path) -> None:
    spec = resolve("multilingual-e5-small")
    root = model_root(tmp_path, spec)
    assert missing_files(root, spec) == list(spec.files)
    assert is_present(root, spec) is False

    root.mkdir(parents=True)
    for name in spec.files:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"data")
    assert is_present(root, spec) is True

    # An empty file is not a present file: that is what an interrupted pull leaves.
    (root / spec.files[0]).write_bytes(b"")
    assert missing_files(root, spec) == [spec.files[0]]


def test_download_writes_the_file_and_cleans_the_partial(served) -> None:
    base, content = served
    dest = Path(content).parent / "out" / "model.onnx"
    download_file(f"{base}/model.onnx", dest)
    assert dest.read_bytes() == (content / "model.onnx").read_bytes()
    assert not (dest.with_name("model.onnx.part")).exists()


def test_download_restarts_when_the_server_ignores_range(served) -> None:
    base, content = served
    dest = Path(content).parent / "out" / "resumed.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.with_name("resumed.onnx.part").write_bytes(b"stale-bytes")
    download_file(f"{base}/model.onnx", dest)
    assert dest.read_bytes() == (content / "model.onnx").read_bytes()


def test_download_reports_http_errors(served) -> None:
    base, content = served
    with pytest.raises(StoreError, match="HTTP 404"):
        download_file(f"{base}/nope.onnx", Path(content).parent / "out" / "nope.onnx")


def test_pull_fills_every_declared_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = resolve("all-MiniLM-L6-v2")
    served_root = tmp_path / "hf"
    for name in spec.files:
        target = served_root / spec.repo / "resolve" / spec.revision / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"onnx" * 128)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served_root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("TRUNDLE_HF_ENDPOINT", f"http://127.0.0.1:{server.server_address[1]}")
    try:
        root = pull(spec, tmp_path / "models")
        assert is_present(root, spec)
        assert disk_usage(tmp_path / "models") == sum(len(b"onnx" * 128) for _ in spec.files)
    finally:
        server.shutdown()
        server.server_close()


def test_unwritable_model_dir_is_explained(tmp_path: Path) -> None:
    spec = resolve("all-MiniLM-L6-v2")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(StoreError, match="not writable"):
            pull(spec, locked)
    finally:
        locked.chmod(0o700)


def test_ensure_refuses_to_download_when_disabled(tmp_path: Path) -> None:
    spec = resolve("all-MiniLM-L6-v2")
    with pytest.raises(StoreError, match="downloading is disabled"):
        ensure(spec, tmp_path, allow_download=False)
