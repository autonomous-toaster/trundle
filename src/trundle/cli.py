"""Command line: serve, pull, list, info.

`trundle` with no arguments prints the help, because a container started without
a command should do something legible rather than crash.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, Settings
from .models import DEFAULT_MODEL, REGISTRY, ModelSpec, UnknownModel, resolve
from .store import StoreError, disk_usage, is_present, missing_files, model_root, pull

LOG = logging.getLogger("trundle")
_PROGRESS: dict[str, int] = {}


def _settings(args: argparse.Namespace) -> Settings:
    base = Settings.from_env()
    return base.with_overrides(
        model=getattr(args, "model", None),
        model_dir=getattr(args, "model_dir", None),
        host=getattr(args, "host", None),
        port=getattr(args, "port", None),
        workers=getattr(args, "workers", None),
        threads=getattr(args, "threads", None),
        max_batch=getattr(args, "max_batch", None),
        max_inputs=getattr(args, "max_inputs", None),
        max_length=getattr(args, "max_length", None),
        prefix_mode=getattr(args, "prefix_mode", None),
        log_level=getattr(args, "log_level", None),
    )


def _progress(name: str, received: int, total: int) -> None:
    """Redraw at most every 5 %: a line per megabyte floods a captured log."""
    percent = int(100 * received / total) if total else 0
    done = bool(total) and received >= total
    if not done and percent - _PROGRESS.get(name, -1) < 5:
        return
    _PROGRESS[name] = percent
    if total:
        sys.stderr.write(f"\r  {name}: {received / 1e6:7.1f} / {total / 1e6:.1f} MB ({percent:3d}%)")
    else:
        sys.stderr.write(f"\r  {name}: {received / 1e6:7.1f} MB")
    if done:
        sys.stderr.write("\n")
    sys.stderr.flush()


def cmd_serve(args: argparse.Namespace) -> int:
    from .api import create_app
    from .embedder import Embedder, EmbedderError

    settings = _settings(args)
    spec = resolve(settings.model)
    logging.getLogger().setLevel(settings.log_level.upper())

    LOG.info("model dir: %s", settings.model_dir)
    try:
        root = _ensure_model(spec, settings, allow_download=not args.no_download)
        embedder = Embedder(
            spec,
            settings.model_dir,
            threads=settings.threads,
            max_batch=settings.max_batch,
            max_length=settings.max_length,
        )
    except (StoreError, EmbedderError) as error:
        LOG.error("%s", error)
        return 1

    LOG.info("loaded %s from %s (%d dim)", spec.name, root, embedder.dim)
    if settings.prefix_mode == "document" and spec.document_prefix:
        LOG.info("prefixing every input with %r; set TRUNDLE_PREFIX_MODE to change", spec.document_prefix)
    if settings.api_key:
        LOG.info("bearer authentication is on")

    import uvicorn

    app = create_app(embedder, settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        # One process: a second copy of the model costs memory, not throughput.
        workers=1,
        log_level="debug" if settings.log_level == "debug" else "info",
        access_log=settings.log_level == "debug",
    )
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    settings = _settings(args)
    names = sorted(REGISTRY) if args.all else [settings.model]
    for name in names:
        spec = resolve(name)
        root = model_root(settings.model_dir, spec)
        missing = missing_files(root, spec)
        if not missing and not args.force:
            LOG.info("%s: already present in %s", name, root)
            continue
        LOG.info("%s: fetching %d file(s) into %s", name, len(missing), root)
        try:
            pull(spec, settings.model_dir, progress=_progress, force=args.force)
        except StoreError as error:
            LOG.error("%s: %s", name, error)
            return 1
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    settings = _settings(args)
    used = disk_usage(settings.model_dir)
    print(f"model directory: {settings.model_dir} ({used / 1e6:.1f} MB used)")
    header = f"{'name':28} {'dim':>4} {'MiB':>5} {'ms/chunk':>9}  state"
    print(header)
    print("-" * len(header))
    for name, spec in sorted(REGISTRY.items()):
        present = is_present(model_root(settings.model_dir, spec), spec)
        state = "present" if present else ("partial" if (model_root(settings.model_dir, spec)).exists() else "absent")
        speed = f"{spec.ms_per_chunk:9.0f}" if spec.ms_per_chunk else f"{'-':>9}"
        print(f"{name:28} {spec.dim:4} {spec.mib:5.0f} {speed}  {state}")
    print("\nms/chunk measured with 128-token chunks, 1 worker, 1 thread, on a 2-core CPU without AVX.")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    import numpy

    settings = _settings(args)
    spec = resolve(settings.model)
    print(f"trundle        {__version__}")
    print(f"python         {sys.version.split()[0]}")
    print(f"numpy          {numpy.__version__}")
    print(f"model dir      {settings.model_dir}")
    print(f"model          {spec.name} ({spec.repo}, {spec.kind}, {spec.dim} dim)")
    print(f"files          {'present' if is_present(model_root(settings.model_dir, spec), spec) else 'missing'}")
    print(f"workers/thread {settings.workers} / {settings.threads}, batch <= {settings.max_batch}")
    print(f"prefix mode    {settings.prefix_mode} ({spec.prefix_for(settings.prefix_mode)!r})")
    print(f"offline        {settings.offline}")
    print(f"auth           {'bearer required' if settings.api_key else 'open'}")
    print(_cuda_flags())
    return 0


def _cuda_flags() -> str:
    """Report the CPU features that decide whether numpy even imports here."""
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return "cpu flags      unknown (no /proc/cpuinfo)"
    model = next((line.split(":", 1)[1].strip() for line in flags.splitlines() if line.startswith("model name")), "?")
    wanted = ("sse4_1", "sse4_2", "avx", "avx2", "avx512f")
    present = [flag for flag in wanted if f" {flag} " in f" {_flags_line(flags)} "]
    return f"cpu            {model}\n               has: {', '.join(present) or 'none of ' + ', '.join(wanted)}"


def _flags_line(cpuinfo: str) -> str:
    for line in cpuinfo.splitlines():
        if line.startswith("flags"):
            return line.split(":", 1)[1].strip()
    return ""


def _ensure_model(spec: ModelSpec, settings: Settings, *, allow_download: bool) -> Path:
    from .store import ensure

    return ensure(spec, settings.model_dir, allow_download=allow_download and not settings.offline, progress=_progress)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trundle", description="OpenAI-compatible embedding server for small CPUs.")
    parser.add_argument("--version", action="version", version=f"trundle {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    def common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--model", help=f"model name (default: {DEFAULT_MODEL})")
        target.add_argument("--model-dir", help="where models are stored (default: /models)")
        target.add_argument("--log-level", choices=("debug", "info", "warning", "error"))

    serve = subparsers.add_parser("serve", help="load a model and serve the API")
    common(serve)
    serve.add_argument("--host", help="bind address (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, help="bind port (default: 8000)")
    serve.add_argument("--workers", type=int, help="concurrent embedding threads (default: 1)")
    serve.add_argument("--threads", type=int, help="onnxruntime intra-op threads (default: 1)")
    serve.add_argument("--max-batch", type=int, help="sequences per forward pass (default: 1)")
    serve.add_argument("--max-inputs", type=int, help="inputs accepted per request (default: 256)")
    serve.add_argument("--max-length", type=int, help="token limit per input (0 = model default)")
    serve.add_argument("--prefix-mode", choices=("document", "query", "none"), help="prefix for unprefixed calls")
    serve.add_argument("--no-download", action="store_true", help="never fetch a missing model")
    serve.set_defaults(func=cmd_serve)

    pull_cmd = subparsers.add_parser("pull", help="download a model into the model directory")
    common(pull_cmd)
    pull_cmd.add_argument("--all", action="store_true", help="fetch every known model")
    pull_cmd.add_argument("--force", action="store_true", help="re-download even if present")
    pull_cmd.set_defaults(func=cmd_pull)

    list_cmd = subparsers.add_parser("list", help="show known models and what is on disk")
    list_cmd.add_argument("--model-dir", help="where models are stored (default: /models)")
    list_cmd.set_defaults(func=cmd_list)

    info_cmd = subparsers.add_parser("info", help="environment report")
    common(info_cmd)
    info_cmd.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", level="INFO")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    try:
        return int(args.func(args) or 0)
    except UnknownModel as error:
        LOG.error("%s", error)
        return 2
    except ConfigError as error:
        LOG.error("configuration: %s", error)
        return 2
    except StoreError as error:
        LOG.error("%s", error)
        return 1
    except KeyboardInterrupt:
        return 130
