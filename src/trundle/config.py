"""Settings: environment first, CLI flags override.

Everything a deployment needs to decide lives in `TRUNDLE_*` variables so the
container can be configured without a config file or a writable filesystem.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ENV_PREFIX = "TRUNDLE_"

PREFIX_MODES = ("document", "query", "none")
LOG_LEVELS = ("debug", "info", "warning", "error")


class ConfigError(ValueError):
    """Bad or contradictory settings."""


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(ENV_PREFIX + name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(ENV_PREFIX + name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(
            f"{ENV_PREFIX}{name} must be an integer, got {raw!r}"
        ) from None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(ENV_PREFIX + name)
    if raw is None or raw == "":
        return default
    lowered = raw.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{ENV_PREFIX}{name} must be a boolean, got {raw!r}")


@dataclass(frozen=True)
class Settings:
    model: str = "multilingual-e5-small"
    model_dir: Path = Path("/models")
    host: str = "127.0.0.1"
    port: int = 8000
    # One thread per worker is not a compromise here: on the reference box 1
    # thread beat 2 by 1.8x, and a second worker adds barely 10 %.
    workers: int = 1
    threads: int = 1
    # One sequence per forward pass. Measured on the reference box for a 32-chunk
    # request: batch 1 = 29 s, 4 = 33 s, 8 = 34 s, 16 = 41 s, with a higher RSS
    # too. Padding to the longest sequence costs more than the reuse it buys.
    max_batch: int = 1
    max_inputs: int = 256
    max_length: int = 0  # 0 = whatever the model allows
    prefix_mode: str = "document"
    api_key: str | None = None
    log_level: str = "info"
    offline: bool = False

    def validated(self) -> Settings:
        if not 1 <= self.port <= 65535:
            raise ConfigError(f"port out of range: {self.port}")
        if self.workers < 1:
            raise ConfigError(f"workers must be >= 1, got {self.workers}")
        if self.threads < 1:
            raise ConfigError(f"threads must be >= 1, got {self.threads}")
        if self.max_batch < 1:
            raise ConfigError(f"max_batch must be >= 1, got {self.max_batch}")
        if self.max_inputs < 1:
            raise ConfigError(f"max_inputs must be >= 1, got {self.max_inputs}")
        if self.max_length < 0:
            raise ConfigError(f"max_length must be >= 0, got {self.max_length}")
        if self.prefix_mode not in PREFIX_MODES:
            raise ConfigError(
                f"prefix_mode must be one of {PREFIX_MODES}, got {self.prefix_mode!r}"
            )
        if self.log_level not in LOG_LEVELS:
            raise ConfigError(
                f"log_level must be one of {LOG_LEVELS}, got {self.log_level!r}"
            )
        return self

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls(
            model=_env_str("MODEL", cls.model),
            model_dir=Path(_env_str("MODEL_DIR", str(cls.model_dir))),
            host=_env_str("HOST", cls.host),
            port=_env_int("PORT", cls.port),
            workers=_env_int("WORKERS", cls.workers),
            threads=_env_int("THREADS", cls.threads),
            max_batch=_env_int("MAX_BATCH", cls.max_batch),
            max_inputs=_env_int("MAX_INPUTS", cls.max_inputs),
            max_length=_env_int("MAX_LENGTH", cls.max_length),
            prefix_mode=_env_str("PREFIX_MODE", cls.prefix_mode),
            api_key=os.environ.get(ENV_PREFIX + "API_KEY") or None,
            log_level=_env_str("LOG_LEVEL", cls.log_level).lower(),
            offline=_env_bool("OFFLINE", cls.offline) or _hub_offline(),
        )
        return settings.validated()

    def with_overrides(self, **overrides: Any) -> Settings:
        """Apply CLI overrides, ignoring the ones the user did not pass."""
        clean = {key: value for key, value in overrides.items() if value is not None}
        if "model_dir" in clean:
            clean["model_dir"] = Path(clean["model_dir"])
        return replace(self, **clean).validated()


def _hub_offline() -> bool:
    """Respect the Hugging Face convention so the same env works for both tools."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
