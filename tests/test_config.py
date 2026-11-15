"""Settings come from TRUNDLE_* env, then CLI overrides win."""

from __future__ import annotations

import os

import pytest

from trundle.config import ConfigError, Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(("TRUNDLE_", "HF_HUB")):
            monkeypatch.delenv(name, raising=False)


def test_defaults() -> None:
    settings = Settings.from_env()
    assert settings.model == "multilingual-e5-small"
    assert settings.workers == 1
    assert settings.threads == 1
    assert settings.prefix_mode == "document"
    assert settings.offline is False
    assert settings.api_key is None


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRUNDLE_MODEL", "bge-small-en-v1.5")
    monkeypatch.setenv("TRUNDLE_MODEL_DIR", "/data/models")
    monkeypatch.setenv("TRUNDLE_PORT", "9001")
    monkeypatch.setenv("TRUNDLE_PREFIX_MODE", "none")
    monkeypatch.setenv("TRUNDLE_API_KEY", "secret")
    settings = Settings.from_env()
    assert settings.model == "bge-small-en-v1.5"
    assert str(settings.model_dir) == "/data/models"
    assert settings.port == 9001
    assert settings.prefix_mode == "none"
    assert settings.api_key == "secret"


def test_hf_offline_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert Settings.from_env().offline is True


def test_cli_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRUNDLE_PORT", "9001")
    settings = Settings.from_env().with_overrides(port=7000, model=None)
    assert settings.port == 7000


def test_empty_env_value_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRUNDLE_MODEL", "")
    assert Settings.from_env().model == "multilingual-e5-small"


@pytest.mark.parametrize(
    "key, value",
    [
        ("TRUNDLE_PORT", "not-a-port"),
        ("TRUNDLE_PREFIX_MODE", "both"),
        ("TRUNDLE_OFFLINE", "maybe"),
        ("TRUNDLE_WORKERS", "0"),
    ],
)
def test_bad_values_raise(
    monkeypatch: pytest.MonkeyPatch, key: str, value: str
) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(ConfigError):
        Settings.from_env()
