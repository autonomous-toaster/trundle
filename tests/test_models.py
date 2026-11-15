"""The registry is the contract: names, dimensions and prefixes must not drift."""

from __future__ import annotations

import pytest

from trundle.models import DEFAULT_MODEL, REGISTRY, UnknownModel, resolve


def test_default_model_is_in_the_registry() -> None:
    assert DEFAULT_MODEL in REGISTRY


def test_every_spec_is_downloadable() -> None:
    for name, spec in REGISTRY.items():
        assert spec.name == name
        assert spec.files, name
        assert spec.dim > 0
        if spec.kind == "onnx":
            assert spec.onnx_file in spec.files, name
            assert "tokenizer.json" in spec.files, name
        else:
            assert "model.safetensors" in spec.files, name


def test_unknown_model_lists_alternatives() -> None:
    with pytest.raises(UnknownModel) as error:
        resolve("mxbai-embed-large-v1")
    assert "multilingual-e5-small" in str(error.value)


@pytest.mark.parametrize(
    "name, mode, expected",
    [
        ("multilingual-e5-small", "document", "passage: "),
        ("multilingual-e5-small", "query", "query: "),
        ("multilingual-e5-small", "none", ""),
        ("bge-small-en-v1.5", "document", ""),
        ("arctic-embed-xs", "query", "Represent this sentence"),
    ],
)
def test_prefixes_follow_the_mode(name: str, mode: str, expected: str) -> None:
    assert resolve(name).prefix_for(mode).startswith(expected)
