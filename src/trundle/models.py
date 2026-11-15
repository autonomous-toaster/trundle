"""The model registry.

Every entry was measured on the reference box (AMD Turion II Neo N54L, 2 cores,
no AVX/SSE4.2) with 128-token chunks, 1 worker / 1 intra-op thread. Numbers are
there for the operator to choose, not to be trusted off that box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["onnx", "static"]
Pooling = Literal["mean", "cls"]


class UnknownModel(KeyError):
    """Raised when a model name is not in the registry."""

    def __init__(self, name: str, known: list[str]) -> None:
        self.name = name
        self.known = known
        super().__init__(f"unknown model {name!r}; known: {', '.join(known)}")


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to download, load and correctly call one model."""

    name: str
    repo: str
    kind: Kind
    dim: int
    files: tuple[str, ...]
    onnx_file: str | None = None
    pooling: Pooling = "mean"
    query_prefix: str = ""
    document_prefix: str = ""
    max_length: int = 512
    revision: str = "main"
    mib: float = 0.0
    ms_per_chunk: float | None = None
    notes: str = ""

    def prefix_for(self, mode: str) -> str:
        if mode == "query":
            return self.query_prefix
        if mode == "document":
            return self.document_prefix
        return ""


BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "

REGISTRY: dict[str, ModelSpec] = {
    "multilingual-e5-small": ModelSpec(
        name="multilingual-e5-small",
        repo="Xenova/multilingual-e5-small",
        kind="onnx",
        dim=384,
        files=("onnx/model_quantized.onnx", "tokenizer.json"),
        onnx_file="onnx/model_quantized.onnx",
        pooling="mean",
        query_prefix="query: ",
        document_prefix="passage: ",
        max_length=512,
        mib=118,
        ms_per_chunk=538,
        notes="default: best French, English on par with bge-small, 100+ languages",
    ),
    "bge-small-en-v1.5": ModelSpec(
        name="bge-small-en-v1.5",
        repo="Xenova/bge-small-en-v1.5",
        kind="onnx",
        dim=384,
        files=("onnx/model.onnx", "tokenizer.json"),
        onnx_file="onnx/model.onnx",
        pooling="cls",
        max_length=512,
        mib=133,
        ms_per_chunk=625,
        notes="best English quality here, no prefixes needed, 512 tokens",
    ),
    "all-MiniLM-L6-v2": ModelSpec(
        name="all-MiniLM-L6-v2",
        repo="Xenova/all-MiniLM-L6-v2",
        kind="onnx",
        dim=384,
        files=("onnx/model_quantized.onnx", "tokenizer.json"),
        onnx_file="onnx/model_quantized.onnx",
        pooling="mean",
        max_length=256,
        mib=23,
        ms_per_chunk=272,
        notes="small and fast for English, 256-token limit",
    ),
    "arctic-embed-xs": ModelSpec(
        name="arctic-embed-xs",
        repo="Snowflake/snowflake-arctic-embed-xs",
        kind="onnx",
        dim=384,
        files=("onnx/model.onnx", "tokenizer.json"),
        onnx_file="onnx/model.onnx",
        pooling="cls",
        query_prefix=BGE_INSTRUCTION,
        max_length=512,
        mib=90,
        ms_per_chunk=307,
        notes="22M params, English, needs the instruction prefix on queries",
    ),
    "potion-retrieval-32M": ModelSpec(
        name="potion-retrieval-32M",
        repo="minishlab/potion-retrieval-32M",
        kind="static",
        dim=512,
        files=("model.safetensors", "tokenizer.json", "config.json"),
        max_length=512,
        mib=129,
        ms_per_chunk=2,
        notes="static embeddings: 400x faster, English only, quality below bge-small",
    ),
    "potion-multilingual-128M": ModelSpec(
        name="potion-multilingual-128M",
        repo="minishlab/potion-multilingual-128M",
        kind="static",
        dim=256,
        files=("model.safetensors", "tokenizer.json", "config.json"),
        max_length=512,
        mib=512,
        ms_per_chunk=2,
        notes="static and multilingual: fast, but well behind e5-small on French",
    ),
}

DEFAULT_MODEL = "multilingual-e5-small"


def resolve(name: str) -> ModelSpec:
    """Look up a model, failing with the list of things that do exist."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise UnknownModel(name, sorted(REGISTRY)) from None
