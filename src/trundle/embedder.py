"""The embedder: ONNX Runtime (or a static model2vec model) plus pooling.

Everything here is shaped by one measurement: on a 2-core CPU with no AVX,
onnxruntime spends its time in the forward pass of a single sequence. Batching
does not help, more threads make it slower, so we embed one small batch at a
time with one intra-op thread and let a thread pool provide concurrency.

Memory is the other constraint. onnxruntime allocates activations for the whole
batch and its CPU arena keeps them, which once pushed a process to 7.6 GB on
this box; the batch cap and disabled arena below are what keep it at ~600 MB.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import ModelSpec
from .store import model_root


class EmbedderError(RuntimeError):
    """The model could not be loaded or used."""


@dataclass
class EmbResult:
    vectors: list[list[float]]
    tokens: int


def batches(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def pool(last_hidden: np.ndarray, mask: np.ndarray, pooling: str) -> np.ndarray:
    """Turn (batch, tokens, dim) into (batch, dim)."""
    if pooling == "cls":
        return last_hidden[:, 0, :]
    weights = mask[..., None].astype(np.float32)
    return (last_hidden * weights).sum(1) / np.clip(weights.sum(1), 1e-9, None)


def normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)


def truncate(matrix: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka-style truncation: keep the first `dim` components, renormalise.

    Only meaningful for models trained that way (nomic, e5-multilingual-large);
    for others it degrades quality, so it is opt-in per request.
    """
    if dim <= 0 or dim > matrix.shape[1]:
        raise EmbedderError(
            f"dimensions must be between 1 and {matrix.shape[1]}, got {dim}"
        )
    if dim == matrix.shape[1]:
        return matrix
    return normalize(matrix[:, :dim])


class Embedder:
    """One loaded model, safe to call from several threads."""

    def __init__(
        self,
        spec: ModelSpec,
        model_dir: Path,
        *,
        threads: int = 1,
        max_batch: int = 16,
        max_length: int = 0,
    ) -> None:
        self.spec = spec
        self.root = model_root(model_dir, spec)
        self.threads = threads
        self.max_batch = max_batch
        self.max_length = max_length or spec.max_length
        # The tokenizer is shared across pool threads; serialising it costs
        # nothing next to a 500 ms forward pass and removes a class of races.
        self._tokenizer_lock = threading.Lock()
        self._session = None
        self._static = None
        self._input_names: set[str] = set()
        self._load()

    # ---------------------------------------------------------------- loading

    def _load(self) -> None:
        from tokenizers import Tokenizer

        tokenizer_path = self.root / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise EmbedderError(
                f"{tokenizer_path} is missing; run `trundle pull --model {self.spec.name}`"
            )
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(
            max_length=min(self.spec.max_length, self.max_length)
        )
        pad_id = self.tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            pad_id = self.tokenizer.token_to_id("<pad>")
        if pad_id is None:
            pad_id = 0
        self.tokenizer.enable_padding(
            pad_id=pad_id, pad_token=self.tokenizer.id_to_token(pad_id) or "[PAD]"
        )

        if self.spec.kind == "static":
            self._load_static()
        else:
            self._load_onnx()

    def _load_static(self) -> None:
        try:
            from model2vec import StaticModel
        except ImportError:
            raise EmbedderError(
                "static models need model2vec: pip install 'trundle-embed[static]'"
            ) from None
        self._static = StaticModel.from_pretrained(str(self.root))

    def _load_onnx(self) -> None:
        import onnxruntime as ort

        model_file = self.root / (self.spec.onnx_file or "model.onnx")
        if not model_file.is_file():
            raise EmbedderError(
                f"{model_file} is missing; run `trundle pull --model {self.spec.name}`"
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # An arena plus varying batch shapes is an unbounded RSS on a small box.
        options.enable_cpu_mem_arena = False
        self._session = ort.InferenceSession(
            str(model_file), options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {entry.name for entry in self._session.get_inputs()}

    # -------------------------------------------------------------- embedding

    @property
    def dim(self) -> int:
        return self.spec.dim

    def count_tokens(self, texts: Sequence[str]) -> int:
        with self._tokenizer_lock:
            return sum(
                len(encoding.ids)
                for encoding in self.tokenizer.encode_batch(list(texts))
            )

    def embed(
        self,
        texts: Sequence[str],
        *,
        mode: str = "document",
        dimensions: int | None = None,
    ) -> EmbResult:
        if not texts:
            return EmbResult([], 0)
        prefix = self.spec.prefix_for(mode)
        prepared = [prefix + text for text in texts] if prefix else list(texts)
        tokens = self.count_tokens(prepared)

        vectors: list[list[float]] = []
        for batch in batches(prepared, self.max_batch):
            vectors.extend(self._embed_batch(list(batch)))

        if dimensions is not None and dimensions != self.dim:
            matrix = truncate(np.asarray(vectors, dtype=np.float32), dimensions)
            vectors = matrix.tolist()
        return EmbResult(vectors, tokens)

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        if self._static is not None:
            matrix = np.asarray(self._static.encode(batch), dtype=np.float32)
            return normalize(matrix).tolist()

        with self._tokenizer_lock:
            encodings = self.tokenizer.encode_batch(batch)
            ids = np.asarray([encoding.ids for encoding in encodings], dtype=np.int64)
            mask = np.asarray(
                [encoding.attention_mask for encoding in encodings], dtype=np.int64
            )

        feed: dict[str, np.ndarray] = {}
        if "input_ids" in self._input_names:
            feed["input_ids"] = ids
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = mask
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)

        output = np.asarray(self._session.run(None, feed)[0], dtype=np.float32)
        # Some exports come pooled (batch, dim), most return the hidden states.
        matrix = output if output.ndim == 2 else pool(output, mask, self.spec.pooling)
        return normalize(matrix).tolist()

    def info(self) -> dict[str, object]:
        return {
            "model": self.spec.name,
            "repo": self.spec.repo,
            "kind": self.spec.kind,
            "dim": self.dim,
            "max_length": min(self.spec.max_length, self.max_length),
            "batch_limit": self.max_batch,
            "intra_op_threads": self.threads,
            "path": str(self.root),
        }
