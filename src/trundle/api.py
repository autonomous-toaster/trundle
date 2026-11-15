"""The HTTP surface: just enough of the OpenAI API for RAG pipelines.

`POST /v1/embeddings` is what Open WebUI, LangChain and LlamaIndex call, so that
is the contract to get exactly right: same field names, same `data[].embedding`
ordering, same `usage`. Everything else here is operational.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Sequence

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import Settings
from .embedder import Embedder, EmbResult

__version__ = "0.1.0"


class ApiError(Exception):
    """An error shaped like the OpenAI API's, because that is what clients parse."""

    def __init__(self, status: int, message: str, *, code: str = "invalid_request_error") -> None:
        self.status = status
        self.message = message
        self.code = code
        super().__init__(message)


class EmbeddingsRequest(BaseModel):
    # Typed as Any so that garbage arrives here and gets a 400 with a readable
    # message, instead of FastAPI's default 422 that OpenAI clients cannot parse.
    input: str | list[Any] = Field(..., description="text or list of texts")
    model: str | None = None
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = Field(None, description="truncate and renormalise (matryoshka models only)")
    # Not part of the OpenAI schema: RAG clients index documents and search with
    # queries, and e5-style models want different prefixes for the two.
    input_type: Literal["query", "document"] | None = None
    user: str | None = None


@dataclass
class Stats:
    requests: int = 0
    inputs: int = 0
    tokens: int = 0
    seconds: float = 0.0
    started_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "inputs": self.inputs,
            "tokens": self.tokens,
            "embedding_seconds": round(self.seconds, 2),
            "uptime_s": round(time.time() - self.started_at, 1),
        }


def _encode(vector: Sequence[float], encoding_format: str) -> list[float] | str:
    if encoding_format == "base64":
        return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")
    return list(vector)


def create_app(embedder: Embedder, settings: Settings) -> FastAPI:
    """Build the ASGI app around an already-loaded embedder."""
    stats = Stats()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.pool = ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="trundle")
        try:
            yield
        finally:
            application.state.pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="trundle", version=__version__, docs_url="/docs", redoc_url=None, lifespan=lifespan)
    app.state.embedder = embedder
    app.state.settings = settings
    app.state.stats = stats

    def check_auth(request: Request) -> None:
        expected = settings.api_key
        if not expected:
            return
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip(), expected):
            raise ApiError(401, "invalid or missing API key", code="invalid_api_key")

    @app.exception_handler(ApiError)
    async def _api_error(_request: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status,
            content={"error": {"message": error.message, "type": "invalid_request_error", "code": error.code}},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'][1:]) or 'body'}: {item['msg']}"
            for item in error.errors()
        )
        return JSONResponse(
            status_code=400,
            content={"error": {"message": detail, "type": "invalid_request_error", "code": "invalid_request_error"}},
        )

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {"name": "trundle", "version": __version__, **embedder.info()}

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Liveness only: the model is loaded before the server binds."""
        return {"ok": True, **stats.as_dict()}

    @app.get("/readyz")
    async def readyz() -> dict[str, Any]:
        return {"ok": True, "model": embedder.spec.name}

    @app.get("/v1/models")
    @app.get("/models")
    async def models(_: None = Depends(check_auth)) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": embedder.spec.name,
                    "object": "model",
                    "owned_by": "trundle",
                    "dim": embedder.dim,
                }
            ],
        }

    @app.post("/v1/embeddings")
    @app.post("/embeddings")
    async def embeddings(body: EmbeddingsRequest, _: None = Depends(check_auth)) -> dict[str, Any]:
        texts = [body.input] if isinstance(body.input, str) else list(body.input)
        if not texts:
            raise ApiError(400, "input must not be empty")
        if any(not isinstance(text, str) for text in texts):
            raise ApiError(400, "input must be a string or a list of strings (token arrays are not supported)")
        if len(texts) > settings.max_inputs:
            raise ApiError(
                400,
                f"{len(texts)} inputs in one request; the limit here is {settings.max_inputs}. "
                "Send smaller batches: this service is sized for a small CPU.",
            )
        mode = body.input_type or settings.prefix_mode
        if mode == "none":
            mode = "document"

        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        try:
            result: EmbResult = await loop.run_in_executor(
                app.state.pool, lambda: embedder.embed(texts, mode=mode, dimensions=body.dimensions)
            )
        except Exception as error:  # noqa: BLE001 - surface the reason, do not 500 silently
            raise ApiError(500, f"embedding failed: {error}", code="internal_error") from error
        elapsed = time.perf_counter() - started

        stats.requests += 1
        stats.inputs += len(texts)
        stats.tokens += result.tokens
        stats.seconds += elapsed

        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": index, "embedding": _encode(vector, body.encoding_format)}
                for index, vector in enumerate(result.vectors)
            ],
            "model": body.model or embedder.spec.name,
            "usage": {"prompt_tokens": result.tokens, "total_tokens": result.tokens},
        }

    return app
