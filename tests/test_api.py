"""The OpenAI-compatible surface, exercised with a fake embedder (no model on disk)."""

from __future__ import annotations

import base64
import struct

import pytest
from fastapi.testclient import TestClient

from trundle.api import create_app
from trundle.config import Settings
from trundle.embedder import EmbResult
from trundle.models import resolve

DIM = 4


class FakeEmbedder:
    """Stands in for a real model so the API can be tested without a download."""

    spec = resolve("multilingual-e5-small")

    def __init__(self, dim: int = DIM) -> None:
        self.calls: list[tuple[list[str], str, int | None]] = []
        self.dim_value = dim

    @property
    def dim(self) -> int:
        return self.dim_value

    def embed(self, texts, *, mode="document", dimensions=None) -> EmbResult:
        self.calls.append((list(texts), mode, dimensions))
        return EmbResult([[1.0, 0.0, 0.0, 0.0] for _ in texts], tokens=3 * len(texts))

    def info(self) -> dict[str, object]:
        return {"model": self.spec.name, "dim": self.dim, "kind": "onnx", "path": "/models/x"}


@pytest.fixture
def client() -> TestClient:
    app = create_app(FakeEmbedder(), Settings(prefix_mode="document").validated())
    with TestClient(app) as test_client:
        yield test_client


def test_string_input_returns_one_vector(client: TestClient) -> None:
    response = client.post("/v1/embeddings", json={"input": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["index"] == 0
    assert len(body["data"][0]["embedding"]) == DIM
    assert body["usage"] == {"prompt_tokens": 3, "total_tokens": 3}


def test_list_input_keeps_order(client: TestClient) -> None:
    response = client.post("/v1/embeddings", json={"input": ["a", "b", "c"]})
    indexes = [item["index"] for item in response.json()["data"]]
    assert indexes == [0, 1, 2]


def test_document_prefix_is_the_default_mode(client: TestClient) -> None:
    app_embedder = client.app.state.embedder
    client.post("/v1/embeddings", json={"input": "hello"})
    assert app_embedder.calls[-1][1] == "document"

    client.post("/v1/embeddings", json={"input": "hello", "input_type": "query"})
    assert app_embedder.calls[-1][1] == "query"


def test_base64_encoding(client: TestClient) -> None:
    response = client.post("/v1/embeddings", json={"input": "hello", "encoding_format": "base64"})
    payload = response.json()["data"][0]["embedding"]
    decoded = struct.unpack("<4f", base64.b64decode(payload))
    assert decoded == (1.0, 0.0, 0.0, 0.0)


def test_dimensions_are_forwarded(client: TestClient) -> None:
    client.post("/v1/embeddings", json={"input": "hello", "dimensions": 2})
    assert client.app.state.embedder.calls[-1][2] == 2


@pytest.mark.parametrize("payload", [{"input": []}, {"input": [1, 2, 3]}])
def test_bad_input_is_a_400(client: TestClient, payload: dict) -> None:
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request_error"


def test_too_many_inputs_is_refused(client: TestClient) -> None:
    response = client.post("/v1/embeddings", json={"input": ["x"] * 300})
    assert response.status_code == 400
    assert "limit here is 256" in response.json()["error"]["message"]


def test_models_and_health(client: TestClient) -> None:
    models = client.get("/v1/models").json()
    assert models["data"][0]["id"] == "multilingual-e5-small"
    assert models["data"][0]["dim"] == DIM
    assert client.get("/healthz").json()["ok"] is True
    assert client.get("/readyz").json()["ok"] is True


def test_auth_is_off_by_default(client: TestClient) -> None:
    assert client.post("/v1/embeddings", json={"input": "x"}).status_code == 200


def test_auth_guard() -> None:
    settings = Settings(api_key="s3cret").validated()
    with TestClient(create_app(FakeEmbedder(), settings)) as client:
        assert client.post("/v1/embeddings", json={"input": "x"}).status_code == 401
        bad = {"Authorization": "Bearer nope"}
        assert client.post("/v1/embeddings", json={"input": "x"}, headers=bad).status_code == 401
        good = {"Authorization": "Bearer s3cret"}
        assert client.post("/v1/embeddings", json={"input": "x"}, headers=good).status_code == 200
        # Health endpoints stay open so containers can be probed without the key.
        assert client.get("/healthz").status_code == 200
