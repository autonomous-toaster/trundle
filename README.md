# trundle

An OpenAI-compatible embedding server that trundles along on CPUs with no AVX.

It exists because the alternative — "just run sentence-transformers" — costs 5.8×
more time and a torch install on hardware like this. `trundle` speaks the one
endpoint RAG pipelines actually call (`POST /v1/embeddings`), runs ONNX Runtime
with the thread settings that were measured to be fastest on a 2-core K10 CPU,
and ships as a distroless image with the model on a volume.

```bash
docker run -d --name trundle -p 127.0.0.1:8000:8000 \
  -v trundle-models:/models \
  ghcr.io/autonomous-toaster/trundle:latest

curl -s localhost:8000/v1/embeddings -H 'content-type: application/json' \
  -d '{"input": ["bonjour le monde"], "model": "multilingual-e5-small"}'
```

The model is fetched on first start into `/models`. Prefer to do it explicitly:

```bash
docker run --rm -v trundle-models:/models ghcr.io/autonomous-toaster/trundle:latest pull
```

## Why these defaults

Every number below was measured on the reference box (AMD Turion II Neo N54L:
2 cores @ 2.2 GHz, 15 GiB RAM, no AVX, no SSE4.2, no VNNI), 128-token chunks,
one worker, one intra-op thread:

| model | dim | MiB | ms/chunk | 10 000 chunks | nDCG EN | nDCG FR |
|---|---:|---:|---:|---:|---:|---:|
| **multilingual-e5-small** (default) | 384 | 118 | 538 | **1.5 h** | 0.771 | **0.965** |
| bge-small-en-v1.5 | 384 | 133 | 625 | 1.7 h | **0.808** | 0.871 |
| arctic-embed-xs | 384 | 90 | 307 | 51 min | 0.755 | – |
| all-MiniLM-L6-v2 | 384 | 23 | 272 | 45 min | 0.780 | – |
| potion-retrieval-32M (static) | 512 | 129 | 2 | 16 s | 0.754 | 0.613 |

Those figures come from the benchmark, which ran while the box was otherwise
mostly quiet. Re-measured with a load average of 4 on the same 2 cores the same
models are 1.4× slower — and so is the original benchmark script, so the drift is
the machine, not the service. Treat the table as relative, not absolute.

That is why `workers=1`, `threads=1` and `max_batch=1` are the defaults: on this
class of CPU one thread beats two by 1.8×, and batching *hurts* — for the same
32-chunk request the image took 29 s at batch 1, 33 s at 4, 34 s at 8 and 41 s at
batch 16, with a larger resident set too. Padding every sequence to the longest
one costs more than the reuse it buys. A second worker adds about 10 %. Query-time
cost is ~95 ms for e5-small, irrelevant next to generating an answer.

## CLI

```
trundle serve         # load a model and serve the API
trundle pull          # download the configured model into the model directory
trundle pull --all    # ... or every known model
trundle list          # what the registry offers and what is already on disk
trundle info          # python, numpy, model dir, CPU flags, auth state
```

`trundle serve` downloads a missing model unless `--no-download` is passed or
`TRUNDLE_OFFLINE=1` is set; in that case it exits non-zero and says which files are
missing rather than serving something wrong.

## Configuration

Environment first (a container has no writable config file), CLI flags override.

| variable | default | meaning |
|---|---|---|
| `TRUNDLE_MODEL` | `multilingual-e5-small` | registry entry to load |
| `TRUNDLE_MODEL_DIR` | `/models` | where models live; mount a volume here |
| `TRUNDLE_HOST` | `127.0.0.1` | bind address (the image sets `0.0.0.0`) |
| `TRUNDLE_PORT` | `8000` | bind port |
| `TRUNDLE_WORKERS` | `1` | concurrent embedding threads |
| `TRUNDLE_THREADS` | `1` | onnxruntime intra-op threads |
| `TRUNDLE_MAX_BATCH` | `1` | sequences per forward pass |
| `TRUNDLE_MAX_INPUTS` | `256` | inputs accepted in one request |
| `TRUNDLE_MAX_LENGTH` | `0` | token cap per input, `0` = model default |
| `TRUNDLE_PREFIX_MODE` | `document` | `document`, `query` or `none`: prefix for callers that cannot say |
| `TRUNDLE_API_KEY` | unset | require `Authorization: Bearer <key>` on the API |
| `TRUNDLE_OFFLINE` | unset | never touch the network; also honours `HF_HUB_OFFLINE` |
| `TRUNDLE_LOG_LEVEL` | `info` | `debug`, `info`, `warning`, `error` |
| `TRUNDLE_HF_ENDPOINT` | `https://huggingface.co` | for mirrors |
| `HF_TOKEN` | unset | for gated repositories |

The model directory is a plain directory per model (`/models/<name>/<file>`), so it
can be inspected, copied or pre-seeded with `cp`. A **named** volume inherits the
image's ownership and is writable by the nonroot user; a **bind** mount does not —
`chown -R 65532:65532 <dir>` first, or run `pull` on the host and mount read-only.

## API

| route | purpose |
|---|---|
| `POST /v1/embeddings` (also `/embeddings`) | the OpenAI contract; `input`, `model`, `encoding_format`, `dimensions` |
| `GET /v1/models` | one entry, with `dim` |
| `GET /healthz` | liveness, counters (`requests`, `inputs`, `tokens`) |
| `GET /readyz` | the model is loaded (it always is once the port is bound) |
| `GET /docs` | the generated OpenAPI page |

Two additions beyond the OpenAI schema, both optional:

- `input_type: "query" | "document"` selects the model's prefix for that call.
  Clients that cannot send it get `TRUNDLE_PREFIX_MODE` (default `document`).
- `dimensions` truncates and renormalises the vector (matryoshka models only;
  harmless but quality-reducing on others). It is *not* a substitute for the
  vector-column migration described below.

## Wiring it into Open WebUI

*Settings → Documents*: embedding engine `OpenAI`, base URL `http://trundle:8000/v1`,
key = `TRUNDLE_API_KEY`, model = the registry name. Or in the environment:

```bash
RAG_EMBEDDING_ENGINE=openai
RAG_EMBEDDING_MODEL=multilingual-e5-small
OPENAI_API_BASE_URL=http://trundle:8000/v1
OPENAI_API_KEY=<TRUNDLE_API_KEY>
RAG_EMBEDDING_BATCH_SIZE=1
```

Open WebUI calls the same endpoint for indexing and for querying, so it can only
ever send one of the two e5 prefixes. `TRUNDLE_PREFIX_MODE=document` keeps
`passage: ` on everything — the indexing side (where quality is decided) stays
correct, and the query side loses a little. If that asymmetry bothers you, pick a
model with no distinct prefixes (`bge-small-en-v1.5`, `all-MiniLM-L6-v2`).

### Vector size, and why nothing here decides it

Open WebUI fixes the pgvector column dimension at startup from
`PGVECTOR_INITIALIZE_MAX_VECTOR_LENGTH` (default **1536**) and refuses to start if
the existing `document_chunk.vector` column disagrees with it. Vectors that arrive
with a different length are **zero-padded** to that width (`adjust_vector_length`),
so a 384-d model still works — cosine distance is unchanged — but every row is 4×
the bytes. Moving to a 384-d model properly is therefore a two-part migration:
set `PGVECTOR_INITIALIZE_MAX_VECTOR_LENGTH=384` *and* drop/re-embed the existing
collection. `trundle` reports its dimension in `/v1/models` so the two numbers can
be compared before any migration is attempted.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest                          # no model download, no network
python -m trundle info
python -m trundle serve --model bge-small-en-v1.5 --model-dir ./models
docker build -t trundle:dev .
```

`numpy` is pinned below 2.4 in the image (`docker/constraints.txt`): builds from
2.4 on target `x86-64-v2` and die with `SIGILL` (`Illegal instruction`) on these
CPUs. If a dependency ever drags a newer numpy in, the container will not start —
that is the constraint doing its job.

## Layout

```
src/trundle/config.py     settings from TRUNDLE_* env, validated
src/trundle/models.py     the registry: repo, files, dim, pooling, prefixes
src/trundle/store.py      model directory, resumable downloads, presence checks
src/trundle/embedder.py   ONNX runtime, pooling, batching, truncation
src/trundle/api.py        the OpenAI-compatible FastAPI app
src/trundle/cli.py        serve / pull / list / info
```
