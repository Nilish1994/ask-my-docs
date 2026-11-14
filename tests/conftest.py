"""Shared fixtures: a tiny corpus, a deterministic fake embedder and a fake
Anthropic client. No test touches the network or downloads a model."""

from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace

import numpy as np
import pytest

from ingest.corpus import chunk_page, page_url
from ingest.markdown import parse_sections
from retrieval.index import Retriever

PAGES = {
    "tutorial/query-params": """# Query Parameters { #query-parameters }

When you declare function parameters that are not part of the path, they are query parameters.

## Optional parameters { #optional-parameters }

Set a default of None to make a query parameter optional.

```python
async def read_item(q: str | None = None):
    return {"q": q}
```
""",
    "tutorial/cors": """# CORS (Cross-Origin Resource Sharing) { #cors-cross-origin-resource-sharing }

A frontend running in a browser on another origin needs CORS headers.

## Use CORSMiddleware { #use-corsmiddleware }

Add CORSMiddleware with the list of allowed origins.
""",
    "advanced/events": """# Lifespan Events { #lifespan-events }

Use the lifespan parameter to run code once when the application starts up and when it shuts down,
for example to load a machine learning model.
""",
}


class FakeEmbedder:
    """Hashed bag-of-words vectors: similar wording gives similar vectors."""

    name = "fake-hash-embedder"
    dimensions = 64

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimensions] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


def build_chunks() -> list[dict]:
    chunks = []
    for page, text in PAGES.items():
        chunks.extend(chunk_page(parse_sections(text), source="fastapi", page=page, url_for=page_url))
    for position, chunk in enumerate(chunks):
        chunk["id"] = position
    return chunks


@pytest.fixture(autouse=True)
def no_real_anthropic(monkeypatch):
    """Tests must never reach the real API, whatever the developer's shell exports."""
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")  # unroutable if anything slips through


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def chunks() -> list[dict]:
    return build_chunks()


@pytest.fixture
def retriever(chunks, embedder) -> Retriever:
    return Retriever.build(chunks, embedder)


# --- fake Anthropic streaming client -------------------------------------------------


def ev(kind: str, **fields) -> SimpleNamespace:
    return SimpleNamespace(type=kind, **fields)


def text_block_start() -> SimpleNamespace:
    return ev("content_block_start", content_block=SimpleNamespace(type="text"))


def text_delta(text: str) -> SimpleNamespace:
    return ev("content_block_delta", delta=SimpleNamespace(type="text_delta", text=text))


def citation_delta(document_index: int, start: int, cited_text: str = "cited") -> SimpleNamespace:
    citation = SimpleNamespace(
        type="char_location",
        document_index=document_index,
        start_char_index=start,
        end_char_index=start + len(cited_text),
        cited_text=cited_text,
        document_title=None,
    )
    return ev("content_block_delta", delta=SimpleNamespace(type="citations_delta", citation=citation))


def final_message(stop_reason: str = "end_turn", model: str = "claude-opus-5-5", category=None):
    return SimpleNamespace(
        stop_reason=stop_reason,
        model=model,
        usage=SimpleNamespace(input_tokens=1200, output_tokens=150),
        stop_details=SimpleNamespace(category=category) if stop_reason == "refusal" else None,
    )


class FakeStream:
    def __init__(self, events, message, error=None):
        self._events, self._message, self._error = events, message, error

    async def __aenter__(self):
        if self._error:
            raise self._error
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def generate():
            for event in self._events:
                yield event

        return generate()

    async def get_final_message(self):
        return self._message


class FakeAnthropic:
    """Records every stream() call; replays a scripted list of events."""

    def __init__(self, events=None, message=None, error=None):
        self.calls: list[dict] = []
        self.events = events or []
        self.message = message or final_message()
        self.error = error
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStream(self.events, self.message, self.error)

    async def close(self):
        pass
