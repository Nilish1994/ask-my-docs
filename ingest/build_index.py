"""Fetch the FastAPI docs, chunk them, embed them and write the index.

Usage:
    python -m ingest.build_index [--tag 0.162.7] [--index-dir data/index]

Writes ``chunks.jsonl``, ``vectors.npy`` and ``manifest.json``. The index is
derived from FastAPI's MIT-licensed docs and is not committed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from retrieval.embeddings import FastEmbedEmbedder
from retrieval.index import Retriever

from .corpus import build_fastapi_chunks
from .fetch_docs import FASTAPI_TAG, fetch


def build(tag: str = FASTAPI_TAG, raw: Path = Path("data/raw"), index_dir: Path = Path("data/index")) -> Retriever:
    raw_root = fetch(tag, raw)
    chunks = build_fastapi_chunks(raw_root)
    print(f"{len(chunks)} chunks from {len({c['page'] for c in chunks})} pages; embedding...")
    started = time.perf_counter()
    retriever = Retriever.build(chunks, FastEmbedEmbedder())
    print(f"embedded in {time.perf_counter() - started:.1f}s")
    retriever.save(index_dir, {"corpus": "fastapi-docs", "fastapi_tag": tag})
    print(f"index written to {index_dir}")
    return retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default=FASTAPI_TAG)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    args = parser.parse_args()
    build(args.tag, args.raw, args.index_dir)


if __name__ == "__main__":
    main()
