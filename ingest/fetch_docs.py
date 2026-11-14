"""Download the FastAPI English docs at a pinned release tag.

The docs are MIT-licensed by Sebastián Ramírez (FastAPI). They are fetched at
build time and never committed to this repository.

Only two parts of the release tarball are extracted:
- ``docs/en/docs/**/*.md``: the Markdown pages served at fastapi.tiangolo.com
- ``docs_src/**/*.py``: example code that the pages include with ``{* ... *}``

Usage:
    python -m ingest.fetch_docs [--tag 0.162.7] [--out data/raw]
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path, PurePosixPath

import httpx

FASTAPI_TAG = "0.162.7"
TARBALL_URL = "https://codeload.github.com/fastapi/fastapi/tar.gz/refs/tags/{tag}"
KEEP_PREFIXES = ("docs/en/docs/", "docs_src/")
KEEP_SUFFIXES = (".md", ".py")


def raw_dir(out: Path, tag: str) -> Path:
    return out / f"fastapi-{tag}"


def _wanted(relative: PurePosixPath) -> bool:
    text = relative.as_posix()
    return text.startswith(KEEP_PREFIXES) and text.endswith(KEEP_SUFFIXES)


def fetch(tag: str = FASTAPI_TAG, out: Path = Path("data/raw"), force: bool = False) -> Path:
    """Download and extract the docs; return the extraction directory."""
    target = raw_dir(out, tag)
    if (target / "docs/en/docs/index.md").exists() and not force:
        print(f"docs already present at {target}")
        return target

    url = TARBALL_URL.format(tag=tag)
    print(f"downloading {url}")
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        response = client.get(url)
        response.raise_for_status()

    count = 0
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            # Strip the "fastapi-<tag>/" directory GitHub puts at the top.
            parts = PurePosixPath(member.name).parts[1:]
            if not parts:
                continue
            relative = PurePosixPath(*parts)
            if ".." in relative.parts or not _wanted(relative):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(extracted.read())
            count += 1

    print(f"extracted {count} files to {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default=FASTAPI_TAG)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    fetch(args.tag, args.out, args.force)


if __name__ == "__main__":
    main()
