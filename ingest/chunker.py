"""Pack heading-delimited sections into retrieval chunks of ~400-800 tokens.

Rules, in order:
1. Never cross a page boundary.
2. Merge consecutive small sections until a chunk reaches ``min_tokens``
   (without exceeding ``max_tokens``). A chunk keeps the anchor of its first
   section, plus the character offset of every section inside it, so a
   citation can later link to the most specific heading it falls under.
3. Split a section longer than ``max_tokens`` at paragraph / code-block
   boundaries into windows of about ``target_tokens``, repeating roughly
   ``overlap_tokens`` of trailing text at the start of the next window.

Token counts are estimated with a word/punctuation regex, which tracks
BPE/WordPiece counts closely enough for sizing (and needs no tokenizer).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .markdown import FENCE_RE, Section

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def estimate_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


@dataclass
class ChunkConfig:
    min_tokens: int = 400
    target_tokens: int = 600
    max_tokens: int = 800
    overlap_tokens: int = 80


@dataclass
class _Draft:
    parts: list[str] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)
    heading_path: list[str] = field(default_factory=list)
    anchor: str = ""
    tokens: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(self.parts)

    def add(self, section: Section, body: str, tokens: int) -> None:
        if not self.parts:
            self.heading_path = section.heading_path
            self.anchor = section.anchor
        start = len(self.text) + (2 if self.parts else 0)
        self.sections.append({"start": start, "anchor": section.anchor, "heading": section.heading})
        self.parts.append(body)
        self.tokens += tokens


def _blocks(body: str) -> list[str]:
    """Split text at blank lines, keeping each fenced code block whole."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in body.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _hard_split(block: str, limit: int) -> list[str]:
    """Last resort for a single block (usually code) longer than ``limit``."""
    pieces: list[str] = []
    current: list[str] = []
    tokens = 0
    for line in block.splitlines():
        line_tokens = estimate_tokens(line)
        if current and tokens + line_tokens > limit:
            pieces.append("\n".join(current))
            current, tokens = [], 0
        current.append(line)
        tokens += line_tokens
    if current:
        pieces.append("\n".join(current))
    return pieces


def split_long(body: str, config: ChunkConfig) -> list[str]:
    """Window a long section into overlapping pieces of ~target_tokens."""
    blocks: list[str] = []
    for block in _blocks(body):
        if estimate_tokens(block) > config.max_tokens:
            blocks.extend(_hard_split(block, config.target_tokens))
        else:
            blocks.append(block)

    windows: list[str] = []
    current: list[str] = []
    tokens = 0
    for block in blocks:
        block_tokens = estimate_tokens(block)
        if current and tokens + block_tokens > config.target_tokens:
            windows.append("\n\n".join(current))
            # Carry trailing blocks forward as overlap.
            overlap: list[str] = []
            overlap_tokens = 0
            for previous in reversed(current):
                previous_tokens = estimate_tokens(previous)
                if overlap_tokens + previous_tokens > config.overlap_tokens:
                    break
                overlap.insert(0, previous)
                overlap_tokens += previous_tokens
            current, tokens = overlap, overlap_tokens
        current.append(block)
        tokens += block_tokens
    if current:
        windows.append("\n\n".join(current))
    return windows


def chunk_sections(sections: list[Section], config: ChunkConfig | None = None) -> list[dict]:
    """Return chunk dicts with text, heading_path, anchor and section offsets."""
    config = config or ChunkConfig()
    chunks: list[_Draft] = []
    draft = _Draft()

    def flush() -> None:
        nonlocal draft
        if draft.parts and draft.text.strip():
            chunks.append(draft)
        draft = _Draft()

    for section in sections:
        body = section.body
        if not body:
            continue
        tokens = estimate_tokens(body)
        if tokens > config.max_tokens:
            flush()
            for window in split_long(body, config):
                piece = _Draft()
                # Windows after the first restate the heading for context.
                if not window.startswith("#") and section.heading:
                    window = f"{'#' * max(section.level, 1)} {section.heading} (continued)\n\n{window}"
                piece.add(section, window, estimate_tokens(window))
                chunks.append(piece)
            continue
        if draft.parts and (draft.tokens >= config.min_tokens or draft.tokens + tokens > config.max_tokens):
            flush()
        draft.add(section, body, tokens)
    flush()

    return [
        {
            "text": d.text,
            "heading_path": d.heading_path,
            "anchor": d.anchor,
            "sections": d.sections,
            "tokens": d.tokens,
        }
        for d in chunks
    ]
