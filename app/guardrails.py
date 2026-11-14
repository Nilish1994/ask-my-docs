"""In-memory cost guardrails: per-client sliding-window limits and a daily cap.

State lives in the API process, which is right for the single-container demo
this is built for. Behind several workers or replicas it would move to Redis.
All methods are synchronous and do no awaiting, so under asyncio each
check-and-consume is atomic without locks.
"""

from __future__ import annotations

import math
import time
from collections import deque
from datetime import UTC, datetime

from fastapi import Request


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float, clock=time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self.clock = clock
        self._hits: dict[str, deque[float]] = {}

    def _prune(self, key: str) -> deque[float]:
        now = self.clock()
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        return hits

    def retry_after(self, key: str) -> int:
        """Seconds until ``key`` may make another request (0 = allowed now)."""
        hits = self._prune(key)
        if len(hits) < self.limit:
            return 0
        return max(1, math.ceil(self.window - (self.clock() - hits[0])))

    def consume(self, key: str) -> None:
        self._prune(key).append(self.clock())

    def remaining(self, key: str) -> int:
        return max(0, self.limit - len(self._prune(key)))

    def sweep(self) -> None:
        for key in list(self._hits):
            if not self._prune(key):
                del self._hits[key]


class DailyCap:
    """Global questions-per-day budget; resets at 00:00 UTC."""

    def __init__(self, limit: int, today=lambda: datetime.now(UTC).date()):
        self.limit = limit
        self.today = today
        self._day = today()
        self._count = 0

    def _roll(self) -> None:
        day = self.today()
        if day != self._day:
            self._day, self._count = day, 0

    def exhausted(self) -> bool:
        self._roll()
        return self._count >= self.limit

    def consume(self) -> None:
        self._roll()
        self._count += 1

    @property
    def used(self) -> int:
        self._roll()
        return self._count


def client_ip(request: Request, forwarded_hops: int) -> str:
    """Client address, trusting only the X-Forwarded-For entries our proxies add.

    Each trusted proxy appends the address it received the request from, so the
    real client is ``forwarded_hops`` entries from the right. Entries further
    left are client-controlled and ignored.
    """
    peer = request.client.host if request.client else "unknown"
    if forwarded_hops <= 0:
        return peer
    header = request.headers.get("x-forwarded-for", "")
    entries = [entry.strip() for entry in header.split(",") if entry.strip()]
    if len(entries) >= forwarded_hops:
        return entries[-forwarded_hops]
    return entries[0] if entries else peer
