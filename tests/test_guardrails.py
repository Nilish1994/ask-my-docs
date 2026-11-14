from datetime import date
from types import SimpleNamespace

from app.guardrails import DailyCap, SlidingWindowLimiter, client_ip


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_sliding_window_limits_and_recovers():
    clock = Clock()
    limiter = SlidingWindowLimiter(limit=2, window_seconds=3600, clock=clock)
    for _ in range(2):
        assert limiter.retry_after("a") == 0
        limiter.consume("a")
    assert limiter.retry_after("a") == 3600
    assert limiter.retry_after("b") == 0, "limits are per client"
    clock.now += 1800
    assert limiter.retry_after("a") == 1800
    clock.now += 1800
    assert limiter.retry_after("a") == 0
    limiter.sweep()


def test_daily_cap_resets_on_a_new_day():
    day = {"value": date(2026, 11, 14)}
    cap = DailyCap(limit=2, today=lambda: day["value"])
    cap.consume()
    cap.consume()
    assert cap.exhausted()
    day["value"] = date(2026, 11, 15)
    assert not cap.exhausted()
    assert cap.used == 0


def request(peer: str, forwarded: str | None = None):
    headers = {"x-forwarded-for": forwarded} if forwarded else {}
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers=headers)


def test_client_ip_ignores_forwarded_header_without_trusted_proxies():
    assert client_ip(request("10.0.0.5", "192.0.2.44"), forwarded_hops=0) == "10.0.0.5"


def test_client_ip_trusts_only_proxy_appended_entries():
    spoofed = request("172.18.0.3", "192.0.2.66, 203.0.113.9")
    assert client_ip(spoofed, forwarded_hops=1) == "203.0.113.9"
    two_proxies = request("172.18.0.4", "192.0.2.66, 203.0.113.9, 172.18.0.2")
    assert client_ip(two_proxies, forwarded_hops=2) == "203.0.113.9"
