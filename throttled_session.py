"""
throttled_session.py

A drop-in replacement for requests.Session that:
  * enforces a client-side maximum request rate
  * honors Retry-After on 429 (and 503) responses
  * retries with exponential backoff + jitter on 5xx / connection errors
  * slows its request rate after errors or slow responses, then gradually
    speeds back up to the configured maximum as things recover

Thread-safe: share one instance across threads and the limit applies to all of them.

Usage:
    session = ThrottledSession(max_rate=2, slow_threshold=3)
    resp = session.get("https://api.example.com/items", params={"page": 1})
"""
from __future__ import annotations

import email.utils
import logging
import random
import threading
import time
from datetime import datetime, timezone

import requests

log = logging.getLogger(__name__)

IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE", "TRACE"})
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
TRANSIENT_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header (delta-seconds or HTTP-date) into seconds from now."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


class ThrottledSession(requests.Session):
    """
    Args:
        max_rate:        Ceiling on requests per second. The session never goes faster.
        slow_threshold:  Responses taking longer than this many seconds trigger a slowdown.
                         Keep it below your read timeout or it can never fire.
        max_retries:     Retries per request for 429 / 5xx / connection errors.
        backoff_base:    First backoff delay in seconds (doubles on each retry).
        backoff_max:     Cap on any single backoff delay.
        slowdown_factor: The gap between requests is multiplied by this on trouble.
        recovery_factor: The gap is multiplied by this after each healthy response,
                         until it's back to 1/max_rate.
        max_interval:    The slowest the limiter will throttle to (seconds between requests).
        max_retry_after: If the server asks for a longer wait than this, give up and
                         return the response instead of sleeping.
        default_timeout: Used when the caller doesn't pass timeout= (requests has no default).
        retry_methods:   Methods retried on 5xx / connection errors. 429s are retried for
                         any method, since the server rejected the request without acting on it.
    """

    def __init__(
        self,
        max_rate: float = 5.0,
        slow_threshold: float = 5.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
        slowdown_factor: float = 2.0,
        recovery_factor: float = 0.8,
        max_interval: float = 10.0,
        max_retry_after: float = 300.0,
        default_timeout: float | tuple[float, float] | None = (5, 30),
        retry_methods: frozenset[str] = IDEMPOTENT_METHODS,
    ):
        super().__init__()
        if max_rate <= 0:
            raise ValueError("max_rate must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self.base_interval = 1.0 / max_rate
        self.interval = self.base_interval
        self.slow_threshold = slow_threshold
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.slowdown_factor = slowdown_factor
        self.recovery_factor = recovery_factor
        self.max_interval = max(max_interval, self.base_interval)
        self.max_retry_after = max_retry_after
        self.default_timeout = default_timeout
        self.retry_methods = frozenset(m.upper() for m in retry_methods)

        self._lock = threading.Lock()
        self._next_slot = 0.0      # monotonic time the next request may start
        self._paused_until = 0.0   # monotonic time a server-requested pause ends

    @property
    def current_rate(self) -> float:
        """Current effective requests/second after any adaptive slowdown."""
        return 1.0 / self.interval

    # ---- rate control ---------------------------------------------------

    def _wait_for_slot(self) -> None:
        # Reserve a slot under the lock, then sleep outside it so threads queue up
        # in evenly spaced slots instead of all waking at once.
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_slot, self._paused_until)
            self._next_slot = start + self.interval
        if start > now:
            time.sleep(start - now)

    def _slow_down(self, reason: str) -> None:
        with self._lock:
            old = self.interval
            self.interval = min(self.interval * self.slowdown_factor, self.max_interval)
        log.warning("Slowing down (%s): %.2f -> %.2f req/s", reason, 1 / old, 1 / self.interval)

    def _speed_up(self) -> None:
        with self._lock:
            self.interval = max(self.interval * self.recovery_factor, self.base_interval)

    def _pause_all(self, seconds: float) -> None:
        """Block every thread using this session until `seconds` from now."""
        with self._lock:
            self._paused_until = max(self._paused_until, time.monotonic() + seconds)

    def _backoff_delay(self, attempt: int) -> float:
        # Exponential backoff with "equal jitter": half fixed, half random.
        cap = min(self.backoff_max, self.backoff_base * (2 ** attempt))
        return cap / 2 + random.uniform(0, cap / 2)

    # ---- the request loop -----------------------------------------------

    def request(self, method, url, *args, **kwargs):
        if self.default_timeout is not None:
            kwargs.setdefault("timeout", self.default_timeout)
        retry_errors = method.upper() in self.retry_methods

        for attempt in range(self.max_retries + 1):
            last_try = attempt == self.max_retries
            self._wait_for_slot()
            started = time.monotonic()

            try:
                resp = super().request(method, url, *args, **kwargs)
            except TRANSIENT_ERRORS as exc:
                self._slow_down(type(exc).__name__)
                # A connect timeout means the request never left, so it's safe for any method.
                safe = retry_errors or isinstance(exc, requests.exceptions.ConnectTimeout)
                if last_try or not safe:
                    raise
                delay = self._backoff_delay(attempt)
                log.warning("%s on %s %s; retry %d/%d in %.1fs",
                            type(exc).__name__, method, url, attempt + 1, self.max_retries, delay)
                time.sleep(delay)
                continue

            elapsed = time.monotonic() - started
            status = resp.status_code

            # Normal response: adjust speed based on how long it took.
            if status not in RETRYABLE_STATUSES:
                if elapsed > self.slow_threshold:
                    self._slow_down(f"slow response {elapsed:.1f}s")
                else:
                    self._speed_up()
                return resp

            # 429 or 5xx.
            self._slow_down(f"HTTP {status}")
            if last_try or not (status == 429 or retry_errors):
                return resp

            retry_after = parse_retry_after(resp.headers.get("Retry-After"))
            if retry_after is not None and retry_after > self.max_retry_after:
                log.error("%s asked us to wait %.0fs (> max_retry_after); giving up", url, retry_after)
                return resp
            resp.close()  # release the connection back to the pool

            if retry_after is not None or status == 429:
                # The server is telling this client to back off: pause every thread.
                wait = retry_after if retry_after is not None else self._backoff_delay(attempt)
                log.warning("HTTP %d from %s; pausing all requests %.1fs (retry %d/%d)",
                            status, url, wait, attempt + 1, self.max_retries)
                self._pause_all(wait)
            else:
                delay = self._backoff_delay(attempt)
                log.warning("HTTP %d from %s; retry %d/%d in %.1fs",
                            status, url, attempt + 1, self.max_retries, delay)
                time.sleep(delay)

        raise AssertionError("unreachable")  # loop always returns or raises on last_try
