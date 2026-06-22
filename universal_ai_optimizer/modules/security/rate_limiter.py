"""
Rate Limiter Middleware
Token bucket algorithm with per-user and per-organization limits.
Supports sliding window counters for production-grade throttling.
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from enum import Enum


class RateLimitStrategy(str, Enum):
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    reset_at: float
    retry_after: Optional[float] = None

    def to_headers(self) -> Dict[str, str]:
        headers = {
            'X-RateLimit-Limit': str(self.limit),
            'X-RateLimit-Remaining': str(self.remaining),
            'X-RateLimit-Reset': str(int(self.reset_at)),
        }
        if not self.allowed:
            headers['Retry-After'] = str(int(self.retry_after or 1))
        return headers


class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> RateLimitResult:
        with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return RateLimitResult(
                    allowed=True, limit=self.capacity,
                    remaining=int(self.tokens), reset_at=now + 1,
                )
            else:
                wait_time = (tokens - self.tokens) / self.refill_rate
                return RateLimitResult(
                    allowed=False, limit=self.capacity,
                    remaining=0, reset_at=now + wait_time,
                    retry_after=wait_time,
                )


class SlidingWindowCounter:
    """Sliding window rate limiter using request counts."""

    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: list = []
        self._lock = threading.Lock()

    def check(self) -> RateLimitResult:
        now = time.time()
        with self._lock:
            cutoff = now - self.window_seconds
            self._requests = [t for t in self._requests if t > cutoff]
            count = len(self._requests)

            if count < self.limit:
                self._requests.append(now)
                return RateLimitResult(
                    allowed=True, limit=self.limit,
                    remaining=self.limit - count - 1,
                    reset_at=now + self.window_seconds,
                )
            else:
                oldest = self._requests[0] if self._requests else now
                retry_after = oldest + self.window_seconds - now
                return RateLimitResult(
                    allowed=False, limit=self.limit,
                    remaining=0,
                    reset_at=oldest + self.window_seconds,
                    retry_after=max(0, retry_after),
                )


class FixedWindowCounter:
    """Fixed window rate limiter."""

    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self._window_start = time.time()
        self._count = 0
        self._lock = threading.Lock()

    def check(self) -> RateLimitResult:
        now = time.time()
        with self._lock:
            if now - self._window_start >= self.window_seconds:
                self._window_start = now
                self._count = 0

            if self._count < self.limit:
                self._count += 1
                return RateLimitResult(
                    allowed=True, limit=self.limit,
                    remaining=self.limit - self._count,
                    reset_at=self._window_start + self.window_seconds,
                )
            else:
                reset_at = self._window_start + self.window_seconds
                return RateLimitResult(
                    allowed=False, limit=self.limit,
                    remaining=0, reset_at=reset_at,
                    retry_after=reset_at - now,
                )


class RateLimiter:
    """
    Multi-tenant rate limiter.
    Supports per-user, per-org, and global rate limits.
    """

    def __init__(self, strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET):
        self.strategy = strategy
        self._limiters: Dict[str, Any] = {}
        self._default_limits: Dict[str, int] = {
            'global': 10000,
            'per_user': 1000,
            'per_org': 5000,
            'per_endpoint': 100,
        }
        self._lock = threading.Lock()

    def set_default_limit(self, scope: str, limit: int):
        self._default_limits[scope] = limit

    def set_limit(self, key: str, limit: int, window_seconds: int = 60):
        with self._lock:
            if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                self._limiters[key] = TokenBucket(limit, limit / window_seconds)
            elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
                self._limiters[key] = SlidingWindowCounter(limit, window_seconds)
            else:
                self._limiters[key] = FixedWindowCounter(limit, window_seconds)

    def check(self, key: str, scope: str = 'per_user') -> RateLimitResult:
        limit = self._default_limits.get(scope, 1000)

        with self._lock:
            if key not in self._limiters:
                if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                    self._limiters[key] = TokenBucket(limit, limit / 60)
                elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
                    self._limiters[key] = SlidingWindowCounter(limit, 60)
                else:
                    self._limiters[key] = FixedWindowCounter(limit, 60)

        limiter = self._limiters[key]

        if isinstance(limiter, TokenBucket):
            return limiter.consume()
        else:
            return limiter.check()

    def allow(self, user_id: str) -> bool:
        result = self.check(user_id, 'per_user')
        return result.allowed

    def get_stats(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy.value,
            'active_limiters': len(self._limiters),
            'default_limits': self._default_limits.copy(),
        }

    def clear(self, key: Optional[str] = None):
        with self._lock:
            if key:
                self._limiters.pop(key, None)
            else:
                self._limiters.clear()
