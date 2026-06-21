"""Rate Limiter - Token Bucket implementation with thread safety"""
import hashlib
import logging
import threading
import time
import json
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)

try:
    import redis as redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""
    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time"""
        now = time.monotonic()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.
        Returns True if successful, False if not enough tokens.
        """
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_tokens(self) -> float:
        """Get current available tokens"""
        with self._lock:
            self._refill()
            return self.tokens

    def get_wait_time(self, tokens: int = 1) -> float:
        """Get estimated wait time for tokens to become available"""
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                return 0.0
            return (tokens - self.tokens) / self.refill_rate


class RateLimiter(BaseOptimizerModule):
    """
    Thread-safe rate limiter using Token Bucket algorithm.
    Supports per-user/context rate limiting with configurable limits.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        default_capacity: int = 100,
        default_refill_rate: float = 10.0,
        max_buckets: int = 10000,
        cleanup_interval: float = 300.0,
        redis_url: Optional[str] = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            config: Optional configuration dictionary
            default_capacity: Maximum tokens in bucket (burst allowance)
            default_refill_rate: Tokens added per second
            max_buckets: Maximum number of buckets to track
            cleanup_interval: Seconds between cleanup of inactive buckets
            redis_url: Optional Redis URL for distributed rate limiting
        """
        super().__init__(config)
        self.default_capacity = default_capacity
        self.default_refill_rate = default_refill_rate
        self.max_buckets = max_buckets
        self.cleanup_interval = cleanup_interval

        self._buckets: Dict[str, TokenBucket] = {}
        self._bucket_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.RLock()
        self._last_cleanup = time.monotonic()
        self._custom_limits: Dict[str, Tuple[int, float]] = {}

        self._redis_client = None
        if redis_url and REDIS_AVAILABLE:
            try:
                self._redis_client = redis_lib.from_url(redis_url)
                self._redis_client.ping()
                self._redis_lua_script = self._redis_client.register_script("""
                    local key = KEYS[1]
                    local capacity = tonumber(ARGV[1])
                    local refill_rate = tonumber(ARGV[2])
                    local tokens_needed = tonumber(ARGV[3])
                    local now = tonumber(ARGV[4])
                    local data = redis.call('get', key)
                    if not data then
                        local tokens = capacity - tokens_needed
                        if tokens >= 0 then
                            redis.call('setex', key, 86400, tostring(tokens) .. ':' .. tostring(now))
                            return 1
                        end
                        redis.call('setex', key, 86400, tostring(capacity) .. ':' .. tostring(now))
                        return 0
                    end
                    local parts = {}
                    for w in string.gmatch(data, '([^:]+)') do
                        table.insert(parts, w)
                    end
                    local tokens = tonumber(parts[1])
                    local last_refill = tonumber(parts[2])
                    local elapsed = now - last_refill
                    tokens = math.min(capacity, tokens + elapsed * refill_rate)
                    if tokens >= tokens_needed then
                        tokens = tokens - tokens_needed
                        redis.call('setex', key, 86400, tostring(tokens) .. ':' .. tostring(now))
                        return 1
                    end
                    redis.call('setex', key, 86400, tostring(tokens) .. ':' .. tostring(now))
                    return 0
                """)
                logger.info("Redis rate limiter connected successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed, falling back to local rate limiter: {e}")
                self._redis_client = None
        elif redis_url and not REDIS_AVAILABLE:
            self.logger.warning("Redis URL provided but redis library not available, using local rate limiter (inconsistent across instances)")

    def process(self, prompt: str, context: Dict[str, Any],
                model_adapter: Optional[Any] = None,
                pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input by applying rate limiting.
        Uses 'rate_limit_key' from context if present, otherwise uses prompt hash.
        """
        self._log_processing(len(prompt), len(context) if context else 0)
        key = (context or {}).get('rate_limit_key', hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16])
        allowed, details = self.allow_with_details(key)
        return {
            'allowed': allowed,
            'rate_limit_details': details,
            'module': self.__class__.__name__
        }

    def _get_or_create_bucket(self, key: str) -> TokenBucket:
        """Get or create a token bucket for the given key"""
        with self._global_lock:
            if key in self._buckets:
                return self._buckets[key]

            if len(self._buckets) >= self.max_buckets:
                self._cleanup_inactive_buckets()

            capacity, refill_rate = self._custom_limits.get(key, (self.default_capacity, self.default_refill_rate))
            bucket = TokenBucket(capacity=capacity, refill_rate=refill_rate)
            self._buckets[key] = bucket
            self._bucket_locks[key] = threading.Lock()
            return bucket

    def _cleanup_inactive_buckets(self) -> None:
        """Remove buckets that haven't been used recently"""
        now = time.monotonic()
        if now - self._last_cleanup < self.cleanup_interval:
            return

        inactive_keys = []
        for key, bucket in self._buckets.items():
            with bucket._lock:
                if now - bucket.last_refill > self.cleanup_interval * 2:
                    inactive_keys.append(key)

        for key in inactive_keys:
            del self._buckets[key]
            if key in self._bucket_locks:
                del self._bucket_locks[key]

        self._last_cleanup = now

    def allow(self, key: str, tokens: int = 1) -> bool:
        if not isinstance(key, str) or not key:
            return False
        if self._redis_client:
            capacity, refill_rate = self._custom_limits.get(key, (self.default_capacity, self.default_refill_rate))
            try:
                result = self._redis_lua_script(
                    keys=[f"rl:{key}"],
                    args=[capacity, refill_rate, tokens, time.time()]
                )
                return bool(result)
            except Exception:
                pass
        bucket = self._get_or_create_bucket(key)
        return bucket.consume(tokens)

    def allow_with_details(self, key: str, tokens: int = 1) -> Tuple[bool, Dict]:
        """
        Check if request is allowed with detailed response.
        
        Returns:
            Tuple of (allowed, info_dict) where info_dict contains:
            - allowed, remaining_tokens, retry_after
        """
        if not isinstance(key, str) or not key:
            return False, {'allowed': False, 'reason': 'invalid_key', 'remaining_tokens': 0, 'retry_after': 0}

        bucket = self._get_or_create_bucket(key)
        allowed = bucket.consume(tokens)
        remaining = bucket.get_tokens()
        
        info = {
            'allowed': allowed,
            'remaining_tokens': remaining,
            'retry_after': 0.0 if allowed else bucket.get_wait_time(tokens)
        }
        
        if not allowed:
            info['reason'] = 'rate_limited'
            
        return allowed, info

    def set_limit(self, key: str, capacity: int, refill_rate: float) -> None:
        """
        Set custom rate limit for a specific key.
        Takes effect on next bucket creation.
        """
        if capacity <= 0 or refill_rate <= 0:
            raise ValueError("Capacity and refill_rate must be positive")
        with self._global_lock:
            self._custom_limits[key] = (capacity, refill_rate)
            if key in self._buckets:
                del self._buckets[key]
                if key in self._bucket_locks:
                    del self._bucket_locks[key]

    def remove_limit(self, key: str) -> None:
        """Remove custom limit for a key, reverting to defaults"""
        with self._global_lock:
            self._custom_limits.pop(key, None)
            if key in self._buckets:
                del self._buckets[key]
            if key in self._bucket_locks:
                del self._bucket_locks[key]

    def get_status(self, key: str) -> Dict:
        """Get current status for a key without consuming tokens"""
        bucket = self._get_or_create_bucket(key)
        current_tokens = bucket.get_tokens()
        return {
            'key': key,
            'capacity': bucket.capacity,
            'refill_rate': bucket.refill_rate,
            'current_tokens': current_tokens,
            'available': current_tokens >= 1
        }

    def reset(self, key: str) -> None:
        """Reset bucket for a key (refill to full capacity)"""
        with self._global_lock:
            if key in self._buckets:
                bucket = self._buckets[key]
                with bucket._lock:
                    bucket.tokens = float(bucket.capacity)
                    bucket.last_refill = time.monotonic()

    def reset_all(self) -> None:
        """Reset all buckets"""
        with self._global_lock:
            for bucket in self._buckets.values():
                with bucket._lock:
                    bucket.tokens = float(bucket.capacity)
                    bucket.last_refill = time.monotonic()

    def get_metrics(self) -> Dict:
        """Get global metrics"""
        with self._global_lock:
            total_buckets = len(self._buckets)
            active_buckets = 0
            total_tokens = 0.0
            
            for bucket in self._buckets.values():
                with bucket._lock:
                    if bucket.tokens < bucket.capacity:
                        active_buckets += 1
                    total_tokens += bucket.tokens
            
            base_metrics = super().get_metrics()
            base_metrics.update({
                'total_buckets': total_buckets,
                'active_buckets': active_buckets,
                'total_available_tokens': total_tokens,
                'max_buckets': self.max_buckets,
                'default_capacity': self.default_capacity,
                'default_refill_rate': self.default_refill_rate
            })
            return base_metrics


class SlidingWindowRateLimiter:
    """
    Alternative rate limiter using sliding window log algorithm.
    More accurate for burst handling but uses more memory.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: float = 60.0,
        max_keys: int = 10000
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._windows: Dict[str, list] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.RLock()

    def _get_window(self, key: str) -> list:
        with self._global_lock:
            if key not in self._windows:
                if len(self._windows) >= self.max_keys:
                    self._cleanup_old_windows()
                self._windows[key] = []
                self._locks[key] = threading.Lock()
            return self._windows[key]

    def _cleanup_old_windows(self) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        to_delete = []
        for key, window in self._windows.items():
            with self._locks[key]:
                while window and window[0] < cutoff:
                    window.pop(0)
                if not window:
                    to_delete.append(key)
        for key in to_delete:
            del self._windows[key]
            del self._locks[key]

    def allow(self, key: str) -> bool:
        if not isinstance(key, str) or not key:
            return False

        window = self._get_window(key)
        lock = self._locks[key]
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with lock:
            while window and window[0] < cutoff:
                window.pop(0)
            if len(window) < self.max_requests:
                window.append(now)
                return True
            return False

    def get_remaining(self, key: str) -> int:
        window = self._get_window(key)
        lock = self._locks[key]
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with lock:
            while window and window[0] < cutoff:
                window.pop(0)
            return max(0, self.max_requests - len(window))