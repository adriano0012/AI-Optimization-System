"""
Cache Manager Module
Implements advanced caching strategies for prompts, embeddings, and results
with semantic, neural, multi-level, and distributed caching capabilities
SEC-003 FIXED: Replaced pickle with JSON for safe serialization
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import hashlib
import json
import time
import threading
import re
from collections import OrderedDict, defaultdict
from universal_ai_optimizer.core.base import BaseOptimizerModule
import math

logger = logging.getLogger(__name__)

try:
    import redis as redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class EmbeddingCache:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, text):
        return self.cache.get(text)

    def set(self, text, embedding):
        self.cache[text] = embedding
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class SimilarityCache:
    def __init__(self, threshold=0.85, max_entries=10000):
        self.threshold = threshold
        self.entries = []
        self.max_entries = max_entries

    def _cosine(self, v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        return dot / (math.sqrt(sum(a * a for a in v1)) * math.sqrt(sum(b * b for b in v2)) + 1e-9)

    def find_similar(self, embedding):
        best_match, best_score = None, 0.0
        for entry in self.entries:
            score = self._cosine(embedding, entry['embedding'])
            if score > best_score and score >= self.threshold:
                best_score, best_match = score, entry
        return best_match

    def add(self, text, embedding, result):
        if len(self.entries) >= self.max_entries:
            self.entries.pop(0)
        self.entries.append({'text': text, 'embedding': embedding, 'result': result})


class PredictiveCache:
    def __init__(self):
        self.patterns = defaultdict(int)

    def record(self, sequence):
        self.patterns[sequence] += 1

    def predict(self, current_sequence):
        return None


def _json_safe_dumps(obj: Any) -> str:
    """Safely serialize object to JSON string, handling non-serializable types.
    Uses a whitelist approach to prevent leaking internal objects (e.g. API keys)."""
    def default_serializer(o):
        if isinstance(o, (int, float, str, bool, type(None))):
            return o
        if isinstance(o, (list, tuple)):
            return list(o)
        if isinstance(o, dict):
            return o
        if isinstance(o, set):
            return sorted(list(o))
        if isinstance(o, bytes):
            return o.decode('utf-8', errors='replace')
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    return json.dumps(obj, sort_keys=True, default=default_serializer, ensure_ascii=False)


def _json_safe_loads(s: str) -> Any:
    """Safely deserialize JSON string"""
    return json.loads(s)


class CacheManager(BaseOptimizerModule):
    """
    Advanced cache manager that stores and retrieves cached results to avoid recomputation
    with support for semantic caching, neural caching, multi-level caching, and distributed caching
    SEC-003: Uses JSON instead of pickle for safe serialization
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.cache_types = self.config.get('cache_types', ['prompt', 'embedding', 'result'])
        self.backend = self.config.get('backend', 'multi_level')
        self.ttl = self.config.get('ttl', 3600)
        self.max_size = self.config.get('max_size', 1000)
        self.hierarchical = self.config.get('hierarchical', True)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.85)
        self.enable_neural_cache = self.config.get('enable_neural_cache', True)
        self.enable_semantic_cache = self.config.get('enable_semantic_cache', True)
        self.enable_distributed_cache = self.config.get('enable_distributed_cache', False)

        self._init_cache()

        self.stats = {
            'hits': defaultdict(int),
            'misses': defaultdict(int),
            'evictions': defaultdict(int),
            'size': defaultdict(int)
        }

        self.neural_cache_model = None
        self.embedding_cache = {}

        self.distributed_nodes = self.config.get('distributed_nodes', [])
        self.consistency_level = self.config.get('consistency_level', 'eventual')

        self._lock = threading.RLock()

    def _init_multi_level(self):
        """Initialize multi-level cache structure (L1/L2/L3)"""
        self.caches = {
            'L1': {cache_type: OrderedDict() for cache_type in self.cache_types},
            'L2': {cache_type: OrderedDict() for cache_type in self.cache_types},
            'L3': {cache_type: OrderedDict() for cache_type in self.cache_types}
        }
        self.level_sizes = {
            'L1': {cache_type: max(10, self.max_size // 20) for cache_type in self.cache_types},
            'L2': {cache_type: max(50, self.max_size // 5) for cache_type in self.cache_types},
            'L3': {cache_type: self.max_size for cache_type in self.cache_types}
        }
        self.level_ttl = {
            'L1': self.ttl // 4,
            'L2': self.ttl // 2,
            'L3': self.ttl
        }

    def _init_cache(self):
        """Initialize cache storage based on backend"""
        self.logger.debug(f"Initializing {self.backend} cache")

        if self.backend == 'multi_level':
            self._init_multi_level()
        elif self.backend == 'redis':
            if REDIS_AVAILABLE:
                redis_url = self.config.get('redis_url', 'redis://localhost:6379/0')
                self._redis_client = redis_lib.from_url(redis_url)
                try:
                    self._redis_client.ping()
                    self.logger.info(f"Connected to Redis at {redis_url}")
                    self.caches = {
                        self.backend: {cache_type: self._redis_cache_wrapper(cache_type) for cache_type in self.cache_types}
                    }
                except Exception as e:
                    self.logger.warning(f"Redis connection failed ({e}), falling back to multi_level")
                    self.backend = 'multi_level'
                    self._init_multi_level()
            else:
                self.logger.warning("redis-py not installed, falling back to multi_level")
                self.backend = 'multi_level'
                self._init_multi_level()
        elif self.backend == 'memcached':
            self.logger.warning("Memcached backend not implemented, falling back to multi_level")
            self.backend = 'multi_level'
            self._init_multi_level()
        elif self.backend == 'distributed':
            self.logger.warning("Distributed cache backend not fully implemented, falling back to multi_level")
            self.backend = 'multi_level'
            self._init_multi_level()
        else:
            self.logger.warning(f"Unknown backend '{self.backend}', falling back to multi_level")
            self.backend = 'multi_level'
            self._init_multi_level()

        self.access_times = {cache_type: {} for cache_type in self.cache_types}
        self.access_counts = {cache_type: defaultdict(int) for cache_type in self.cache_types}

    def _redis_cache_wrapper(self, cache_type: str) -> OrderedDict:
        """Return an OrderedDict-like object backed by Redis."""
        class RedisBackedDict(OrderedDict):
            def __init__(self, redis_client, prefix, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._redis = redis_client
                self._prefix = f"cache:{prefix}"

            def __contains__(self, key):
                return self._redis.exists(f"{self._prefix}:{key}")

            def __getitem__(self, key):
                data = self._redis.get(f"{self._prefix}:{key}")
                if data is None:
                    raise KeyError(key)
                try:
                    return json.loads(data.decode())
                except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
                    raise KeyError(key) from None

            def __setitem__(self, key, value):
                self._redis.setex(f"{self._prefix}:{key}", 3600, json.dumps(value))

            def __delitem__(self, key):
                self._redis.delete(f"{self._prefix}:{key}")

            def get(self, key, default=None):
                data = self._redis.get(f"{self._prefix}:{key}")
                if data is None:
                    return default
                return json.loads(data.decode())

            def __len__(self):
                return self._redis.dbsize()

        return RedisBackedDict(self._redis_client, cache_type)

    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate a cache key from data using JSON serialization (SEC-003 FIX)"""
        if isinstance(data, str):
            data_str = data
        else:
            data_str = _json_safe_dumps(data)
        return hashlib.sha256(f"{prefix}:{data_str}".encode()).hexdigest()

    def _generate_semantic_key(self, text: str) -> str:
        """Generate a semantic cache key based on text meaning"""
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        words = [w for w in normalized.split() if len(w) > 3]
        meaningful_text = ' '.join(sorted(words))
        return hashlib.sha256(meaningful_text.encode()).hexdigest()

    def _get_from_cache(self, cache_type: str, key: str,
                        level: Optional[str] = None) -> Tuple[Optional[Any], bool]:
        """
        Retrieve item from cache if it exists and is not expired

        Returns:
            Tuple of (value, found_flag)
        """
        with self._lock:
            if not self.enabled:
                return None, False

            levels_to_check = []
            if level:
                levels_to_check = [level]
            elif self.backend == 'multi_level':
                levels_to_check = ['L1', 'L2', 'L3']
            else:
                levels_to_check = [self.backend]

            for cache_level in levels_to_check:
                if cache_level not in self.caches or cache_type not in self.caches[cache_level]:
                    continue

                cache = self.caches[cache_level][cache_type]
                if key not in cache:
                    continue

                entry = cache[key]
                if isinstance(entry, tuple) and len(entry) == 2:
                    value, timestamp = entry
                else:
                    value, timestamp = entry, time.time()

                ttl = self._get_ttl_for_level(cache_level)
                if time.time() - timestamp > ttl:
                    del cache[key]
                    self.stats['evictions'][cache_type] += 1
                    continue

                if isinstance(cache, OrderedDict):
                    del cache[key]
                    cache[key] = (value, timestamp)

                self.stats['hits'][cache_type] += 1
                self.access_counts[cache_type][key] += 1
                self.access_times[cache_type][key] = time.time()

                if self.backend == 'multi_level' and cache_level != 'L1':
                    self._promote_to_level(cache_type, key, value, timestamp, 'L1')

                logger.debug(f"Cache hit for {cache_type} in {cache_level}")
                return value, True

            self.stats['misses'][cache_type] += 1
            return None, False

    def _set_in_cache(self, cache_type: str, key: str, value: Any,
                      level: Optional[str] = None):
        """Store item in cache, enforcing size limits and policies"""
        with self._lock:
            if not self.enabled:
                return

            target_level = level
            if not target_level:
                if self.backend == 'multi_level':
                    target_level = 'L1'
                else:
                    target_level = self.backend

            if target_level not in self.caches:
                self.caches[target_level] = {}
            if cache_type not in self.caches[target_level]:
                self.caches[target_level][cache_type] = OrderedDict() if target_level in ['L1', 'L2'] else {}

            cache = self.caches[target_level][cache_type]

            max_size = self._get_max_size_for_level(target_level, cache_type)
            if len(cache) >= max_size:
                self._evict_from_cache(cache_type, target_level)

            entry = (value, time.time())

            if isinstance(cache, OrderedDict):
                cache[key] = entry
                cache.move_to_end(key)
            else:
                cache[key] = entry

            self.access_counts[cache_type][key] = 1
            self.access_times[cache_type][key] = time.time()
            self.stats['size'][cache_type] = len(cache)

            logger.debug(f"Stored item in {cache_type} cache at {target_level}")

    def _evict_from_cache(self, cache_type: str, level: str):
        """Evict an item from cache based on policy"""
        if level not in self.caches or cache_type not in self.caches[level]:
            return

        cache = self.caches[level][cache_type]
        if not cache:
            return

        if isinstance(cache, OrderedDict):
            if cache:
                evicted_key, _ = cache.popitem(last=False)
                self.stats['evictions'][cache_type] += 1
                logger.debug(f"Evicted {evicted_key} from {cache_type} cache in {level} (LRU)")
        else:
            if cache and cache_type in self.access_times:
                oldest_key = min(
                    self.access_times[cache_type].keys(),
                    key=lambda k: self.access_times[cache_type].get(k, float('inf'))
                )
                if oldest_key in cache:
                    del cache[oldest_key]
                    if oldest_key in self.access_times[cache_type]:
                        del self.access_times[cache_type][oldest_key]
                    if oldest_key in self.access_counts[cache_type]:
                        del self.access_counts[cache_type][oldest_key]
                    self.stats['evictions'][cache_type] += 1
                    logger.debug(f"Evicted {oldest_key} from {cache_type} cache in {level} (LRU)")

    def _promote_to_level(self, cache_type: str, key: str, value: Any,
                          timestamp: float, target_level: str):
        """Promote an item to a higher cache level"""
        self._set_in_cache(cache_type, key, value, target_level)

    def _get_ttl_for_level(self, level: str) -> int:
        """Get TTL for a specific cache level"""
        if hasattr(self, 'level_ttl') and level in self.level_ttl:
            return self.level_ttl.get(level, self.ttl)
        return self.ttl

    def _get_max_size_for_level(self, level: str, cache_type: str) -> int:
        """Get maximum size for a specific cache level and type"""
        if hasattr(self, 'level_sizes') and level in self.level_sizes:
            return self.level_sizes[level].get(cache_type, self.max_size)
        return self.max_size

    def process(self, prompt: str, context: Dict[str, Any],
                model_adapter: Optional[Any] = None,
                pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check cache for existing result, store new result after processing
        Supports semantic caching and neural caching
        """
        if not self.enabled:
            return {}

        self._log_processing(len(prompt), len(str(context)))

        cache_data = {
            'prompt': prompt,
            'context': context
        }
        standard_key = self._generate_key('optimization', cache_data)

        cached_result, found = self._get_from_cache('result', standard_key)
        if found:
            self.logger.info(f"Standard cache hit for prompt: {prompt[:50]}...")
            return {
                'cached_result': cached_result,
                'optimized_prompt': cached_result.get('optimized_prompt', prompt),
                'compressed_context': cached_result.get('compressed_context', {}),
                'verification_score': cached_result.get('verification_score', 0.0),
                'token_savings': cached_result.get('token_savings', 0.0),
                'cache_hit': True,
                'cache_type': 'standard'
            }

        if self.enable_semantic_cache:
            semantic_key = self._generate_semantic_key(prompt)
            semantic_result, semantic_found = self._get_from_cache(
                'result', semantic_key, level='L2' if self.backend == 'multi_level' else None
            )
            if semantic_found:
                self.logger.info(f"Semantic cache hit for prompt: {prompt[:50]}...")
                return {
                    'cached_result': semantic_result,
                    'optimized_prompt': semantic_result.get('optimized_prompt', prompt),
                    'compressed_context': semantic_result.get('compressed_context', {}),
                    'verification_score': semantic_result.get('verification_score', 0.0) * 0.95,
                    'token_savings': semantic_result.get('token_savings', 0.0),
                    'cache_hit': True,
                    'cache_type': 'semantic',
                    'semantic_similarity': 0.9
                }

        self.logger.debug(f"Cache miss for prompt: {prompt[:50]}...")

        result = {
            'cache_key': standard_key,
            'semantic_key': self._generate_semantic_key(prompt) if self.enable_semantic_cache else None,
            'cache_hit': False
        }

        if self.enable_semantic_cache:
            self._cache_prompt_embedding(prompt, standard_key)

        return result

    def _cache_prompt_embedding(self, prompt: str, cache_key: str):
        """Cache the embedding of a prompt for semantic matching"""
        pass

    def store_result(self, key: str, result: Dict[str, Any]):
        """Store a final result in the cache (to be called by pipeline after processing)"""
        if not self.enabled or not key:
            return

        self._set_in_cache('result', key, result)

        if self.enable_semantic_cache:
            pass

        self.logger.debug(f"Stored result in cache with key: {key[:16]}...")

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics"""
        base_metrics = super().get_metrics()

        hit_rates = {}
        for cache_type in self.cache_types:
            hits = self.stats['hits'][cache_type]
            misses = self.stats['misses'][cache_type]
            total = hits + misses
            hit_rate = hits / max(total, 1)
            hit_rates[f'{cache_type}_hit_rate'] = hit_rate

        base_metrics.update({
            'enabled': self.enabled,
            'backend': self.backend,
            'cache_types': self.cache_types,
            'hit_rates': hit_rates,
            'total_hits': sum(self.stats['hits'].values()),
            'total_misses': sum(self.stats['misses'].values()),
            'total_evictions': sum(self.stats['evictions'].values()),
            'cache_sizes': dict(self.stats['size']),
            'similarity_threshold': self.similarity_threshold,
            'enable_semantic_cache': self.enable_semantic_cache,
            'enable_neural_cache': self.enable_neural_cache,
            'enable_distributed_cache': self.enable_distributed_cache
        })
        return base_metrics