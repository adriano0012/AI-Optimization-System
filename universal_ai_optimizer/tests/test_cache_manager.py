import pytest
from collections import OrderedDict
from universal_ai_optimizer.modules.cache_manager import (
    EmbeddingCache, SimilarityCache, PredictiveCache,
    CacheManager, _json_safe_dumps, _json_safe_loads
)


class TestEmbeddingCache:
    def test_init_default(self):
        cache = EmbeddingCache()
        assert cache.capacity == 1000

    def test_init_custom_capacity(self):
        cache = EmbeddingCache(capacity=10)
        assert cache.capacity == 10

    def test_get_empty(self):
        cache = EmbeddingCache()
        assert cache.get("text") is None

    def test_set_and_get(self):
        cache = EmbeddingCache()
        cache.set("hello", [1.0, 2.0])
        assert cache.get("hello") == [1.0, 2.0]

    def test_overwrite_key(self):
        cache = EmbeddingCache()
        cache.set("k", [1.0])
        cache.set("k", [2.0])
        assert cache.get("k") == [2.0]

    def test_eviction(self):
        cache = EmbeddingCache(capacity=2)
        cache.set("a", [1])
        cache.set("b", [2])
        cache.set("c", [3])
        assert cache.get("a") is None
        assert cache.get("b") == [2]
        assert cache.get("c") == [3]

    def test_capacity_one(self):
        cache = EmbeddingCache(capacity=1)
        cache.set("a", [1])
        assert cache.get("a") == [1]
        cache.set("b", [2])
        assert cache.get("a") is None


class TestSimilarityCache:
    def test_init_default(self):
        cache = SimilarityCache()
        assert cache.threshold == 0.85

    def test_add_and_find_exact(self):
        cache = SimilarityCache(threshold=0.5)
        emb = [1.0, 0.0, 0.0]
        cache.add("text1", emb, "result1")
        found = cache.find_similar([1.0, 0.0, 0.0])
        assert found is not None
        assert found["text"] == "text1"

    def test_find_similar_empty(self):
        cache = SimilarityCache()
        assert cache.find_similar([1.0, 0.0]) is None

    def test_find_similar_below_threshold(self):
        cache = SimilarityCache(threshold=0.99)
        cache.add("a", [1.0, 0.0], "r1")
        assert cache.find_similar([0.0, 1.0]) is None

    def test_fifo_eviction(self):
        cache = SimilarityCache(max_entries=2)
        cache.add("a", [1.0, 0.0], "r1")
        cache.add("b", [0.0, 1.0], "r2")
        cache.add("c", [1.0, 0.0], "r3")
        found = cache.find_similar([1.0, 0.0])
        assert found["text"] == "c"


class TestPredictiveCache:
    def test_record(self):
        cache = PredictiveCache()
        cache.record("seq1")
        cache.record("seq1")
        assert cache.patterns["seq1"] == 2

    def test_predict_returns_none(self):
        cache = PredictiveCache()
        assert cache.predict("anything") is None


class TestJsonSafe:
    def test_primitives(self):
        assert _json_safe_dumps(42) == "42"
        assert _json_safe_dumps("hi") == '"hi"'
        assert _json_safe_dumps(None) == "null"

    def test_set(self):
        result = _json_safe_loads(_json_safe_dumps({1, 2, 3}))
        assert sorted(result) == [1, 2, 3]

    def test_bytes(self):
        result = _json_safe_dumps(b"hello")
        assert "hello" in result

    def test_nested(self):
        data = {"a": [1, 2], "b": {"c": 3}}
        assert _json_safe_loads(_json_safe_dumps(data)) == data

    def test_unserializable_raises(self):
        class Foo:
            pass
        with pytest.raises(TypeError):
            _json_safe_dumps(Foo())


class TestCacheManager:
    def test_init_defaults(self):
        cm = CacheManager()
        assert cm.config is not None

    def test_disabled(self):
        cm = CacheManager({"enabled": False})
        result = cm.process("test", {})
        assert result == {}

    def test_store_result(self):
        cm = CacheManager()
        cm.store_result("key1", "result1")
        metrics = cm.get_metrics()
        assert "total_hits" in metrics

    def test_process_returns_dict(self):
        cm = CacheManager()
        result = cm.process("hello world", {})
        assert isinstance(result, dict)

    def test_metrics(self):
        cm = CacheManager()
        metrics = cm.get_metrics()
        assert "total_hits" in metrics
        assert "total_misses" in metrics
        assert "hit_rates" in metrics
        assert "enabled" in metrics
        assert "backend" in metrics

    def test_unknown_backend_fallback(self):
        cm = CacheManager({"backend": "unknown_backend"})
        assert cm.process("test", {}) is not None
