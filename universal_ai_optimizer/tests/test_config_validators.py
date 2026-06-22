"""
Tests for Pydantic Config Validators
"""

import pytest
from api.config_validators import (
    OptimizerConfig, ContextCompressionConfig, MemoryConfig,
    CacheConfig, ExecutionConfig, VerificationConfig, ResponseConfig,
    LoggingConfig, MetricsConfig, MonitoringConfig, TelemetryConfig,
    QuantizationType, VectorDBType, CacheBackend, Environment,
)


class TestContextCompressionConfig:
    def test_defaults(self):
        cfg = ContextCompressionConfig()
        assert cfg.enabled is True
        assert cfg.compression_ratio == 0.3
        assert cfg.max_context_length == 4096

    def test_compression_ratio_bounds(self):
        ContextCompressionConfig(compression_ratio=0.0)
        ContextCompressionConfig(compression_ratio=1.0)
        with pytest.raises(Exception):
            ContextCompressionConfig(compression_ratio=-0.1)
        with pytest.raises(Exception):
            ContextCompressionConfig(compression_ratio=1.1)

    def test_max_context_length_bounds(self):
        ContextCompressionConfig(max_context_length=256)
        with pytest.raises(Exception):
            ContextCompressionConfig(max_context_length=100)

    def test_invalid_method(self):
        with pytest.raises(Exception):
            ContextCompressionConfig(methods=['invalid_method'])


class TestMemoryConfig:
    def test_defaults(self):
        cfg = MemoryConfig()
        assert cfg.vector_db == VectorDBType.FAISS
        assert cfg.max_memories == 10000

    def test_vector_db_enum(self):
        cfg = MemoryConfig(vector_db=VectorDBType.PINECONE)
        assert cfg.vector_db == VectorDBType.PINECONE

    def test_memory_threshold_bounds(self):
        MemoryConfig(memory_threshold=0.0)
        MemoryConfig(memory_threshold=1.0)
        with pytest.raises(Exception):
            MemoryConfig(memory_threshold=2.0)

    def test_max_memories_bounds(self):
        with pytest.raises(Exception):
            MemoryConfig(max_memories=0)


class TestCacheConfig:
    def test_defaults(self):
        cfg = CacheConfig()
        assert cfg.backend == CacheBackend.MULTI_LEVEL
        assert cfg.ttl == 3600

    def test_backend_enum(self):
        cfg = CacheConfig(backend=CacheBackend.REDIS)
        assert cfg.backend == CacheBackend.REDIS

    def test_ttl_bounds(self):
        CacheConfig(ttl=1)
        CacheConfig(ttl=86400)
        with pytest.raises(Exception):
            CacheConfig(ttl=0)

    def test_max_size_bounds(self):
        with pytest.raises(Exception):
            CacheConfig(max_size=0)


class TestExecutionConfig:
    def test_defaults(self):
        cfg = ExecutionConfig()
        assert cfg.quantization == QuantizationType.INT8
        assert cfg.batch_size == 1

    def test_quantization_all_values(self):
        for qt in QuantizationType:
            cfg = ExecutionConfig(quantization=qt)
            assert cfg.quantization == qt

    def test_batch_size_bounds(self):
        ExecutionConfig(batch_size=1)
        ExecutionConfig(batch_size=256)
        with pytest.raises(Exception):
            ExecutionConfig(batch_size=0)
        with pytest.raises(Exception):
            ExecutionConfig(batch_size=257)

    def test_max_sequence_length_bounds(self):
        with pytest.raises(Exception):
            ExecutionConfig(max_sequence_length=64)


class TestVerificationConfig:
    def test_defaults(self):
        cfg = VerificationConfig()
        assert cfg.threshold == 0.95
        assert cfg.max_iterations == 3

    def test_threshold_bounds(self):
        VerificationConfig(threshold=0.0)
        VerificationConfig(threshold=1.0)
        with pytest.raises(Exception):
            VerificationConfig(threshold=1.5)

    def test_max_iterations_bounds(self):
        with pytest.raises(Exception):
            VerificationConfig(max_iterations=0)
        with pytest.raises(Exception):
            VerificationConfig(max_iterations=11)


class TestResponseConfig:
    def test_defaults(self):
        cfg = ResponseConfig()
        assert cfg.latency_target_ms == 1000.0

    def test_latency_bounds(self):
        ResponseConfig(latency_target_ms=1.0)
        ResponseConfig(latency_target_ms=60000.0)
        with pytest.raises(Exception):
            ResponseConfig(latency_target_ms=0.0)


class TestLoggingConfig:
    def test_defaults(self):
        cfg = LoggingConfig()
        assert cfg.log_level == "INFO"
        assert cfg.pii_redaction is True

    def test_log_level_case_insensitive(self):
        cfg = LoggingConfig(log_level="debug")
        assert cfg.log_level == "DEBUG"

    def test_invalid_log_level(self):
        with pytest.raises(Exception):
            LoggingConfig(log_level="INVALID")


class TestMetricsConfig:
    def test_defaults(self):
        cfg = MetricsConfig()
        assert cfg.prometheus_port == 8000

    def test_port_bounds(self):
        with pytest.raises(Exception):
            MetricsConfig(prometheus_port=100)
        with pytest.raises(Exception):
            MetricsConfig(prometheus_port=70000)


class TestMonitoringConfig:
    def test_defaults(self):
        cfg = MonitoringConfig()
        assert cfg.health_check_interval == 30.0

    def test_interval_bounds(self):
        with pytest.raises(Exception):
            MonitoringConfig(health_check_interval=0.0)


class TestTelemetryConfig:
    def test_defaults(self):
        cfg = TelemetryConfig()
        assert cfg.enabled is False
        assert cfg.environment == Environment.DEVELOPMENT

    def test_all_environments(self):
        for env in Environment:
            cfg = TelemetryConfig(environment=env)
            assert cfg.environment == env


class TestOptimizerConfig:
    def test_defaults(self):
        cfg = OptimizerConfig()
        assert cfg.debug is False
        assert cfg.log_level == "INFO"
        assert isinstance(cfg.context_compression, ContextCompressionConfig)
        assert isinstance(cfg.cache, CacheConfig)

    def test_to_dict(self):
        cfg = OptimizerConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert 'context_compression' in d
        assert 'cache' in d
        assert 'debug' in d

    def test_nested_config_override(self):
        cfg = OptimizerConfig(
            context_compression=ContextCompressionConfig(compression_ratio=0.5),
            cache=CacheConfig(ttl=7200),
        )
        assert cfg.context_compression.compression_ratio == 0.5
        assert cfg.cache.ttl == 7200

    def test_invalid_global_log_level(self):
        with pytest.raises(Exception):
            OptimizerConfig(log_level="BANANA")

    def test_roundtrip_dict(self):
        cfg = OptimizerConfig(debug=True, log_level="DEBUG")
        d = cfg.to_dict()
        cfg2 = OptimizerConfig(**d)
        assert cfg2.debug is True
        assert cfg2.log_level == "DEBUG"
