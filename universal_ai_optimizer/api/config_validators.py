"""
Pydantic Config Validators
Validates and serializes configuration with type safety.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum


class QuantizationType(str, Enum):
    NONE = "none"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT6 = "int6"
    INT4 = "int4"


class VectorDBType(str, Enum):
    FAISS = "faiss"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    CHROMADB = "chromadb"


class CacheBackend(str, Enum):
    REDIS = "redis"
    LOCAL = "local"
    MEMCACHED = "memcached"
    MULTI_LEVEL = "multi_level"


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ContextCompressionConfig(BaseModel):
    enabled: bool = True
    compression_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    methods: List[str] = Field(default_factory=lambda: [
        'hierarchical_summarization', 'semantic_compression',
        'token_compression', 'context_pruning',
    ])
    max_context_length: int = Field(default=4096, ge=256, le=1000000)
    preserve_recent: bool = True
    recent_tokens: int = Field(default=512, ge=0, le=100000)

    @field_validator('methods')
    @classmethod
    def validate_methods(cls, v):
        valid = {'hierarchical_summarization', 'semantic_compression',
                 'token_compression', 'context_pruning'}
        for m in v:
            if m not in valid:
                raise ValueError(f"Invalid compression method: {m}. Must be one of {valid}")
        return v


class MemoryConfig(BaseModel):
    enabled: bool = True
    memory_types: List[str] = Field(default_factory=lambda: [
        'working', 'episodic', 'semantic', 'long_term',
    ])
    vector_db: VectorDBType = VectorDBType.FAISS
    embedding_model: str = Field(default='sentence-transformers/all-MiniLM-L6-v2', min_length=1)
    max_memories: int = Field(default=10000, ge=1, le=10000000)
    memory_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class CacheConfig(BaseModel):
    enabled: bool = True
    cache_types: List[str] = Field(default_factory=lambda: ['prompt', 'embedding', 'result'])
    backend: CacheBackend = CacheBackend.MULTI_LEVEL
    ttl: int = Field(default=3600, ge=1, le=86400)
    max_size: int = Field(default=1000, ge=1, le=1000000)
    hierarchical: bool = True


class ExecutionConfig(BaseModel):
    enabled: bool = True
    quantization: QuantizationType = QuantizationType.INT8
    batch_size: int = Field(default=1, ge=1, le=256)
    use_flash_attention: bool = True
    max_sequence_length: int = Field(default=4096, ge=128, le=1000000)
    offload_to_cpu: bool = False
    offload_to_disk: bool = False


class VerificationConfig(BaseModel):
    enabled: bool = True
    methods: List[str] = Field(default_factory=lambda: [
        'self_consistency', 'fact_checking', 'confidence_scoring',
    ])
    threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    max_iterations: int = Field(default=3, ge=1, le=10)


class ResponseConfig(BaseModel):
    enabled: bool = True
    compression: bool = True
    rephrasing: bool = False
    factuality_check: bool = True
    latency_target_ms: float = Field(default=1000.0, ge=1.0, le=60000.0)


class SimpleConfig(BaseModel):
    enabled: bool = True


class LoggingConfig(BaseModel):
    enabled: bool = True
    json_format: bool = True
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = None
    pii_redaction: bool = True

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid}")
        return v.upper()


class MetricsConfig(BaseModel):
    enabled: bool = True
    prefix: str = Field(default='universal_ai_optimizer', min_length=1)
    enable_prometheus: bool = False
    prometheus_port: int = Field(default=8000, ge=1024, le=65535)


class MonitoringConfig(BaseModel):
    enabled: bool = True
    health_check_interval: float = Field(default=30.0, ge=1.0, le=3600.0)
    enable_system_monitoring: bool = True


class TelemetryConfig(BaseModel):
    enabled: bool = False
    service_name: str = Field(default='universal_ai_optimizer', min_length=1)
    environment: Environment = Environment.DEVELOPMENT


class OptimizerConfig(BaseModel):
    context_compression: ContextCompressionConfig = Field(default_factory=ContextCompressionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    response: ResponseConfig = Field(default_factory=ResponseConfig)
    optimization_brain: SimpleConfig = Field(default_factory=SimpleConfig)
    auto_tuning: SimpleConfig = Field(default_factory=SimpleConfig)
    learning_router: SimpleConfig = Field(default_factory=SimpleConfig)
    benchmark: SimpleConfig = Field(default_factory=SimpleConfig)
    multi_agent: SimpleConfig = Field(default_factory=SimpleConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    debug: bool = False
    log_level: str = Field(default="INFO")

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid}")
        return v.upper()

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
