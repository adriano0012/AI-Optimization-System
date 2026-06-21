"""
Default Configuration for Universal AI Optimizer
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class ContextCompressionConfig:
    """Configuration for context compression"""
    enabled: bool = True
    compression_ratio: float = 0.3  # Target compression ratio (0.0 to 1.0)
    methods: list = field(default_factory=lambda: [
        'hierarchical_summarization',
        'semantic_compression',
        'token_compression',
        'context_pruning'
    ])
    max_context_length: int = 4096
    preserve_recent: bool = True
    recent_tokens: int = 512

    def to_dict(self):
        return self.__dict__

@dataclass
class MemoryConfig:
    """Configuration for memory management"""
    enabled: bool = True
    memory_types: list = field(default_factory=lambda: [
        'working',
        'episodic',
        'semantic',
        'long_term'
    ])
    vector_db: str = 'faiss'  # Options: faiss, pinecone, weaviate, chromadb
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    max_memories: int = 10000
    memory_threshold: float = 0.7

    def to_dict(self):
        return self.__dict__

@dataclass
class CacheConfig:
    """Configuration for caching"""
    enabled: bool = True
    cache_types: list = field(default_factory=lambda: [
        'prompt',
        'embedding',
        'result'
    ])
    backend: str = 'multi_level'  # Options: redis, local, memcached, multi_level
    ttl: int = 3600  # Time to live in seconds
    max_size: int = 1000  # Maximum number of entries
    hierarchical: bool = True

    def to_dict(self):
        return self.__dict__

@dataclass
class ExecutionConfig:
    """Configuration for execution engine"""
    enabled: bool = True
    quantization: str = 'int8'  # Options: none, fp16, bf16, int8, int6, int4
    batch_size: int = 1
    use_flash_attention: bool = True
    max_sequence_length: int = 4096
    offload_to_cpu: bool = False
    offload_to_disk: bool = False

    def to_dict(self):
        return self.__dict__

@dataclass
class VerificationConfig:
    """Configuration for verification engine"""
    enabled: bool = True
    methods: list = field(default_factory=lambda: [
        'self_consistency',
        'fact_checking',
        'confidence_scoring'
    ])
    threshold: float = 0.95
    max_iterations: int = 3

    def to_dict(self):
        return self.__dict__

@dataclass
class ResponseConfig:
    """Configuration for response optimization"""
    enabled: bool = True
    compression: bool = True
    rephrasing: bool = False
    factuality_check: bool = True
    latency_target_ms: float = 1000.0

    def to_dict(self):
        return self.__dict__

@dataclass
class OptimizationBrainConfig:
    """Configuration for Optimization Brain"""
    enabled: bool = True
    def to_dict(self): return self.__dict__

@dataclass
class AutoTuningConfig:
    """Configuration for Auto Tuning"""
    enabled: bool = True
    def to_dict(self): return self.__dict__

@dataclass
class LearningRouterConfig:
    """Configuration for Learning Router"""
    enabled: bool = True
    def to_dict(self): return self.__dict__

@dataclass
class BenchmarkConfig:
    """Configuration for Benchmark"""
    enabled: bool = True
    def to_dict(self): return self.__dict__

@dataclass
class MultiAgentConfig:
    """Configuration for Multi-Agent System"""
    enabled: bool = True
    def to_dict(self): return self.__dict__

@dataclass
class LoggingConfig:
    """Configuration for structured logging"""
    enabled: bool = True
    json_format: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    pii_redaction: bool = True

    def to_dict(self): return self.__dict__

@dataclass
class MetricsConfig:
    """Configuration for metrics collection and export"""
    enabled: bool = True
    prefix: str = 'universal_ai_optimizer'
    enable_prometheus: bool = False
    prometheus_port: int = 8000

    def to_dict(self): return self.__dict__

@dataclass
class MonitoringConfig:
    """Configuration for health checks and system monitoring"""
    enabled: bool = True
    health_check_interval: float = 30.0
    enable_system_monitoring: bool = True

    def to_dict(self): return self.__dict__

@dataclass
class TelemetryConfig:
    """Configuration for telemetry collection"""
    enabled: bool = False  # Disabled by default (privacy)
    service_name: str = 'universal_ai_optimizer'
    environment: str = 'development'

    def to_dict(self): return self.__dict__

@dataclass
class OptimizerConfig:
    """Main configuration for the Universal AI Optimizer"""
    context_compression: ContextCompressionConfig = field(default_factory=ContextCompressionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    response: ResponseConfig = field(default_factory=ResponseConfig)
    optimization_brain: OptimizationBrainConfig = field(default_factory=OptimizationBrainConfig)
    auto_tuning: AutoTuningConfig = field(default_factory=AutoTuningConfig)
    learning_router: LearningRouterConfig = field(default_factory=LearningRouterConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    multi_agent: MultiAgentConfig = field(default_factory=MultiAgentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    
    # Global settings
    debug: bool = False
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'context_compression': self.context_compression.to_dict(),
            'memory': self.memory.to_dict(),
            'cache': self.cache.to_dict(),
            'execution': self.execution.to_dict(),
            'verification': self.verification.to_dict(),
            'response': self.response.to_dict(),
            'optimization_brain': self.optimization_brain.to_dict(),
            'auto_tuning': self.auto_tuning.to_dict(),
            'learning_router': self.learning_router.to_dict(),
            'benchmark': self.benchmark.to_dict(),
            'multi_agent': self.multi_agent.to_dict(),
            'logging': self.logging.to_dict(),
            'metrics': self.metrics.to_dict(),
            'monitoring': self.monitoring.to_dict(),
            'telemetry': self.telemetry.to_dict(),
            'debug': self.debug,
            'log_level': self.log_level
        }