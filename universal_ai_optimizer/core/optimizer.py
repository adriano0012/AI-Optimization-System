"""
Universal AI Optimizer Core
Main orchestrator for all optimization modules
"""

from typing import Dict, Any, Optional, List, Callable, Type
import logging
import hashlib
import secrets
import time
import importlib
from dataclasses import dataclass

from universal_ai_optimizer.core.pipeline import OptimizationPipeline
from universal_ai_optimizer.configs.default import OptimizerConfig

logger = logging.getLogger(__name__)

MAX_PROMPT_LENGTH = 100_000


class SecurityViolation(Exception):
    pass


class RateLimitExceeded(SecurityViolation):
    pass


class InjectionDetected(SecurityViolation):
    pass


class PromptTooLargeError(ValueError):
    pass


class LazyModule:
    """True lazy loading proxy - module is only imported when first accessed."""
    
    def __init__(self, module_path: str, class_name: str, config: Optional[Dict] = None):
        self._module_path = module_path
        self._class_name = class_name
        self._config = config or {}
        self._instance = None
        self._loaded = False
    
    def _load(self):
        if not self._loaded:
            try:
                module = importlib.import_module(self._module_path)
                cls = getattr(module, self._class_name)
                self._instance = cls(self._config)
                self._loaded = True
            except Exception as e:
                logger.error(f"Failed to lazy-load {self._module_path}.{self._class_name}: {e}")
                self._instance = None
                self._loaded = True
    
    def __getattr__(self, name):
        self._load()
        if self._instance is None:
            raise AttributeError(f"LazyModule {self._module_path}.{self._class_name} failed to load")
        return getattr(self._instance, name)
    
    def __call__(self, *args, **kwargs):
        self._load()
        if self._instance is None:
            raise RuntimeError(f"LazyModule {self._module_path}.{self._class_name} failed to load")
        return self._instance(*args, **kwargs)
    
    @property
    def is_loaded(self):
        return self._loaded and self._instance is not None


class SecurityPipeline:
    def __init__(self, custom_checks: Optional[List[Callable[..., Any]]] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.custom_checks = custom_checks or []
        
        # Lazy-load security modules
        self._rate_limiter = None
        self._injection_detector = None
        self._context_sanitizer = None
        self._pii_filter = None
        
    @property
    def rate_limiter(self):
        if self._rate_limiter is None:
            from universal_ai_optimizer.modules.security.rate_limiter import RateLimiter
            self._rate_limiter = RateLimiter()
        return self._rate_limiter
        
    @property
    def injection_detector(self):
        if self._injection_detector is None:
            from universal_ai_optimizer.modules.security.injection_detector import InjectionDetector
            self._injection_detector = InjectionDetector()
        return self._injection_detector
        
    @property
    def context_sanitizer(self):
        if self._context_sanitizer is None:
            from universal_ai_optimizer.modules.security.context_sanitizer import ContextSanitizer
            self._context_sanitizer = ContextSanitizer()
        return self._context_sanitizer
        
    @property
    def pii_filter(self):
        if self._pii_filter is None:
            from universal_ai_optimizer.modules.security.pii_filter import PIIFilter
            self._pii_filter = PIIFilter()
        return self._pii_filter

    def _run_builtin_checks(self, prompt: str, context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        if not self.rate_limiter.allow(user_id):
            raise RateLimitExceeded("Too many requests. Please try again later.")
        if self.injection_detector.detect(prompt):
            raise InjectionDetected("Input rejected by security policy.")
        sanitized_context = self.context_sanitizer.sanitize(context)
        filtered_context = self.pii_filter.filter_dict(sanitized_context)
        return filtered_context

    def enforce(self, prompt: str, context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        safe_context = self._run_builtin_checks(prompt, context, user_id)
        for check in self.custom_checks:
            try:
                check(prompt, safe_context, user_id)
            except Exception as e:
                self.logger.warning(f"Custom security check failed: {e}")
                raise
        return safe_context


@dataclass
class OptimizationResult:
    """Result of optimization process"""
    original_prompt: str
    optimized_prompt: str
    compressed_context: Dict[str, Any]
    cached_result: Optional[Any]
    verification_score: float
    latency_ms: float
    token_savings: float
    resource_savings: Dict[str, float]

class UniversalAIOptimizer:
    """
    Main optimizer class that coordinates all optimization modules
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.config = config or OptimizerConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize security pipeline
        self.security_pipeline = SecurityPipeline()
        
        # Lazy-load observability modules (only when accessed)
        self.logging = LazyModule(
            'universal_ai_optimizer.modules.observability.logging', 
            'Logging',
            self.config.logging.to_dict() if hasattr(self.config, 'logging') else {}
        )
        self.metrics = LazyModule(
            'universal_ai_optimizer.modules.observability.metrics', 
            'Metrics',
            self.config.metrics.to_dict() if hasattr(self.config, 'metrics') else {}
        )
        self.monitoring = LazyModule(
            'universal_ai_optimizer.modules.observability.monitoring', 
            'Monitoring',
            self.config.monitoring.to_dict() if hasattr(self.config, 'monitoring') else {}
        )
        self.telemetry = LazyModule(
            'universal_ai_optimizer.modules.observability.telemetry', 
            'Telemetry',
            self.config.telemetry.to_dict() if hasattr(self.config, 'telemetry') else {}
        )
        
        # Lazy-load metric trackers for routing
        cost_tracker = LazyModule(
            'universal_ai_optimizer.modules.routing.real_metrics.cost_tracker', 
            'RealCostTracker',
            {}
        )
        latency_tracker = LazyModule(
            'universal_ai_optimizer.modules.routing.real_metrics.latency_tracker', 
            'RealLatencyTracker',
            {}
        )
        quality_tracker = LazyModule(
            'universal_ai_optimizer.modules.routing.real_metrics.quality_tracker', 
            'QualityTracker',
            {}
        )
        
        # Lazy-load all optimization modules
        self.context_compressor = LazyModule(
            'universal_ai_optimizer.modules.context_compressor', 
            'ContextCompressor',
            self.config.context_compression.to_dict()
        )
        
        # Lazy-load GPU optimizer
        self.gpu_optimizer = LazyModule(
            'universal_ai_optimizer.modules.gpu_optimizer', 
            'GPUOptimizer',
            self.config.gpu.to_dict() if hasattr(self.config, 'gpu') else {}
        )
        
        self.memory_manager = LazyModule(
            'universal_ai_optimizer.modules.memory_manager', 
            'MemoryManager',
            self.config.memory.to_dict()
        )
        
        self.cache_manager = LazyModule(
            'universal_ai_optimizer.modules.cache_manager', 
            'CacheManager',
            self.config.cache.to_dict()
        )
        
        self.execution_engine = LazyModule(
            'universal_ai_optimizer.modules.execution_engine', 
            'ExecutionEngine',
            self.config.execution.to_dict()
        )
        
        self.verification_engine = LazyModule(
            'universal_ai_optimizer.modules.verification_engine', 
            'VerificationEngine',
            self.config.verification.to_dict()
        )
        
        self.response_optimizer = LazyModule(
            'universal_ai_optimizer.modules.response_optimizer', 
            'ResponseOptimizer',
            self.config.response.to_dict()
        )
        
        # Lazy-load Optimization Brain
        self.optimization_brain = LazyModule(
            'universal_ai_optimizer.modules.optimization_brain.optimization_brain', 
            'OptimizationBrain',
            self.config.optimization_brain.to_dict() if hasattr(self.config, 'optimization_brain') else {}
        )
        
        # Lazy-load auto-tuner
        self.auto_tuner = LazyModule(
            'universal_ai_optimizer.modules.auto.auto_tuner', 
            'AutoTuner',
            self.config.auto_tuning.to_dict() if hasattr(self.config, 'auto_tuning') else {}
        )
        
        # Lazy-load learning router
        self.learning_router = LazyModule(
            'universal_ai_optimizer.modules.routing.learning_router', 
            'LearningRouter',
            self.config.learning_router.to_dict() if hasattr(self.config, 'learning_router') else {}
        )

        # Lazy-load task classifier
        self.task_classifier = LazyModule(
            'universal_ai_optimizer.modules.task_classifier', 
            'TaskClassifier',
            {}
        )
        
        # Lazy-load benchmark manager
        self.benchmark_manager = LazyModule(
            'universal_ai_optimizer.modules.benchmark.benchmark_manager', 
            'BenchmarkManager',
            self.config.benchmark.to_dict() if hasattr(self.config, 'benchmark') else {}
        )
        
        # Lazy-load model router with the trackers
        self.model_router = LazyModule(
            'universal_ai_optimizer.modules.routing.model_router', 
            'ModelRouter',
            {
                'enabled': True,
                'strategy': 'adaptive',
                'strategy_weights': {
                    'cost': 0.25,
                    'latency': 0.25,
                    'quality': 0.35,
                    'recency': 0.15
                },
                'cost_tracker': cost_tracker,
                'latency_tracker': latency_tracker,
                'quality_tracker': quality_tracker,
                'max_history_size': 10000
            }
        )
        
        # Initialize multi-agent orchestrator
        multi_agent_config = getattr(self.config, 'multi_agent', None)
        self.multi_agent_enabled = (
            multi_agent_config is not None 
            and hasattr(multi_agent_config, 'enabled') 
            and multi_agent_config.enabled is True
        )
        if self.multi_agent_enabled:
            self.orchestrator = LazyModule(
                'universal_ai_optimizer.modules.multi_agent.orchestrator_agent', 
                'OrchestratorAgent',
                self.config.multi_agent.to_dict() if hasattr(self.config, 'multi_agent') else {}
            )
        else:
            self.orchestrator = None
        
        # Create optimization pipeline with module lookup dict
        pipeline_modules = [
            self.task_classifier,
            self.context_compressor,
            self.memory_manager,
            self.cache_manager,
            self.execution_engine,
            self.verification_engine,
            self.response_optimizer,
            self.gpu_optimizer
        ]
        self.pipeline = OptimizationPipeline(pipeline_modules)
        
        self.logger.info("Universal AI Optimizer initialized")
    
    def optimize(self, prompt: str, context: Optional[Dict[str, Any]] = None, 
                model_adapter: Optional[Any] = None,
                user_id: Optional[str] = None) -> OptimizationResult:
        """
        Main optimization entry point
        
        Args:
            prompt: User input prompt
            context: Additional context (conversation history, documents, etc.)
            model_adapter: Adapter for specific LLM backend
            user_id: User identifier for rate limiting
            
        Returns:
            OptimizationResult with optimized outputs and metrics
        """
        start_time = time.time()
        
        max_prompt_len = getattr(self.config, 'max_prompt_length', MAX_PROMPT_LENGTH)
        if len(prompt) > max_prompt_len:
            raise PromptTooLargeError(f"Prompt exceeds maximum length of {max_prompt_len} characters.")
        
        if user_id is None:
            # Use HMAC with secret key for stable anonymous rate limiting to prevent prompt reconstruction
            import hmac
            import os
            # Use a secret key from config or generate one
            if hasattr(self.config, 'security'):
                secret_key = self.config.security.get('anonymous_user_secret', os.urandom(16).hex())
            else:
                secret_key = os.urandom(16).hex()
            
            # Ensure secret_key is bytes for HMAC
            if isinstance(secret_key, str):
                secret_key = secret_key.encode('utf-8')
                
            # Create HMAC object and get hex digest
            h = hmac.new(secret_key, prompt.encode('utf-8'), hashlib.sha256)
            user_id = f"anon_{h.hexdigest()[:16]}"
        user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:8]
        self.logger.info(f"Starting optimization for prompt: [{len(prompt)} chars] user={user_hash}")
        
        safe_context = self.security_pipeline.enforce(prompt, context or {}, user_id)
        
        # Task classification: enrich context with task type
        if self.task_classifier is not None:
            classification = self.task_classifier.process(prompt, safe_context)
            if classification:
                if 'task_type' not in safe_context or not safe_context.get('task_type'):
                    safe_context['task_type'] = classification.get('task_type', 'general')
                safe_context['task_confidence'] = classification.get('task_confidence', 0.0)
        
        # Multi-Agent orchestration: enrich context with agent analysis
        if self.multi_agent_enabled and self.orchestrator is not None:
            try:
                orchestrator_result = self.orchestrator.process(prompt, {
                    'task_type': safe_context.get('task_type', 'general'),
                    'difficulty': safe_context.get('difficulty', 'medium'),
                })
                if orchestrator_result and not orchestrator_result.get('error'):
                    safe_context['orchestrator_result'] = orchestrator_result
                    agent_outputs = orchestrator_result.get('agent_outputs', {})
                    # Extract planner's plan into context
                    if 'PlannerAgent' in agent_outputs:
                        plan = agent_outputs['PlannerAgent']
                        if isinstance(plan, dict):
                            safe_context['plan'] = plan
                            safe_context['goal'] = plan.get('goal', '')
                            safe_context['estimated_complexity'] = plan.get('estimated_complexity', 'medium')
                            safe_context['task_type'] = safe_context.get('task_type') or plan.get('goal', 'general')
                    # Inject verification agent confidence
                    if 'VerificationAgent' in agent_outputs:
                        ver = agent_outputs['VerificationAgent']
                        if isinstance(ver, dict) and 'confidence' in ver:
                            safe_context['agent_verification_confidence'] = ver['confidence']
                    # Inject research topics
                    if 'ResearchAgent' in agent_outputs:
                        research = agent_outputs['ResearchAgent']
                        if isinstance(research, dict):
                            safe_context['research_topics'] = research.get('key_topics', [])
            except Exception as e:
                self.logger.warning(f"Multi-agent orchestration failed: {e}")
        
        # Get optimization decisions from the brain
        decisions = None
        if self.optimization_brain is not None:
            try:
                decisions = self.optimization_brain.make_optimization_decision(
                    prompt=prompt,
                    context=safe_context,
                    available_models=list(self.execution_engine.adapters.keys()) if self.execution_engine.adapters else ['default'],
                    task_type=safe_context.get('task_type', 'general'),
                    difficulty=safe_context.get('difficulty', 'medium'),
                )
                safe_context['optimization_decisions'] = decisions

                # Use brain decisions to guide the pipeline
                strategy_dec = decisions.get('strategy', {})
                routing_dec = decisions.get('routing', {}).get('final_decision', {})
                quality_dec = decisions.get('quality', {})
                latency_dec = decisions.get('latency', {})
                cost_dec = decisions.get('cost', {})
                opt_config = decisions.get('optimization_config', {})

                safe_context['brain_guidance'] = {
                    'strategy': strategy_dec.get('selected_approach', 'balanced'),
                    'recommended_model': routing_dec.get('recommended_model'),
                    'routing_confidence': routing_dec.get('confidence', 0.0),
                    'quality_threshold': quality_dec.get('min_threshold', 0.7),
                    'max_latency_ms': latency_dec.get('max_latency_ms', 5000),
                    'max_cost_per_token': cost_dec.get('max_cost_per_token', float('inf')),
                    'optimization_flags': opt_config.get('optimization_flags', {}),
                }

                # If brain recommends a specific model, inject it for the execution engine
                if routing_dec.get('recommended_model') and routing_dec.get('confidence', 0) > 0.5:
                    safe_context['preferred_model'] = routing_dec['recommended_model']

                self.logger.debug(
                    f"Brain guidance: strategy={safe_context['brain_guidance']['strategy']}, "
                    f"model={safe_context['brain_guidance']['recommended_model']}, "
                    f"confidence={safe_context['brain_guidance']['routing_confidence']:.2f}"
                )
            except Exception as e:
                self.logger.warning(f"OptimizationBrain decision failed, using defaults: {e}")
                safe_context['brain_guidance'] = {}

        # Get auto-tuner current parameters to guide this run
        if self.auto_tuner is not None:
            try:
                tuning_recs = self.auto_tuner.get_tuning_recommendations()
                safe_context['tuning_parameters'] = tuning_recs.get('current_parameters', {})
                self.logger.debug(f"Auto-tuner parameters injected: {safe_context['tuning_parameters']}")
            except Exception as e:
                self.logger.debug(f"Failed to get auto-tuner recommendations: {e}")

        # Get learning router recommendation
        if self.learning_router is not None:
            router_ctx = {
                'task_type': safe_context.get('task_type', 'general'),
                'difficulty': safe_context.get('difficulty', 'medium'),
                'prompt_length': len(prompt),
                'available_options': list(self.execution_engine.adapters.keys()) if self.execution_engine.adapters else ['default']
            }
            routing = self.learning_router.route(router_ctx)
            safe_context['routing_decision'] = routing

        # Run through optimization pipeline
        pipeline_result = self.pipeline.process(
            prompt=prompt,
            context=safe_context,
            model_adapter=model_adapter
        )
        
        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        
        result = OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=pipeline_result.get('optimized_prompt', prompt),
            compressed_context=pipeline_result.get('compressed_context', {}),
            cached_result=pipeline_result.get('cached_result'),
            verification_score=pipeline_result.get('verification_score', 0.0),
            latency_ms=latency_ms,
            token_savings=pipeline_result.get('token_savings', 0.0),
            resource_savings=pipeline_result.get('resource_savings', {})
        )
        
        # Record experience in optimization brain for learning
        if self.optimization_brain is not None:
            try:
                exec_model = pipeline_result.get('adapter_name', 'default')
                experience = {
                    'model_name': exec_model,
                    'latency_ms': latency_ms,
                    'token_count': pipeline_result.get('tokens_generated', 0),
                    'quality_score': result.verification_score,
                    'token_savings': result.token_savings,
                    'task_type': safe_context.get('task_type', 'general'),
                    'difficulty': safe_context.get('difficulty', 'medium'),
                    'strategy': safe_context.get('brain_guidance', {}).get('strategy', 'balanced'),
                    'success': True,
                }
                self.optimization_brain.record_experience(experience)
            except Exception as e:
                self.logger.debug(f"Failed to record brain experience: {e}")

        # Update model router with execution metrics
        if hasattr(self, 'model_router') and self.model_router is not None:
            exec_model = pipeline_result.get('adapter_name', 'default')
            exec_tokens = pipeline_result.get('tokens_generated', 0)
            self.model_router.update_execution_result(
                model_name=exec_model,
                latency_ms=latency_ms,
                token_count=exec_tokens,
                quality_score=result.verification_score
            )
        
        # Update learning router with execution result
        if self.learning_router is not None:
            self.learning_router.update_performance(
                option=safe_context.get('routing_decision', 'default'),
                task_context={'task_type': safe_context.get('task_type', 'general')},
                reward=result.verification_score
            )
        
        # Auto-tune parameters based on performance metrics
        if self.auto_tuner is not None:
            tune_metrics = {
                'latency': latency_ms / 1000.0,
                'quality': result.verification_score,
                'token_savings': result.token_savings / 100.0,
            }
            try:
                self.auto_tuner.tune_parameters(tune_metrics)
            except Exception as e:
                self.logger.warning(f"Auto-tuning failed: {e}")

        # Use benchmark manager for lightweight performance evaluation
        if self.benchmark_manager is not None:
            try:
                self.benchmark_manager.run_benchmark_suite(self)
            except Exception as e:
                self.logger.debug(f"Benchmark evaluation skipped: {e}")
        
        # Record observability metrics
        if self.metrics is not None:
            try:
                self.metrics.increment('optimization.total')
                self.metrics.record('optimization.latency_ms', latency_ms)
                self.metrics.record('optimization.token_savings', result.token_savings)
                self.metrics.record('optimization.verification_score', result.verification_score)
            except Exception:
                pass
        
        if self.telemetry is not None:
            try:
                self.telemetry.capture('optimization.completed', {
                    'latency_ms': latency_ms,
                    'token_savings': result.token_savings,
                    'verification_score': result.verification_score,
                    'prompt_length': len(prompt),
                })
            except Exception:
                pass
        
        self.logger.info(f"Optimization completed in {latency_ms:.2f}ms")
        return result
    
    def register_model_adapter(self, name: str, adapter: Any):
        """Register a model adapter for specific LLM backend"""
        self.execution_engine.register_adapter(name, adapter)
        self.logger.info(f"Registered model adapter: {name}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current optimization metrics"""
        metrics = {
            'context_compression': self.context_compressor.get_metrics(),
            'memory': self.memory_manager.get_metrics(),
            'cache': self.cache_manager.get_metrics(),
            'execution': self.execution_engine.get_metrics(),
            'verification': self.verification_engine.get_metrics(),
            'response': self.response_optimizer.get_metrics(),
            'gpu_optimizer': self.gpu_optimizer.get_metrics(),
        }
        if self.optimization_brain is not None:
            metrics['optimization_brain'] = self.optimization_brain.get_metrics()
        if self.learning_router is not None:
            metrics['learning_router'] = self.learning_router.get_metrics()
        if self.auto_tuner is not None:
            metrics['auto_tuner'] = self.auto_tuner.get_metrics()
        if self.multi_agent_enabled and self.orchestrator is not None:
            metrics['orchestrator'] = self.orchestrator.get_metrics()
        if self.metrics is not None:
            metrics['metrics'] = self.metrics.get_metrics()
        if self.monitoring is not None:
            metrics['monitoring'] = self.monitoring.get_metrics()
        if self.telemetry is not None:
            metrics['telemetry'] = self.telemetry.get_metrics()
        return metrics

# Convenience function for simple usage
def optimize_prompt(prompt: str, context: Optional[Dict[str, Any]] = None, 
                   config: Optional[OptimizerConfig] = None) -> str:
    """
    Simple function to optimize a prompt
    
    Args:
        prompt: Input prompt
        context: Optional context
        config: Optional configuration
        
    Returns:
        Optimized prompt string
    """
    optimizer = UniversalAIOptimizer(config)
    result = optimizer.optimize(prompt, context)
    return result.optimized_prompt