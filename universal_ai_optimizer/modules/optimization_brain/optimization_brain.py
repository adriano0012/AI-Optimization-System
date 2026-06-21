"""
Optimization Brain
The central decision-making engine that coordinates all optimization strategies
"""

import time
import threading
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque
import json
import os
from universal_ai_optimizer.core.base import BaseOptimizerModule
from universal_ai_optimizer.modules.routing.model_router import ModelRouter
from universal_ai_optimizer.modules.routing.real_metrics.cost_tracker import RealCostTracker
from universal_ai_optimizer.modules.routing.real_metrics.latency_tracker import RealLatencyTracker
from universal_ai_optimizer.modules.routing.real_metrics.quality_tracker import QualityTracker
import logging

class DecisionEngine:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.experience_buffer = []
        self._lock = threading.RLock()

    def make_decision(self, inputs):
        alternatives = inputs.get('available_models', [])
        if not alternatives:
            return {'decision': 'default', 'confidence': 0.0, 'reasoning': 'no alternatives'}
        scores = []
        for alt in alternatives:
            score = self._score_alternative(alt, inputs)
            scores.append((score, alt))
        scores.sort(reverse=True)
        best_score, best_alt = scores[0]
        return {
            'decision': best_alt,
            'confidence': min(best_score / 10.0, 1.0),
            'reasoning': f'weighted scoring selected {best_alt}',
            'scores': {alt: s for s, alt in scores}
        }

    def _score_alternative(self, alt, inputs):
        prompt_len = inputs.get('prompt_length', 100)
        context_size = inputs.get('context_size', 100)
        return (
            -0.001 * prompt_len
            - 0.0005 * context_size
            + (0.5 if alt in self._get_recent_best() else 0.0)
        )

    def _get_recent_best(self):
        recent = self.experience_buffer[-50:] if len(self.experience_buffer) > 50 else self.experience_buffer
        if not recent:
            return set()
        best = {}
        for exp in recent:
            model = exp.get('model_name')
            score = exp.get('quality_score', 0)
            if model and (model not in best or score > best[model]):
                best[model] = score
        return {m for m, s in best.items() if s > 0.5}

    def update_from_experience(self, experiences):
        self.experience_buffer.extend(experiences)
        if len(self.experience_buffer) > 10000:
            self.experience_buffer = self.experience_buffer[-5000:]

    def get_metrics(self):
        return {'experience_count': len(self.experience_buffer)}

    def shutdown(self):
        pass


class StrategyEngine:
    STRATEGIES = {
        'balanced': {'params': {'speed_weight': 0.5, 'quality_weight': 0.5, 'cost_weight': 0.3}},
        'quality_first': {'params': {'speed_weight': 0.2, 'quality_weight': 0.8, 'cost_weight': 0.2}},
        'speed_first': {'params': {'speed_weight': 0.8, 'quality_weight': 0.2, 'cost_weight': 0.3}},
        'cost_effective': {'params': {'speed_weight': 0.3, 'quality_weight': 0.3, 'cost_weight': 0.8}},
    }

    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_strategy(self, context):
        task_type = context.get('task_type', 'general')
        difficulty = context.get('difficulty', 'medium')
        if task_type in ('code_generation', 'reasoning'):
            return 'quality_first'
        elif difficulty == 'easy':
            return 'speed_first'
        elif task_type == 'summarization':
            return 'cost_effective'
        return 'balanced'

    def make_decision(self, inputs):
        context = {k: v for k, v in inputs.items() if k in ('task_type', 'difficulty')}
        strategy = self.get_strategy(context)
        return {
            'decision': strategy,
            'selected_approach': strategy,
            'parameters': self.STRATEGIES[strategy]['params'],
            'enable_speculative_decoding': strategy == 'speed_first',
            'enable_dynamic_batching': strategy != 'quality_first',
            'enable_parallel_execution': strategy != 'quality_first',
            'enable_caching': True,
            'enable_memory_optimization': True,
            'confidence': 0.7
        }

    def update_from_experience(self, experiences):
        pass

    def get_metrics(self):
        return {}

    def shutdown(self):
        pass


class CostDecisionEngine:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cost_tracker = None

    def make_decision(self, inputs):
        available = inputs.get('available_models', [])
        costs = {}
        if self.cost_tracker:
            for model in available:
                costs[model] = self.cost_tracker.get_average_cost(model) or 0.01
        else:
            costs = {m: 0.01 for m in available}
        min_cost = min(costs.values()) if costs else 0.01
        threshold = min_cost * 2
        return {
            'prefer_lower_cost': True,
            'max_cost_per_token': threshold,
            'weight': 0.25,
            'recommended_model': min(costs, key=costs.get) if costs else None,
            'costs': costs
        }

    def update_from_experience(self, experiences):
        pass

    def get_metrics(self):
        return {}

    def shutdown(self):
        pass


class QualityDecisionEngine:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.quality_tracker = None

    def make_decision(self, inputs):
        task_type = inputs.get('task_type', 'general')
        difficulty = inputs.get('difficulty', 'medium')
        threshold_map = {'easy': 0.5, 'medium': 0.7, 'hard': 0.85}
        threshold = threshold_map.get(difficulty, 0.7)
        return {
            'min_threshold': threshold,
            'prefer_higher_quality': True,
            'weight': 0.35,
            'task_quality_requirements': {'task_type': task_type, 'difficulty': difficulty}
        }

    def update_from_experience(self, experiences):
        pass

    def get_metrics(self):
        return {}

    def shutdown(self):
        pass


class LatencyDecisionEngine:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.latency_tracker = None

    def make_decision(self, inputs):
        task_type = inputs.get('task_type', 'general')
        interactive = task_type in ('chat', 'conversation', 'streaming')
        max_latency = 500 if interactive else 10000
        return {
            'max_latency_ms': max_latency,
            'prefer_lower_latency': interactive,
            'weight': 0.25,
            'interactive_mode': interactive
        }

    def update_from_experience(self, experiences):
        pass

    def get_metrics(self):
        return {}

    def shutdown(self):
        pass


class AdaptiveCompute:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def make_decision(self, inputs):
        strategy = inputs.get('strategy_decisions', {}).get('selected_approach', 'balanced')
        complexity_map = {
            'speed_first': {'cpu_allocation': 0.3, 'gpu_allocation': 0.7, 'batch_size': 4},
            'quality_first': {'cpu_allocation': 0.6, 'gpu_allocation': 0.4, 'batch_size': 1},
            'cost_effective': {'cpu_allocation': 0.8, 'gpu_allocation': 0.2, 'batch_size': 2},
            'balanced': {'cpu_allocation': 0.5, 'gpu_allocation': 0.5, 'batch_size': 2},
        }
        return complexity_map.get(strategy, complexity_map['balanced'])

    def update_from_experience(self, experiences):
        pass

    def get_metrics(self):
        return {}

    def shutdown(self):
        pass


class ResourceMonitor:
    """Generic resource monitor for CPU, GPU, RAM, VRAM tracking."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.metrics = {}

    def update_from_experience(self, experiences):
        for exp in experiences[-100:]:
            for key in ('cpu_usage', 'memory_usage', 'gpu_usage'):
                if key in exp:
                    self.metrics[key] = exp[key]

    def get_metrics(self):
        return self.metrics.copy()

    def shutdown(self):
        pass


class ResourcePredictor:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.history = []

    def predict(self, workload):
        if not self.history:
            return {'cpu': 0.5, 'memory': 0.5, 'gpu': 0.0}
        recent = self.history[-50:] if len(self.history) > 50 else self.history
        avg_cpu = sum(r.get('cpu', 0.5) for r in recent) / len(recent)
        avg_mem = sum(r.get('memory', 0.5) for r in recent) / len(recent)
        avg_gpu = sum(r.get('gpu', 0.0) for r in recent) / len(recent)
        return {'cpu': avg_cpu, 'memory': avg_mem, 'gpu': avg_gpu}

    def record_usage(self, cpu, memory, gpu):
        self.history.append({'cpu': cpu, 'memory': memory, 'gpu': gpu})
        if len(self.history) > 1000:
            self.history = self.history[-500:]

    def update_from_experience(self, experiences):
        for exp in experiences[-100:]:
            cpu = exp.get('cpu_usage', 0.5)
            mem = exp.get('memory_usage', 0.5)
            gpu = exp.get('gpu_usage', 0.0)
            self.record_usage(cpu, mem, gpu)

    def get_metrics(self):
        return {'history_size': len(self.history)}

    def shutdown(self):
        pass


class AdvancedRouting:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def make_decision(self, inputs):
        router_recs = inputs.get('router_recommendations', {})
        votes = {}
        for name, rec in router_recs.items():
            model = rec.get('recommended_model')
            conf = rec.get('confidence', 0)
            if model:
                votes[model] = votes.get(model, 0) + conf
        if votes:
            best = max(votes, key=votes.get)
            return {'recommended_model': best, 'confidence': votes[best] / max(len(votes), 1), 'method': 'majority_vote'}
        return {'recommended_model': None, 'confidence': 0.0, 'method': 'no_consensus'}

    def update_from_experience(self, experiences):
        pass

    def get_metrics(self):
        return {}

    def shutdown(self):
        pass

class _BaseRouter:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def make_decision(self, inputs):
        available = inputs.get('available_models', ['default'])
        return {'recommended_model': available[0], 'confidence': 0.1}

    def update_from_experience(self, experiences):
        pass

    def get_metrics(self):
        return {}

    def shutdown(self):
        pass


logger = logging.getLogger(__name__)

class OptimizationBrain(BaseOptimizerModule):
    """
    The central optimization brain that makes all high-level decisions
    about model selection, resource allocation, and optimization strategies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        
        # Initialize core components
        self.decision_engine = DecisionEngine(self.config.get('decision_engine', {}))
        self.strategy_engine = StrategyEngine(self.config.get('strategy_engine', {}))
        self.resource_decision_engine = DecisionEngine(self.config.get('resource_decision_engine', {}))
        self.cost_decision_engine = CostDecisionEngine(self.config.get('cost_decision_engine', {}))
        self.quality_decision_engine = QualityDecisionEngine(self.config.get('quality_decision_engine', {}))
        self.latency_decision_engine = LatencyDecisionEngine(self.config.get('latency_decision_engine', {}))
        self.adaptive_compute = AdaptiveCompute(self.config.get('adaptive_compute', {}))
        
        # Initialize resource monitors (single generic class)
        self.cpu_manager = ResourceMonitor(self.config.get('cpu_manager', {}))
        self.gpu_manager = ResourceMonitor(self.config.get('gpu_manager', {}))
        self.ram_manager = ResourceMonitor(self.config.get('ram_manager', {}))
        self.vram_manager = ResourceMonitor(self.config.get('vram_manager', {}))
        
        # Initialize resource predictor
        self.resource_predictor = ResourcePredictor(self.config.get('resource_predictor', {}))
        
        # Initialize advanced routing components
        self.advanced_routing = AdvancedRouting(self.config.get('advanced_routing', {}))
        self.benchmark_router = _BaseRouter(self.config.get('benchmark_router', {}))
        self.quality_router = _BaseRouter(self.config.get('quality_router', {}))
        self.cost_router = _BaseRouter(self.config.get('cost_router', {}))
        self.latency_router = _BaseRouter(self.config.get('latency_router', {}))
        
        # Initialize the model router (will be enhanced with real metrics)
        self.model_router = None  # Will be set by the main optimizer
        
        # Initialize learning router
        self.learning_router = None  # Will be set externally
        
        # Learning and adaptation components
        self.experience_buffer = deque(maxlen=self.config.get('experience_buffer_size', 10000))
        self.adaptation_rate = self.config.get('adaptation_rate', 0.01)
        self.last_adaptation_time = time.time()
        self.adaptation_interval = self.config.get('adaptation_interval', 300)  # 5 minutes
        
        # Performance tracking for self-improvement
        self.performance_history = deque(maxlen=self.config.get('performance_history_size', 50000))
        self.improvement_flags = defaultdict(bool)  # Track what improvements have been made
        
        # Configuration versioning
        self.config_version = 0
        self.config_history = deque(maxlen=100)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Shutdown event for background thread
        self._shutdown_event = threading.Event()
        
        # Thread control flag
        self._should_run_adaptation = self.enabled
        
        # Start background adaptation thread
        if self.enabled and self.config.get('start_adaptation_thread', True):
            self._start_adaptation_thread()
        
        self.logger.info("Optimization Brain initialized")
    
    def process(self, prompt: str, context: Dict[str, Any],
                model_adapter: Optional[Any] = None,
                pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process method to satisfy BaseOptimizerModule interface.
        OptimizationBrain is not a pipeline module but provides optimization decisions.
        """
        return {}
    
    def _start_adaptation_thread(self):
        """Start background thread for continuous adaptation"""
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True, name="optimization-brain-adaptation")
        self.adaptation_thread.start()
        self.logger.debug("Optimization Brain adaptation thread started")
    
    def _adaptation_loop(self):
        """Background loop for continuous learning and adaptation with exponential backoff"""
        consecutive_failures = 0
        max_failures = 10
        base_interval = self.adaptation_interval
        
        while not self._shutdown_event.is_set():
            try:
                self._shutdown_event.wait(timeout=base_interval)
                if self._shutdown_event.is_set():
                    break
                self._adapt_and_improve()
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                self.logger.error(f"Error in adaptation loop ({consecutive_failures}/{max_failures}): {e}")
                if consecutive_failures >= max_failures:
                    self.logger.error("Max consecutive failures reached, stopping adaptation thread")
                    break
                # Exponential backoff: base * 2^failures, capped at 1 hour
                backoff = min(base_interval * (2 ** consecutive_failures), 3600)
                self._shutdown_event.wait(timeout=backoff)
    
    def _adapt_and_improve(self):
        """Perform adaptation and self-improvement based on accumulated experience"""
        if len(self.experience_buffer) < 10:  # Need minimum experience
            return
        
        self.logger.debug("Starting adaptation cycle")
        start_time = time.time()
        
        # 1. Analyze recent experiences
        with self._lock:
            recent_experiences = list(self.experience_buffer)[-1000:]
        
        # 2. Update decision engines based on experience (skip if method is a no-op)
        for engine in [self.decision_engine, self.strategy_engine,
                       self.resource_decision_engine, self.cost_decision_engine,
                       self.quality_decision_engine, self.latency_decision_engine]:
            if hasattr(engine, 'update_from_experience') and engine.update_from_experience is not None:
                engine.update_from_experience(recent_experiences)
        
        # 3. Update resource managers (only if they have real implementation)
        for manager in [self.cpu_manager, self.gpu_manager, self.ram_manager, self.vram_manager]:
            if hasattr(manager, 'update_from_experience') and manager.update_from_experience is not None:
                manager.update_from_experience(recent_experiences)
        
        # 4. Update predictors
        self.resource_predictor.update_from_experience(recent_experiences)
        
        # 5. Update routing engines
        self.advanced_routing.update_from_experience(recent_experiences)
        if self.learning_router is not None:
            for exp in recent_experiences:
                try:
                    self.learning_router.update_performance(
                        option=exp.get('model_name', 'unknown'),
                        task_context={'task_type': exp.get('task_type', 'unknown')},
                        reward=exp.get('quality_score', 0.5)
                    )
                except Exception:
                    self.logger.debug("Failed to update learning router performance")
        self.benchmark_router.update_from_experience(recent_experiences)
        self.quality_router.update_from_experience(recent_experiences)
        self.cost_router.update_from_experience(recent_experiences)
        self.latency_router.update_from_experience(recent_experiences)
        
        # 6. Update adaptive compute strategies
        self.adaptive_compute.update_from_experience(recent_experiences)
        
        # 7. Generate new configuration if improvements found
        new_config = self._generate_improved_configuration()
        if new_config:
            self._apply_configuration_update(new_config)
        
        # 8. Record adaptation
        adaptation_record = {
            'timestamp': time.time(),
            'experiences_processed': len(recent_experiences),
            'duration_seconds': time.time() - start_time,
            'config_version_before': self.config_version
        }
        
        self.config_history.append(adaptation_record)
        self.config_version += 1
        self.last_adaptation_time = time.time()
        
        self.logger.info(f"Adaptation cycle completed in {time.time() - start_time:.2f}s. "
                        f"New config version: {self.config_version}")
    
    def _generate_improved_configuration(self) -> Optional[Dict[str, Any]]:
        """Generate an improved configuration based on learned patterns"""
        # This would analyze the experience buffer to find better configurations
        # For now, we'll return None to indicate no automatic changes
        # In a full implementation, this would use techniques like:
        #   - Bayesian optimization
        #   - Genetic algorithms
        #   - Reinforcement learning
        #   - Multi-armed bandits
        
        # Placeholder: analyze experience for obvious improvements
        if len(self.experience_buffer) < 100:
            return None
        
        recent_exp = list(self.experience_buffer)[-100:]
        
        # Example: if we notice a particular model consistently gives better quality/latency/cost
        model_performance = defaultdict(list)
        for exp in recent_exp:
            model_name = exp.get('model_name')
            if model_name:
                # Calculate a composite score
                latency = exp.get('latency_ms', 1000)
                quality = exp.get('quality_score', 0.0)
                cost = exp.get('cost_usd', 0.01)
                
                # Normalize and combine (lower latency/cost is better, higher quality is better)
                # These would be properly normalized in a real implementation
                score = quality * 0.5 - (latency / 1000.0) * 0.3 - cost * 100 * 0.2  # Rough example
                model_performance[model_name].append(score)
        
        # Find the best performing model
        best_model = None
        best_avg_score = -float('inf')
        for model, scores in model_performance.items():
            if len(scores) >= 5:  # Minimum samples
                avg_score = sum(scores) / len(scores)
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_model = model
        
        # If we found a significantly better model, suggest a configuration change
        if best_model and best_avg_score > 0.5:  # Threshold for significance
            return {
                'type': 'model_preference_update',
                'preferred_model': best_model,
                'confidence': min(best_avg_score / 2.0, 1.0),  # Rough confidence
                'reasoning': f'Model {best_model} showed superior performance in recent experiences'
            }
        
        return None
    
    def _apply_configuration_update(self, config_update: Dict[str, Any]):
        """Apply a configuration update from the optimization brain"""
        self.logger.info(f"Applying configuration update: {config_update}")
        
        # In a real implementation, this would:
        #   1. Validate the update
        #   2. Apply it to the relevant components
        #   3. Restart or reconfigure affected services if needed
        #   4. Log the change for auditability
        
        # For now, we'll just log it
        if config_update.get('type') == 'model_preference_update':
            preferred = config_update.get('preferred_model', 'unknown')
            confidence = config_update.get('confidence', 0.0)
            self.logger.info(f"Learning suggests preferring model: {preferred} "
                           f"(confidence: {confidence:.2f})")
            # The actual model preference would be used by the model router
    
    def record_experience(self, experience: Dict[str, Any]):
        """
        Record an execution experience for learning
        
        Args:
            experience: Dictionary containing details about an execution
                       Should include: model_name, latency_ms, token_count, 
                       cost_usd, quality_score, task_type, success, etc.
        """
        if not self.enabled:
            return
        
        with self._lock:
            if 'timestamp' not in experience:
                experience['timestamp'] = time.time()
            self.experience_buffer.append(experience)
            self.performance_history.append(experience)
        
        self.logger.debug(f"Recorded experience for model {experience.get('model_name', 'unknown')}")
    
    def make_optimization_decision(self, prompt: str, context: Dict[str, Any], 
                                 available_models: List[str], 
                                 task_type: Optional[str] = None,
                                 difficulty: Optional[str] = None) -> Dict[str, Any]:
        """
        Make a comprehensive optimization decision
        
        Args:
            prompt: The input prompt
            context: Context information
            available_models: List of available model identifiers
            task_type: Type of task (e.g., 'question_answering', 'code_generation')
            difficulty: Difficulty level (e.g., 'easy', 'medium', 'hard')
            
        Returns:
            Dictionary with all optimization decisions
        """
        if not self.enabled:
            return self._get_default_decisions()
        
        self._log_processing(len(prompt), len(str(context)))
        
        # Gather inputs for decision making
        decision_inputs = {
            'prompt': prompt,
            'context': context,
            'available_models': available_models,
            'task_type': task_type,
            'difficulty': difficulty,
            'prompt_length': len(prompt),
            'context_size': len(str(context)),
            'timestamp': time.time()
        }
        
        # Make decisions in order of dependency
        decisions = {}
        
        # 1. Resource decisions (foundational)
        decisions['resources'] = self.resource_decision_engine.make_decision(decision_inputs)
        
        # 2. Cost decisions
        decisions['cost'] = self.cost_decision_engine.make_decision(decision_inputs)
        
        # 3. Quality decisions
        decisions['quality'] = self.quality_decision_engine.make_decision(decision_inputs)
        
        # 4. Latency decisions
        decisions['latency'] = self.latency_decision_engine.make_decision(decision_inputs)
        
        # 5. Strategy decisions (depends on above)
        decisions['strategy'] = self.strategy_engine.make_decision({
            **decision_inputs,
            'resource_decisions': decisions['resources'],
            'cost_decisions': decisions['cost'],
            'quality_decisions': decisions['quality'],
            'latency_decisions': decisions['latency']
        })
        
        # 6. Advanced routing decisions
        decisions['routing'] = self._make_routing_decision(decision_inputs, decisions)
        
        # 7. Adaptive compute decisions
        decisions['adaptive_compute'] = self.adaptive_compute.make_decision({
            **decision_inputs,
            'resource_decisions': decisions['resources'],
            'strategy_decisions': decisions['strategy']
        })
        
        # 8. Final optimization configuration
        decisions['optimization_config'] = self._synthesize_final_configuration(
            decision_inputs, decisions
        )
        
        # Record that we made this decision (for learning)
        decision_record = {
            'timestamp': time.time(),
            'prompt_hash': hashlib.sha256(prompt.encode()).hexdigest()[:16] if isinstance(prompt, str) else hashlib.sha256(str(prompt).encode()).hexdigest()[:16],
            'context_size': len(str(context)),
            'task_type': task_type,
            'difficulty': difficulty,
            'decisions_made': decisions,
            'available_models_count': len(available_models)
        }
        
        # We don't add this to experience buffer yet - that happens after execution
        # But we could add it to a decision history for analysis
        
        self.logger.debug(f"Made optimization decision for task {task_type} "
                         f"(difficulty: {difficulty})")
        
        return decisions
    
    def _make_routing_decision(self, inputs: Dict[str, Any], 
                             prior_decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Make routing decisions based on all prior decisions"""
        # Ask each router for its recommendation
        routing_decisions = {}
        
        # Learning router (uses historical performance)
        if self.learning_router is not None:
            routing_decisions['learning'] = self.learning_router.make_decision({
                **inputs,
                'prior_decisions': prior_decisions
            })
        else:
            routing_decisions['learning'] = {'recommended_model': None, 'confidence': 0.0}
        
        # Benchmark router (uses benchmark performance)
        routing_decisions['benchmark'] = self.benchmark_router.make_decision({
            **inputs,
            'prior_decisions': prior_decisions
        })
        
        # Quality router
        routing_decisions['quality'] = self.quality_router.make_decision({
            **inputs,
            'prior_decisions': prior_decisions
        })
        
        # Cost router
        routing_decisions['cost'] = self.cost_router.make_decision({
            **inputs,
            'prior_decisions': prior_decisions
        })
        
        # Latency router
        routing_decisions['latency'] = self.latency_router.make_decision({
            **inputs,
            'prior_decisions': prior_decisions
        })
        
        # Advanced routing (combines multiple factors)
        routing_decisions['advanced'] = self.advanced_routing.make_decision({
            **inputs,
            'prior_decisions': prior_decisions,
            'router_recommendations': routing_decisions
        })
        
        # Synthesize final routing decision
        final_routing = self._synthesize_routing_decision(routing_decisions)
        
        return {
            'individual_recommendations': routing_decisions,
            'final_decision': final_routing
        }
    
    def _synthesize_routing_decision(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize multiple routing recommendations into a final decision"""
        # Extract model recommendations from each router
        model_votes = defaultdict(float)
        total_weight = 0.0
        
        # Weights for different routing approaches (could be learned)
        router_weights = {
            'learning': 0.30,
            'benchmark': 0.25,
            'quality': 0.20,
            'cost': 0.15,
            'latency': 0.10
        }
        
        for router_name, recommendation in recommendations.items():
            if router_name in ['advanced', 'individual_recommendations']:
                continue  # Skip meta-recommendations
            
            weight = router_weights.get(router_name, 0.1)
            recommended_model = recommendation.get('recommended_model')
            confidence = recommendation.get('confidence', 0.5)
            
            if recommended_model:
                model_votes[recommended_model] += weight * confidence
                total_weight += weight
        
        # Also consider the advanced router's synthesized recommendation
        if 'advanced' in recommendations:
            advanced_rec = recommendations['advanced']
            advanced_model = advanced_rec.get('recommended_model')
            advanced_confidence = advanced_rec.get('confidence', 0.5)
            if advanced_model:
                # Give advanced router extra weight
                model_votes[advanced_model] += 0.25 * advanced_confidence
                total_weight += 0.25
        
        # Normalize votes
        if total_weight > 0:
            for model in model_votes:
                model_votes[model] /= total_weight
        
        # Select the model with highest vote
        if model_votes:
            best_model = max(model_votes, key=model_votes.get)
            confidence = model_votes[best_model]
        else:
            best_model = 'default'
            confidence = 0.0
        
        return {
            'recommended_model': best_model,
            'confidence': confidence,
            'vote_distribution': dict(model_votes),
            'method': 'weighted_ensemble'
        }
    
    def _synthesize_final_configuration(self, inputs: Dict[str, Any], 
                                      decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize all decisions into a final optimization configuration"""
        # Start with base configuration
        config = {
            'timestamp': time.time(),
            'task_type': inputs.get('task_type'),
            'difficulty': inputs.get('difficulty'),
            'prompt_length': inputs.get('prompt_length'),
            'context_size': inputs.get('context_size')
        }
        
        # Add resource configuration
        config['resources'] = {
            'cpu_allocation': decisions['resources'].get('cpu_allocation', 0.5),
            'gpu_allocation': decisions['resources'].get('gpu_allocation', 0.5),
            'ram_allocation_mb': decisions['resources'].get('ram_allocation_mb', 2048),
            'vram_allocation_mb': decisions['resources'].get('vram_allocation_mb', 1024),
            'offload_to_cpu': decisions['resources'].get('offload_to_cpu', False),
            'offload_to_disk': decisions['resources'].get('offload_to_disk', False)
        }
        
        # Add cost optimization
        config['cost_optimization'] = {
            'prefer_lower_cost': decisions['cost'].get('prefer_lower_cost', True),
            'max_cost_per_token': decisions['cost'].get('max_cost_per_token', float('inf')),
            'cost_weight': decisions['cost'].get('weight', 0.25)
        }
        
        # Add quality optimization
        config['quality_optimization'] = {
            'min_quality_threshold': decisions['quality'].get('min_threshold', 0.7),
            'prefer_higher_quality': decisions['quality'].get('prefer_higher_quality', True),
            'quality_weight': decisions['quality'].get('weight', 0.35)
        }
        
        # Add latency optimization
        config['latency_optimization'] = {
            'max_latency_ms': decisions['latency'].get('max_latency_ms', 5000),
            'prefer_lower_latency': decisions['latency'].get('prefer_lower_latency', True),
            'latency_weight': decisions['latency'].get('weight', 0.25)
        }
        
        # Add strategy selection
        config['strategy'] = {
            'approach': decisions['strategy'].get('selected_approach', 'balanced'),
            'parameters': decisions['strategy'].get('parameters', {})
        }
        
        # Add routing decision
        config['routing'] = decisions.get('routing', {}).get('final_decision', {})
        
        # Add adaptive compute settings
        config['adaptive_compute'] = decisions.get('adaptive_compute', {})
        
        # Add optimization flags
        config['optimization_flags'] = {
            'enable_speculative_decoding': decisions['strategy'].get('enable_speculative_decoding', False),
            'enable_dynamic_batching': decisions['strategy'].get('enable_dynamic_batching', False),
            'enable_parallel_execution': decisions['strategy'].get('enable_parallel_execution', False),
            'enable_caching': decisions['strategy'].get('enable_caching', True),
            'enable_memory_optimization': decisions['strategy'].get('enable_memory_optimization', True)
        }
        
        return config
    
    def _get_default_decisions(self) -> Dict[str, Any]:
        """Return default decisions when optimization brain is disabled"""
        return {
            'resources': {
                'cpu_allocation': 0.5,
                'gpu_allocation': 0.5,
                'ram_allocation_mb': 2048,
                'vram_allocation_mb': 1024,
                'offload_to_cpu': False,
                'offload_to_disk': False
            },
            'cost': {
                'prefer_lower_cost': True,
                'max_cost_per_token': float('inf'),
                'weight': 0.25
            },
            'quality': {
                'min_threshold': 0.7,
                'prefer_higher_quality': True,
                'weight': 0.35
            },
            'latency': {
                'max_latency_ms': 5000,
                'prefer_lower_latency': True,
                'weight': 0.25
            },
            'strategy': {
                'selected_approach': 'balanced',
                'parameters': {}
            },
            'routing': {
                'final_decision': {
                    'recommended_model': 'default',
                    'confidence': 0.0,
                    'method': 'fallback'
                }
            },
            'adaptive_compute': {},
            'optimization_config': {
                'timestamp': time.time(),
                'resources': {},
                'cost_optimization': {},
                'quality_optimization': {},
                'latency_optimization': {},
                'strategy': {},
                'routing': {},
                'adaptive_compute': {},
                'optimization_flags': {
                    'enable_speculative_decoding': False,
                    'enable_dynamic_batching': False,
                    'enable_parallel_execution': False,
                    'enable_caching': True,
                    'enable_memory_optimization': True
                }
            }
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get optimization brain metrics"""
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'experience_buffer_size': len(self.experience_buffer),
            'performance_history_size': len(self.performance_history),
            'config_version': self.config_version,
            'last_adaptation_seconds_ago': time.time() - self.last_adaptation_time,
            'adaptation_thread_active': getattr(self, '_should_run_adaptation', False) and 
                                      getattr(self, 'adaptation_thread', None) is not None and 
                                      self.adaptation_thread.is_alive(),
            'decision_engine': self.decision_engine.get_metrics() if hasattr(self.decision_engine, 'get_metrics') else {},
            'strategy_engine': self.strategy_engine.get_metrics() if hasattr(self.strategy_engine, 'get_metrics') else {},
            'resource_decision_engine': self.resource_decision_engine.get_metrics() if hasattr(self.resource_decision_engine, 'get_metrics') else {},
            'cost_decision_engine': self.cost_decision_engine.get_metrics() if hasattr(self.cost_decision_engine, 'get_metrics') else {},
            'quality_decision_engine': self.quality_decision_engine.get_metrics() if hasattr(self.quality_decision_engine, 'get_metrics') else {},
            'latency_decision_engine': self.latency_decision_engine.get_metrics() if hasattr(self.latency_decision_engine, 'get_metrics') else {},
            'adaptive_compute': self.adaptive_compute.get_metrics() if hasattr(self.adaptive_compute, 'get_metrics') else {},
            'cpu_manager': self.cpu_manager.get_metrics() if hasattr(self.cpu_manager, 'get_metrics') else {},
            'gpu_manager': self.gpu_manager.get_metrics() if hasattr(self.gpu_manager, 'get_metrics') else {},
            'ram_manager': self.ram_manager.get_metrics() if hasattr(self.ram_manager, 'get_metrics') else {},
            'vram_manager': self.vram_manager.get_metrics() if hasattr(self.vram_manager, 'get_metrics') else {},
            'resource_predictor': self.resource_predictor.get_metrics() if hasattr(self.resource_predictor, 'get_metrics') else {},
            'advanced_routing': self.advanced_routing.get_metrics() if hasattr(self.advanced_routing, 'get_metrics') else {},
            'benchmark_router': self.benchmark_router.get_metrics() if hasattr(self.benchmark_router, 'get_metrics') else {},
            'quality_router': self.quality_router.get_metrics() if hasattr(self.quality_router, 'get_metrics') else {},
            'cost_router': self.cost_router.get_metrics() if hasattr(self.cost_router, 'get_metrics') else {},
            'latency_router': self.latency_router.get_metrics() if hasattr(self.latency_router, 'get_metrics') else {}
        })
        return base_metrics
    
    def shutdown(self):
        """Shutdown the optimization brain and its components"""
        self._should_run_adaptation = False
        
        # Signal the adaptation thread to stop
        if hasattr(self, '_shutdown_event'):
            self._shutdown_event.set()
        
        if hasattr(self, 'adaptation_thread') and self.adaptation_thread.is_alive():
            self.adaptation_thread.join(timeout=10.0)
            if self.adaptation_thread.is_alive():
                self.logger.warning("Adaptation thread did not stop within timeout")
        
        # Shutdown subcomponents
        for component in [self.decision_engine, self.strategy_engine, 
                         self.resource_decision_engine, self.cost_decision_engine,
                         self.quality_decision_engine, self.latency_decision_engine,
                         self.adaptive_compute, self.cpu_manager, self.gpu_manager,
                         self.ram_manager, self.vram_manager, self.resource_predictor,
                         self.advanced_routing, self.benchmark_router,
                         self.quality_router, self.cost_router, self.latency_router]:
            if hasattr(component, 'shutdown'):
                try:
                    component.shutdown()
                except Exception as e:
                    self.logger.warning(f"Error shutting down component {component}: {e}")
        
        self.logger.info("Optimization Brain shutdown complete")