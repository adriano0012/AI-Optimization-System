"""
Model Router Module
Responsible for selecting the appropriate model based on task, cost, latency, etc.
"""

from typing import Dict, Any, Optional, List
import logging
import time
from universal_ai_optimizer.core.base import BaseOptimizerModule
from universal_ai_optimizer.modules.routing.real_metrics.cost_tracker import RealCostTracker
from universal_ai_optimizer.modules.routing.real_metrics.latency_tracker import RealLatencyTracker
from universal_ai_optimizer.modules.routing.real_metrics.quality_tracker import QualityTracker

logger = logging.getLogger(__name__)

class ModelRouter(BaseOptimizerModule):
    """
    Routes tasks to the most suitable model based on various factors
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.routing_strategy = self.config.get('strategy', 'adaptive')  # adaptive, cost, latency, accuracy, quality
        self.available_models = self.config.get('available_models', {})
        self.model_performance = {}  # Cache for model performance metrics
        
        # Initialize real metric trackers (passed in from optimizer, but we can create defaults if not provided)
        self.cost_tracker = config.get('cost_tracker')
        self.latency_tracker = config.get('latency_tracker')
        self.quality_tracker = config.get('quality_tracker')
        
        # If trackers are not provided, create default ones (should not happen in normal operation)
        if self.cost_tracker is None:
            self.cost_tracker = RealCostTracker()
        if self.latency_tracker is None:
            self.latency_tracker = RealLatencyTracker()
        if self.quality_tracker is None:
            self.quality_tracker = QualityTracker()
        
        # Strategy weights for adaptive routing
        self.strategy_weights = self.config.get('strategy_weights', {
            'cost': 0.25,
            'latency': 0.25,
            'quality': 0.35,
            'recency': 0.15
        })
        
        # For continuous learning
        self.performance_history = []  # List of execution records
        self.max_history_size = self.config.get('max_history_size', 10000)
        
        self.logger.debug(f"ModelRouter initialized with strategy: {self.routing_strategy}")
    
    def process(self, prompt: str, context: Dict[str, Any], 
               model_adapter: Optional[Any] = None, 
               pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Select the best model for the given prompt and context based on real metrics
        
        Args:
            prompt: Input prompt
            context: Context dictionary (may include task type, difficulty, etc.)
            model_adapter: Model adapter (not used in routing, but required by interface)
            pipeline_state: Current pipeline state (may contain task type and difficulty from task classifier)
            
        Returns:
            Dictionary with selected model information to update pipeline_state
        """
        if not self.enabled:
            return {}
        
        self._log_processing(len(prompt), len(str(context)))
        
        # Extract task information from context or pipeline state
        task_type = None
        difficulty = None
        if pipeline_state:
            task_type = pipeline_state.get('task_type')
            difficulty = pipeline_state.get('difficulty')
        if task_type is None:
            task_type = context.get('task_type')
        if difficulty is None:
            difficulty = context.get('difficulty')
        
        # If we don't have task info, we can still route based on general performance
        # But we prefer to have task-specific routing
        
        selected_model = self._select_model(task_type, difficulty, context)
        
        # Store the selected model in the pipeline state for the execution engine to use
        result = {
            'selected_model': selected_model,
            'routing_reason': self._get_routing_reason(selected_model, task_type, difficulty),
            'routing_strategy': self.routing_strategy
        }
        
        self.logger.info(f"Selected model: {selected_model} for task: {task_type}")
        return result
    
    def _select_model(self, task_type: Optional[str], difficulty: Optional[str], 
                     context: Dict[str, Any]) -> str:
        """Select the best model based on the routing strategy using real metrics"""
        if not self.available_models:
            # Fallback to a default model
            return "default"
        
        if self.routing_strategy == 'cost':
            return self._select_by_cost(task_type, difficulty)
        elif self.routing_strategy == 'latency':
            return self._select_by_latency(task_type, difficulty)
        elif self.routing_strategy == 'quality':
            return self._select_by_quality(task_type, difficulty)
        elif self.routing_strategy == 'recency':
            return self._select_by_recency(task_type, difficulty)
        else:  # adaptive or default
            return self._select_adaptive(task_type, difficulty, context)
    
    def _select_by_cost(self, task_type: Optional[str], difficulty: Optional[str]) -> str:
        """Select model with lowest cost per token for the task"""
        # If we have task-specific cost data, we would use it, but for now we use overall cost
        best_model = None
        lowest_cost = float('inf')
        
        for model_name in self.available_models.keys():
            cost_per_token = self.cost_tracker.get_cost_per_token(model_name)
            if cost_per_token < lowest_cost:
                lowest_cost = cost_per_token
                best_model = model_name
        
        return best_model or list(self.available_models.keys())[0] if self.available_models else "default"
    
    def _select_by_latency(self, task_type: Optional[str], difficulty: Optional[str]) -> str:
        """Select model with lowest latency (p95) for the task"""
        best_model = None
        lowest_latency = float('inf')
        
        for model_name in self.available_models.keys():
            latency_p95 = self.latency_tracker.get_latency(model_name, 'p95')
            if latency_p95 < lowest_latency:
                lowest_latency = latency_p95
                best_model = model_name
        
        return best_model or list(self.available_models.keys())[0] if self.available_models else "default"
    
    def _select_by_quality(self, task_type: Optional[str], difficulty: Optional[str]) -> str:
        """Select model with highest quality score for the task"""
        best_model = None
        highest_quality = -1.0
        
        for model_name in self.available_models.keys():
            quality_score = self.quality_tracker.get_quality_score(model_name)
            if quality_score > highest_quality:
                highest_quality = quality_score
                best_model = model_name
        
        return best_model or list(self.available_models.keys())[0] if self.available_models else "default"
    
    def _select_by_recency(self, task_type: Optional[str], difficulty: Optional[str],
                          context: Optional[Dict[str, Any]] = None) -> str:
        """Select model that was used most recently (falls back to adaptive)"""
        return self._select_adaptive(task_type, difficulty, context or {})
    
    def _select_adaptive(self, task_type: Optional[str], difficulty: Optional[str], 
                        context: Dict[str, Any]) -> str:
        """Select model using adaptive strategy (balance of cost, latency, quality)"""
        # Placeholder: simple weighted score
        best_model = None
        best_score = -1
        
        for model_name in self.available_models.keys():
            # Get normalized scores (0-1, higher is better)
            cost_score = self.cost_tracker.get_cost_efficiency_score(model_name)  # higher is better
            latency_score = self.latency_tracker.get_latency_efficiency_score(model_name)  # higher is better
            quality_score = self.quality_tracker.get_quality_score(model_name)  # higher is better
            
            # For recency, we don't have a score yet, so we use 0.5 as neutral
            recency_score = 0.5
            
            # Weighted sum
            total_score = (
                self.strategy_weights['cost'] * cost_score +
                self.strategy_weights['latency'] * latency_score +
                self.strategy_weights['quality'] * quality_score +
                self.strategy_weights['recency'] * recency_score
            )
            
            if total_score > best_score:
                best_score = total_score
                best_model = model_name
        
        return best_model or list(self.available_models.keys())[0] if self.available_models else "default"
    
    def _get_routing_reason(self, model_name: str, task_type: Optional[str], 
                           difficulty: Optional[str]) -> str:
        """Generate a human-readable reason for the routing decision"""
        cost_per_token = self.cost_tracker.get_cost_per_token(model_name)
        latency_p95 = self.latency_tracker.get_latency(model_name, 'p95')
        quality_score = self.quality_tracker.get_quality_score(model_name)
        
        reason_parts = [f"Selected {model_name}"]
        
        if task_type:
            reason_parts.append(f"for task type '{task_type}'")
        if difficulty:
            reason_parts.append(f"with difficulty '{difficulty}'")
        
        reason_parts.append(f"using {self.routing_strategy} strategy")
        
        reason_parts.append(f"(cost: ${cost_per_token:.6f}/1K tokens, "
                          f"latency: {latency_p95:.2f}ms p95, "
                          f"quality: {quality_score:.3f})")
        
        return " ".join(reason_parts)
    
    def update_execution_result(self, model_name: str, latency_ms: float, 
                               token_count: int, quality_score: Optional[float] = None):
        """
        Update the trackers with the result of an execution
        Called by the execution engine after a successful run
        
        Args:
            model_name: The name of the model that was executed
            latency_ms: The latency of the execution in milliseconds
            token_count: The number of tokens used (input + output)
            quality_score: Optional quality score from benchmark or human evaluation
        """
        if not self.enabled:
            return
        
        # Update cost tracker
        self.cost_tracker.record_usage(model_name, token_count)
        
        # Update latency tracker
        self.latency_tracker.record_latency(model_name, latency_ms)
        
        # Update quality tracker if a score is provided
        if quality_score is not None:
            # We don't know which benchmark this score is from, so we use a generic benchmark name
            # In a real system, we would know the benchmark name
            self.quality_tracker.record_quality(model_name, 'general', quality_score)
        
        # Also add to performance history for potential future use
        self.performance_history.append({
            'model': model_name,
            'latency_ms': latency_ms,
            'token_count': token_count,
            'quality_score': quality_score,
            'timestamp': time.time()
        })
        
        # Trim history if needed
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size:]
        
        self.logger.debug(f"Updated trackers for model {model_name} with latency {latency_ms:.2f}ms, "
                         f"tokens {token_count}, quality {quality_score}")
    
    def register_model(self, name: str, metadata: Dict[str, Any]):
        """Register a model with its metadata"""
        self.available_models[name] = metadata
        self.logger.info(f"Registered model: {name}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get router metrics"""
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'routing_strategy': self.routing_strategy,
            'registered_models': list(self.available_models.keys()),
            'model_performance_cache_size': len(self.model_performance),
            'cost_tracker': self.cost_tracker.get_metrics(),
            'latency_tracker': self.latency_tracker.get_metrics(),
            'quality_tracker': self.quality_tracker.get_metrics(),
            'strategy_weights': self.strategy_weights,
            'performance_history_size': len(self.performance_history)
        })
        return base_metrics