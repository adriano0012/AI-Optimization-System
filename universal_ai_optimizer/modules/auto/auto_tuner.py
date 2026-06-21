"""
Auto-Tuning Module
Automatically tunes optimization parameters based on performance feedback
"""

import logging
import os
import json
import threading
import time
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque
import numpy as np
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)


class AutoTuner(BaseOptimizerModule):
    """
    Auto-tuner that automatically adjusts optimization parameters
    based on performance feedback and historical data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.tuning_strategy = self.config.get('tuning_strategy', 'bayesian')  # bayesian, random, grid, rule_based
        
        # Performance tracking
        self.performance_history = defaultdict(lambda: deque(maxlen=10000))  # metric -> list of values
        self.parameter_history = defaultdict(lambda: deque(maxlen=10000))    # parameter -> list of values
        self.outcome_history = deque(maxlen=10000)  # List of (parameters, outcome) tuples
        
        # Tuning parameters
        self.tuning_parameters = self.config.get('tuning_parameters', {
            'learning_rate': {'min': 0.001, 'max': 0.1, 'type': 'float'},
            'batch_size': {'min': 16, 'max': 256, 'type': 'int'},
            'temperature': {'min': 0.0, 'max': 2.0, 'type': 'float'},
            'top_p': {'min': 0.1, 'max': 1.0, 'type': 'float'}
        })
        
        # Current parameter values
        self.current_parameters = {}
        for param_name, param_config in self.tuning_parameters.items():
            if param_config['type'] == 'float':
                self.current_parameters[param_name] = (param_config['min'] + param_config['max']) / 2.0
            else:  # int
                self.current_parameters[param_name] = (param_config['min'] + param_config['max']) // 2
        
        # Thread safety
        self._lock = threading.RLock()
        
        # For Bayesian optimization (simplified)
        self.gp_models = {}  # Gaussian Process models for each metric
        self.X_samples = []  # Parameter samples
        self.y_samples = defaultdict(list)  # Metric values for each sample
        
        # Load persisted state
        self._load_persisted_state()
        
        self.logger.debug(f"AutoTuner initialized with strategy: {self.tuning_strategy}")

    def _get_data_dir(self) -> str:
        """Return a safe absolute directory for persisting state.
        Validates that the resolved path does not escape the base directory."""
        base = os.path.join(os.getcwd(), "data", "auto_tuner")
        base = os.path.realpath(base)
        os.makedirs(base, exist_ok=True)
        return base

    def _validate_path(self, path: str, base_dir: str) -> bool:
        """Check that resolved path stays within base_dir."""
        resolved = os.path.realpath(path)
        return resolved.startswith(base_dir + os.sep) or resolved == base_dir

    def process(self, prompt: str, context: Dict[str, Any],
                model_adapter: Optional[Any] = None, 
                pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Dummy process method to satisfy BaseOptimizerModule abstract class.
        The auto tuner is not used in the pipeline but for internal parameter tuning.
        """
        return {}
    
    def tune_parameters(self, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Tune parameters based on current performance metrics
        
        Args:
            performance_metrics: Dictionary of metric names to values (higher is better)
            
        Returns:
            Dictionary of recommended parameter adjustments
        """
        if not self.enabled:
            with self._lock:
                return self.current_parameters.copy()
        
        with self._lock:
            for metric_name, value in performance_metrics.items():
                self.performance_history[metric_name].append(value)
            
            for param_name, value in self.current_parameters.items():
                self.parameter_history[param_name].append(value)
            
            outcome = self._compute_outcome(performance_metrics)
            self.outcome_history.append((self.current_parameters.copy(), outcome))
        
        self._update_bayesian_samples(performance_metrics)
        
        if self.tuning_strategy == 'bayesian':
            new_params = self._bayesian_optimization()
        elif self.tuning_strategy == 'random':
            new_params = self._random_search()
        elif self.tuning_strategy == 'grid':
            new_params = self._grid_search()
        else:
            new_params = self._rule_based_tuning(performance_metrics)
        
        new_params = self._apply_constraints(new_params)
        
        with self._lock:
            self.current_parameters = self._smooth_parameters(new_params)
            result = self.current_parameters.copy()
        
        self.logger.debug(f"Tuned parameters: {result}")
        return result
    
    def _compute_outcome(self, metrics: Dict[str, float]) -> float:
        """
        Compute overall outcome score from multiple metrics
        Higher is better
        """
        # Simple weighted average (can be made more sophisticated)
        weights = self.config.get('outcome_weights', {
            'latency': -0.3,  # Lower latency is better
            'quality': 0.4,   # Higher quality is better
            'cost': -0.2,     # Lower cost is better
            'throughput': 0.1 # Higher throughput is better
        })
        
        outcome = 0.0
        total_weight = 0.0
        for metric_name, value in metrics.items():
            # Sanitize input
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                continue
            weight = weights.get(metric_name, 0.0)
            outcome += weight * value
            total_weight += abs(weight)
        
        if total_weight > 0:
            outcome /= total_weight
        
        # Clamp outcome to prevent overflow in sigmoid
        outcome = max(-500.0, min(500.0, outcome))
        
        # Normalize to 0-1 range (sigmoid-like)
        return 1.0 / (1.0 + np.exp(-outcome))
    
    def _update_bayesian_samples(self, metrics: Dict[str, float]):
        """Update samples for Gaussian Process models"""
        with self._lock:
            X = [self.current_parameters[param] for param in sorted(self.tuning_parameters.keys())]
            self.X_samples.append(X)
            outcome = self._compute_outcome(metrics)
            for metric_name in self.performance_history.keys():
                self.y_samples[metric_name].append(metrics.get(metric_name, 0.0))
        
        # Retrain GP models if we have enough samples
        if len(self.X_samples) >= 10:
            self._train_gp_models()
    
    def _train_gp_models(self):
        """Train Gaussian Process models for each metric (simplified)"""
        # In a real implementation, we would use scikit-learn or GPy
        # For now, we'll just note that we have enough samples
        self.logger.debug(f"Would train GP models with {len(self.X_samples)} samples")
    
    def _bayesian_optimization(self) -> Dict[str, float]:
        """
        Suggest next parameters using Bayesian optimization (simplified)
        In reality, this would use acquisition functions on GP models
        """
        # For simplicity, we'll add small random perturbations to the best parameters so far
        if not self.outcome_history:
            return self.current_parameters.copy()
        
        # Find best parameters so far
        best_outcome = -float('inf')
        best_params = None
        for params, outcome in self.outcome_history:
            if outcome > best_outcome:
                best_outcome = outcome
                best_params = params
        
        if best_params is None:
            return self.current_parameters.copy()
        
        # Add Gaussian noise to best parameters
        new_params = {}
        for param_name, value in best_params.items():
            param_config = self.tuning_parameters[param_name]
            if param_config['type'] == 'float':
                noise = np.random.normal(0, (param_config['max'] - param_config['min']) * 0.1)
                new_value = value + noise
                new_value = max(param_config['min'], min(param_config['max'], new_value))
            else:  # int
                noise = np.random.randint(-2, 3)  # Small integer step
                new_value = value + noise
                new_value = max(param_config['min'], min(param_config['max'], new_value))
            new_params[param_name] = new_value
        
        return new_params
    
    def _random_search(self) -> Dict[str, float]:
        """Suggest random parameters within bounds"""
        new_params = {}
        for param_name, param_config in self.tuning_parameters.items():
            if param_config['type'] == 'float':
                new_value = np.random.uniform(param_config['min'], param_config['max'])
            else:  # int
                new_value = np.random.randint(param_config['min'], param_config['max'] + 1)
            new_params[param_name] = new_value
        return new_params
    
    def _grid_search(self) -> Dict[str, float]:
        """Suggest next parameters in a grid search pattern (simplified)"""
        # For simplicity, we'll just return a random point in the grid
        # A real grid search would iterate through all combinations
        return self._random_search()
    
    def _rule_based_tuning(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Adjust parameters based on simple rules"""
        new_params = self.current_parameters.copy()
        
        # Example rules:
        # If latency is high, decrease batch size or increase temperature for faster sampling
        # If quality is low, decrease temperature or increase top_p for more diversity
        
        latency = metrics.get('latency', 0.0)
        quality = metrics.get('quality', 0.0)
        cost = metrics.get('cost', 0.0)
        
        # Latency rules
        if latency > 1.0:  # High latency (seconds)
            if 'batch_size' in new_params:
                new_params['batch_size'] = max(
                    self.tuning_parameters['batch_size']['min'],
                    new_params['batch_size'] - 16
                )
            if 'temperature' in new_params:
                new_params['temperature'] = min(
                    self.tuning_parameters['temperature']['max'],
                    new_params['temperature'] + 0.1
                )
        
        # Quality rules
        if quality < 0.5:  # Low quality
            if 'temperature' in new_params:
                new_params['temperature'] = max(
                    self.tuning_parameters['temperature']['min'],
                    new_params['temperature'] - 0.1
                )
            if 'top_p' in new_params:
                new_params['top_p'] = min(
                    self.tuning_parameters['top_p']['max'],
                    new_params['top_p'] + 0.05
                )
        
        # Cost rules
        if cost > 0.1:  # High cost
            if 'batch_size' in new_params:
                new_params['batch_size'] = max(
                    self.tuning_parameters['batch_size']['min'],
                    int(new_params['batch_size'] * 0.9)
                )
        
        return new_params
    
    def _apply_constraints(self, params: Dict[str, float]) -> Dict[str, float]:
        """Apply parameter constraints"""
        constrained_params = {}
        for param_name, value in params.items():
            if param_name in self.tuning_parameters:
                param_config = self.tuning_parameters[param_name]
                if param_config['type'] == 'float':
                    constrained_value = max(param_config['min'], min(param_config['max'], float(value)))
                else:  # int
                    constrained_value = max(param_config['min'], min(param_config['max'], int(round(value))))
                constrained_params[param_name] = constrained_value
            else:
                constrained_params[param_name] = value
        return constrained_params
    
    def _smooth_parameters(self, new_params: Dict[str, float]) -> Dict[str, float]:
        """Smooth parameter changes to avoid drastic updates"""
        smoothed_params = {}
        smoothing_factor = self.config.get('smoothing_factor', 0.7)  # 0 = no change, 1 = full change
        
        for param_name, new_value in new_params.items():
            old_value = self.current_parameters.get(param_name, new_value)
            if isinstance(new_value, float):
                smoothed_value = smoothing_factor * new_value + (1 - smoothing_factor) * old_value
                smoothed_params[param_name] = smoothed_value
            else:  # int
                smoothed_value = int(smoothing_factor * new_value + (1 - smoothing_factor) * old_value)
                smoothed_params[param_name] = smoothed_value
        
        return smoothed_params
    
    def get_tuning_recommendations(self) -> Dict[str, Any]:
        """
        Get current tuning recommendations and status
        
        Returns:
            Dictionary with tuning information
        """
        return {
            'enabled': self.enabled,
            'tuning_strategy': self.tuning_strategy,
            'current_parameters': self.current_parameters.copy(),
            'parameter_history_size': {
                param: len(history) for param, history in self.parameter_history.items()
            },
            'performance_history_size': {
                metric: len(history) for metric, history in self.performance_history.items()
            },
            'outcome_history_size': len(self.outcome_history),
            'best_outcome_so_far': max([outcome for _, outcome in self.outcome_history]) if self.outcome_history else 0.0,
            'best_parameters_so_far': max(self.outcome_history, key=lambda x: x[1])[0] if self.outcome_history else {}
        }
    
    def _persist_state(self):
        """Persist the auto tuner's state to disk"""
        from universal_ai_optimizer.core.file_utils import atomic_write_json

        state = {
            'enabled': self.enabled,
            'tuning_strategy': self.tuning_strategy,
            'tuning_parameters': self.tuning_parameters,
            'current_parameters': self.current_parameters,
            'parameter_history': {k: list(v) for k, v in self.parameter_history.items()},
            'performance_history': {k: list(v) for k, v in self.performance_history.items()},
            'outcome_history': list(self.outcome_history)
        }

        data_dir = self._get_data_dir()
        if atomic_write_json(data_dir, "state.json", state):
            self.logger.debug("Persisted state for auto tuner")
    
    def _load_persisted_state(self):
        """Load the auto tuner's state from disk with atomic read and deque maxlen restore"""
        try:
            data_dir = self._get_data_dir()
            state_path = os.path.join(data_dir, "state.json")
            if not self._validate_path(state_path, data_dir):
                self.logger.warning("State path validation failed, refusing to load")
                return
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                self.enabled = state.get('enabled', self.enabled)
                self.tuning_strategy = state.get('tuning_strategy', self.tuning_strategy)
                self.tuning_parameters = state.get('tuning_parameters', self.tuning_parameters)
                self.current_parameters = state.get('current_parameters', self.current_parameters)
                
                # Restore history with maxlen preservation
                for param_name, history in state.get('parameter_history', {}).items():
                    self.parameter_history[param_name] = deque(history, maxlen=10000)
                
                for metric_name, history in state.get('performance_history', {}).items():
                    self.performance_history[metric_name] = deque(history, maxlen=10000)
                
                self.outcome_history = deque(state.get('outcome_history', []), maxlen=10000)
                
                self.logger.info("Loaded persisted state for auto tuner")
            else:
                self.logger.debug("No persisted state found for auto tuner")
        except Exception as e:
            self.logger.warning(f"Failed to load persisted auto tuner state: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get auto tuner metrics"""
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'tuning_strategy': self.tuning_strategy,
            'tuning_parameters': self.tuning_parameters,
            'current_parameters': self.current_parameters,
            'parameter_history_size': sum(len(v) for v in self.parameter_history.values()),
            'performance_history_size': sum(len(v) for v in self.performance_history.values()),
            'outcome_history_size': len(self.outcome_history),
            'best_outcome_so_far': max([outcome for _, outcome in self.outcome_history]) if self.outcome_history else 0.0
        })
        return base_metrics