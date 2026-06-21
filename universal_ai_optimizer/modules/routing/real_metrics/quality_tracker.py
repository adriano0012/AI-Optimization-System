"""
Quality Tracking for Model Routing
Tracks quality scores from benchmarks and human evaluation
"""

import time
import threading
import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
import json
import os
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)

class QualityTracker(BaseOptimizerModule):
    """
    Tracks quality scores from benchmarks (MMLU, HumanEval, etc.) and human feedback
    for informed routing decisions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.quality_history = defaultdict(lambda: deque(maxlen=10000))  # model -> list of quality scores
        self.benchmark_scores = defaultdict(dict)  # model -> {benchmark_name: score}
        self.overall_quality = defaultdict(float)  # model -> weighted average quality score
        self.benchmark_weights = self.config.get('benchmark_weights', {
            'mmlu': 0.25,
            'gpqa': 0.20,
            'humaneval': 0.15,
            'swe_bench': 0.15,
            'longbench': 0.10,
            'hallucination': 0.10,
            'aider': 0.05
        })
        
        # For sliding window calculations
        self.window_size = self.config.get('window_size', 1000)  # Last N evaluations
        self.update_interval = self.config.get('score_update_interval', 50)  # Update scores every N evaluations
        self.evaluation_counts = defaultdict(int)  # model -> evaluation count since last score update
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load persisted data
        self._load_persisted_data()

    def process(self, prompt: str, context: Dict[str, Any], 
                model_adapter: Optional[Any] = None, 
                pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Dummy process method to satisfy BaseOptimizerModule abstract class.
        This tracker is not used in the pipeline but for metrics recording only.
        """
        return {}
    
    def record_quality(self, model_name: str, benchmark_name: str, score: float):
        """Record a quality score from a benchmark or human evaluation"""
        if not self.enabled:
            return
        
        with self._lock:
            # Validate score is between 0 and 1
            score = max(0.0, min(1.0, score))
            
            # Record in history
            self.quality_history[model_name].append({
                'benchmark': benchmark_name,
                'score': score,
                'timestamp': time.time()
            })
            
            # Update benchmark-specific score
            self.benchmark_scores[model_name][benchmark_name] = score
            
            # Update evaluation count and potentially recalculate overall score
            self.evaluation_counts[model_name] += 1
            if self.evaluation_counts[model_name] >= self.update_interval:
                self._update_overall_score(model_name)
                self.evaluation_counts[model_name] = 0
            
            # Persist periodically
            total_evaluations = sum(self.evaluation_counts.values())
            if total_evaluations % 200 == 0:
                self._persist_data()
        
        logger.debug(f"Recorded quality for {model_name} on {benchmark_name}: {score:.3f}")
    
    def _update_overall_score(self, model_name: str):
        """Update the overall quality score for a model based on benchmark weights"""
        scores = self.benchmark_scores[model_name]
        if not scores:
            self.overall_quality[model_name] = 0.0
            return
        
        # Calculate weighted average
        weighted_sum = 0.0
        total_weight = 0.0
        for benchmark, weight in self.benchmark_weights.items():
            if benchmark in scores:
                weighted_sum += scores[benchmark] * weight
                total_weight += weight
        
        # If we have benchmarks not in our weights, give them equal remaining weight
        remaining_benchmarks = set(scores.keys()) - set(self.benchmark_weights.keys())
        if remaining_benchmarks:
            remaining_weight = (1.0 - total_weight) / len(remaining_benchmarks) if total_weight < 1.0 else 0.0
            for benchmark in remaining_benchmarks:
                weighted_sum += scores[benchmark] * remaining_weight
                total_weight += remaining_weight
        
        # Normalize if total weight is not 1.0 (shouldn't happen, but just in case)
        if total_weight > 0:
            self.overall_quality[model_name] = weighted_sum / total_weight
        else:
            self.overall_quality[model_name] = 0.0
        
        logger.debug(f"Updated overall quality for {model_name}: {self.overall_quality[model_name]:.3f}")
    
    def get_quality_score(self, model_name: str) -> float:
        """Get the overall quality score for a model"""
        # Ensure score is up to date
        if self.evaluation_counts[model_name] > 0:
            self._update_overall_score(model_name)
        
        return self.overall_quality.get(model_name, 0.0)
    
    def get_benchmark_score(self, model_name: str, benchmark_name: str) -> Optional[float]:
        """Get a specific benchmark score for a model"""
        return self.benchmark_scores[model_name].get(benchmark_name)
    
    def get_quality_trend(self, model_name: str, window: int = 100) -> Optional[float]:
        """Get the quality trend (improving/declining) over recent evaluations"""
        history = list(self.quality_history[model_name])
        if len(history) < 2:
            return None
        
        recent = history[-window:] if len(history) >= window else history
        if len(recent) < 2:
            return None
        
        # Calculate simple linear trend on scores
        scores = [h['score'] for h in recent]
        n = len(scores)
        if n < 2:
            return None
        
        # Calculate slope of least squares line
        sum_x = sum(range(n))
        sum_y = sum(scores)
        sum_xy = sum(i * scores[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
        return slope  # Positive means improving, negative means declining
    
    def _persist_data(self):
        """Persist quality tracking data to disk"""
        from universal_ai_optimizer.core.file_utils import atomic_write_json

        history_data = {
            model: list(history)[-100:]
            for model, history in self.quality_history.items()
        }

        data = {
            'benchmark_scores': dict(self.benchmark_scores),
            'overall_quality': dict(self.overall_quality),
            'history_samples': history_data
        }

        if atomic_write_json("quality_tracking_data", "quality_data.json", data):
            logger.debug("Persisted quality tracking data")
    
    def _load_persisted_data(self):
        """Load persisted quality tracking data"""
        try:
            data_path = "quality_tracking_data/quality_data.json"
            if os.path.exists(data_path):
                with open(data_path, 'r') as f:
                    data = json.load(f)
                
                self.benchmark_scores = defaultdict(dict, data.get('benchmark_scores', {}))
                self.overall_quality = defaultdict(float, data.get('overall_quality', {}))
                
                # Restore recent history
                history_samples = data.get('history_samples', {})
                for model, samples in history_samples.items():
                    self.quality_history[model] = deque(samples, maxlen=10000)
                    
                logger.info("Loaded persisted quality tracking data")
        except Exception as e:
            logger.warning(f"Failed to load persisted quality data: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get quality tracking metrics"""
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'models_tracked': list(self.quality_history.keys()),
            'total_models': len(self.quality_history),
            'overall_quality_scores': dict(self.overall_quality),
            'benchmark_weights': self.benchmark_weights,
            # Show a sample of benchmark scores for the first few models
            'sample_benchmark_scores': {
                model: dict(list(scores.items())[:3])  # First 3 benchmarks
                for model, scores in list(self.benchmark_scores.items())[:3]
            } if self.benchmark_scores else {}
        })
        return base_metrics