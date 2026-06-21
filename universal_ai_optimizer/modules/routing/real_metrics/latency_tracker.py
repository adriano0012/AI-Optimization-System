"""
Real Latency Tracking for Model Routing
Tracks actual latency from model API calls
"""

import time
import threading
import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
import statistics
import json
import os
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)

class RealLatencyTracker(BaseOptimizerModule):
    """
    Tracks actual latency from model API calls for informed routing decisions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.latency_history = defaultdict(lambda: deque(maxlen=10000))  # model -> list of latencies
        self.latency_stats = defaultdict(dict)  # model -> {p50, p95, p99, mean, std}
        self.min_latency = defaultdict(float)   # model -> minimum observed latency
        self.max_latency = defaultdict(float)   # model -> maximum observed latency
        
        # For sliding window calculations
        self.window_size = self.config.get('window_size', 1000)  # Last N requests
        self.update_interval = self.config.get('stats_update_interval', 100)  # Update stats every N requests
        self.request_counts = defaultdict(int)  # model -> request count since last stats update
        
        # Thread safety for concurrent updates
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
    
    def record_latency(self, model_name: str, latency_ms: float):
        """Record actual latency for a model"""
        if not self.enabled:
            return
        
        with self._lock:
            # Add to history
            self.latency_history[model_name].append({
                'latency_ms': latency_ms,
                'timestamp': time.time()
            })
            
            # Update min/max
            if self.min_latency[model_name] == 0 or latency_ms < self.min_latency[model_name]:
                self.min_latency[model_name] = latency_ms
            
            if latency_ms > self.max_latency[model_name]:
                self.max_latency[model_name] = latency_ms
            
            # Update request count and potentially recalculate stats
            self.request_counts[model_name] += 1
            if self.request_counts[model_name] >= self.update_interval:
                self._update_stats(model_name)
                self.request_counts[model_name] = 0
            
            # Persist periodically
            total_requests = sum(self.request_counts.values())
            if total_requests % 500 == 0:
                self._persist_data()
        
        logger.debug(f"Recorded latency for {model_name}: {latency_ms:.2f}ms")
    
    def _update_stats(self, model_name: str):
        """Update latency statistics for a model"""
        history = list(self.latency_history[model_name])
        if not history:
            return
        
        # Extract latency values from recent history
        recent_latencies = [h['latency_ms'] for h in history[-self.window_size:]]
        
        if len(recent_latencies) < 2:
            # Not enough data for meaningful stats
            self.latency_stats[model_name] = {
                'mean': recent_latencies[0] if recent_latencies else 0.0,
                'p50': recent_latencies[0] if recent_latencies else 0.0,
                'p95': recent_latencies[0] if recent_latencies else 0.0,
                'p99': recent_latencies[0] if recent_latencies else 0.0,
                'std': 0.0,
                'sample_size': len(recent_latencies)
            }
            return
        
        # Calculate statistics
        mean_latency = statistics.mean(recent_latencies)
        try:
            std_latency = statistics.stdev(recent_latencies) if len(recent_latencies) >= 2 else 0.0
        except statistics.StatisticsError:
            std_latency = 0.0
        
        sorted_latencies = sorted(recent_latencies)
        n = len(sorted_latencies)
        
        self.latency_stats[model_name] = {
            'mean': mean_latency,
            'p50': sorted_latencies[n // 2],
            'p95': sorted_latencies[int(n * 0.95)],
            'p99': sorted_latencies[int(n * 0.99)],
            'std': std_latency,
            'sample_size': n
        }
        
        logger.debug(f"Updated latency stats for {model_name}: "
                    f"p50={self.latency_stats[model_name]['p50']:.2f}ms, "
                    f"p95={self.latency_stats[model_name]['p95']:.2f}ms")
    
    def get_latency(self, model_name: str, percentile: str = 'p50') -> float:
        """
        Get latency for a model at a specific percentile
        percentile: 'p50' (median), 'p95', 'p99', 'mean', 'min', 'max'
        """
        if not self.enabled or model_name not in self.latency_history:
            # Return a reasonable default
            return 1000.0  # 1 second default
        
        # Ensure stats are up to date
        if self.request_counts[model_name] > 0:
            self._update_stats(model_name)
        
        stats = self.latency_stats.get(model_name, {})
        if not stats:
            # Fallback to raw data if no stats available
            history = list(self.latency_history[model_name])
            if not history:
                return 1000.0
            latencies = [h['latency_ms'] for h in history]
            if percentile == 'mean':
                return statistics.mean(latencies)
            elif percentile == 'p50':
                return statistics.median(latencies)
            elif percentile == 'p95':
                sorted_lat = sorted(latencies)
                return sorted_lat[int(len(sorted_lat) * 0.95)]
            elif percentile == 'p99':
                sorted_lat = sorted(latencies)
                return sorted_lat[int(len(sorted_lat) * 0.99)]
            elif percentile == 'min':
                return min(latencies)
            elif percentile == 'max':
                return max(latencies)
            else:
                return statistics.mean(latencies)
        
        return stats.get(percentile, stats.get('mean', 0.0))
    
    def get_latency_efficiency_score(self, model_name: str) -> float:
        """
        Get a latency efficiency score (higher is better)
        Based on inverse of latency, normalized
        """
        # Use p95 latency for conservative estimate
        p95_latency = self.get_latency(model_name, 'p95')
        if p95_latency <= 0:
            return 1.0
        
        # Normalize against a reasonable maximum (e.g., 10 seconds)
        max_latency = 10000.0  # 10 seconds in ms
        normalized_latency = min(p95_latency / max_latency, 1.0)
        return 1.0 - normalized_latency  # Invert so higher is better
    
    def get_consistency_score(self, model_name: str) -> float:
        """Get latency consistency score (lower std dev is better)"""
        stats = self.latency_stats.get(model_name, {})
        if not stats or stats.get('sample_size', 0) < 2:
            return 0.5
        
        mean = stats.get('mean', 0.0)
        std = stats.get('std', 0.0)
        
        if mean == 0:
            return 1.0 if std == 0 else 0.0
        
        # Coefficient of variation
        cv = std / mean if mean > 0 else float('inf')
        # Convert to score: CV of 0 -> 1.0, CV of 1.0 -> 0.0, etc.
        return max(0.0, min(1.0, 1.0 - cv))
    
    def _persist_data(self):
        """Persist latency tracking data to disk"""
        from universal_ai_optimizer.core.file_utils import atomic_write_json

        history_data = {
            model: list(history)[-100:]
            for model, history in self.latency_history.items()
        }

        data = {
            'latency_stats': dict(self.latency_stats),
            'min_latency': dict(self.min_latency),
            'max_latency': dict(self.max_latency),
            'history_samples': history_data
        }

        if atomic_write_json("latency_tracking_data", "latency_data.json", data):
            logger.debug("Persisted latency tracking data")
    
    def _load_persisted_data(self):
        """Load persisted latency tracking data"""
        try:
            data_path = "latency_tracking_data/latency_data.json"
            if os.path.exists(data_path):
                with open(data_path, 'r') as f:
                    data = json.load(f)
                
                self.latency_stats = defaultdict(dict, data.get('latency_stats', {}))
                self.min_latency = defaultdict(float, data.get('min_latency', {}))
                self.max_latency = defaultdict(float, data.get('max_latency', {}))
                
                # Restore recent history
                history_samples = data.get('history_samples', {})
                for model, samples in history_samples.items():
                    self.latency_history[model] = deque(samples, maxlen=10000)
                    
                logger.info("Loaded persisted latency tracking data")
        except Exception as e:
            logger.warning(f"Failed to load persisted latency data: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get latency tracking metrics"""
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'models_tracked': list(self.latency_history.keys()),
            'total_models': len(self.latency_history),
            'latency_stats': {model: stats.get('p95', 0.0) 
                             for model, stats in self.latency_stats.items()},
            'min_latency': dict(self.min_latency),
            'max_latency': dict(self.max_latency),
        })
        return base_metrics