"""
Real Cost Tracking for Model Routing
Tracks actual costs from model API calls
"""

import time
import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
import json
import os
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)

class RealCostTracker(BaseOptimizerModule):
    """
    Tracks actual costs from model API calls for informed routing decisions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.cost_history = defaultdict(lambda: deque(maxlen=10000))  # model -> list of costs
        self.token_usage = defaultdict(lambda: deque(maxlen=10000))   # model -> list of token counts
        self.cost_per_token = defaultdict(float)  # model -> average cost per token
        self.total_cost = defaultdict(float)      # model -> total cost spent
        self.total_tokens = defaultdict(int)      # model -> total tokens used
        
        # Cost configuration (in USD per 1K tokens)
        self.base_costs = self.config.get('base_costs', {
            'gpt-3.5-turbo': 0.0015,   # $0.0015 per 1K tokens
            'gpt-4': 0.03,             # $0.03 per 1K tokens
            'gpt-4-turbo': 0.01,       # $0.01 per 1K tokens
            'claude-2': 0.008,         # $0.008 per 1K tokens
            'claude-instant': 0.0008,  # $0.0008 per 1K tokens
            'llama2:7b': 0.0,          # Local models have zero API cost
            'llama2:13b': 0.0,
            'llama2:70b': 0.0,
            'gemini-pro': 0.0005,      # $0.0005 per 1K tokens
        })
        
        # Load persisted cost data if available
        self._load_persisted_data()

    def process(self, prompt: str, context: Dict[str, Any], 
                model_adapter: Optional[Any] = None, 
                pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Dummy process method to satisfy BaseOptimizerModule abstract class.
        This tracker is not used in the pipeline but for metrics recording only.
        """
        return {}
    
    def record_usage(self, model_name: str, token_count: int, 
                    custom_cost_per_token: Optional[float] = None):
        """Record actual usage and cost for a model"""
        if not self.enabled:
            return
        
        # Calculate cost
        if custom_cost_per_token is not None:
            cost_per_token = custom_cost_per_token
        elif model_name in self.base_costs:
            cost_per_token = self.base_costs[model_name]
        else:
            # Default fallback
            cost_per_token = 0.002  # $0.002 per 1K tokens
        
        cost = (token_count / 1000.0) * cost_per_token
        
        # Update tracking
        self.cost_history[model_name].append({
            'cost': cost,
            'token_count': token_count,
            'timestamp': time.time()
        })
        
        self.token_usage[model_name].append(token_count)
        
        # Update running averages
        self.total_cost[model_name] += cost
        self.total_tokens[model_name] += token_count
        
        if self.total_tokens[model_name] > 0:
            self.cost_per_token[model_name] = self.total_cost[model_name] / (self.total_tokens[model_name] / 1000.0)
        
        # Persist periodically
        if len(self.cost_history[model_name]) % 100 == 0:
            self._persist_data()
        
        logger.debug(f"Recorded usage for {model_name}: {token_count} tokens, ${cost:.6f}")
    
    def get_cost_per_token(self, model_name: str) -> float:
        """Get the average cost per token for a model"""
        return self.cost_per_token.get(model_name, self.base_costs.get(model_name, 0.002))
    
    def get_estimated_cost(self, model_name: str, token_count: int) -> float:
        """Get estimated cost for a given token count"""
        cost_per_token = self.get_cost_per_token(model_name)
        return (token_count / 1000.0) * cost_per_token
    
    def get_cost_efficiency_score(self, model_name: str) -> float:
        """
        Get a cost efficiency score (higher is better)
        Based on inverse of cost per token, normalized
        """
        cost_per_token = self.get_cost_per_token(model_name)
        if cost_per_token <= 0:
            return 1.0  # Free/local models get max score
        
        # Normalize against most expensive model we track
        max_cost = max(self.base_costs.values()) if self.base_costs else 0.1
        normalized_cost = min(cost_per_token / max_cost, 1.0)
        return 1.0 - normalized_cost  # Invert so higher is better
    
    def get_reliability_score(self, model_name: str) -> float:
        """Get reliability score based on consistency of costs/performance"""
        history = list(self.cost_history[model_name])
        if len(history) < 10:
            return 0.5  # Not enough data
        
        # Calculate coefficient of variation of costs per token
        costs_per_token = [h['cost'] / (h['token_count'] / 1000.0) if h['token_count'] > 0 else 0 
                          for h in history if h['token_count'] > 0]
        
        if not costs_per_token or len(costs_per_token) < 2:
            return 0.5
        
        mean_cost = sum(costs_per_token) / len(costs_per_token)
        if mean_cost == 0:
            return 1.0
        
        variance = sum((x - mean_cost) ** 2 for x in costs_per_token) / len(costs_per_token)
        std_dev = variance ** 0.5
        cv = std_dev / mean_cost  # Coefficient of variation
        
        # Lower CV means more reliable
        # Convert to score: CV of 0 -> 1.0, CV of 1.0 -> 0.0, etc.
        return max(0.0, min(1.0, 1.0 - cv))
    
    def _persist_data(self):
        """Persist cost tracking data to disk"""
        from universal_ai_optimizer.core.file_utils import atomic_write_json

        data = {
            'total_cost': dict(self.total_cost),
            'total_tokens': dict(self.total_tokens),
            'cost_per_token': dict(self.cost_per_token),
            'history_samples': {
                model: list(history)[-100:]
                for model, history in self.cost_history.items()
            }
        }

        if atomic_write_json("cost_tracking_data", "cost_data.json", data):
            logger.debug("Persisted cost tracking data")
    
    def _load_persisted_data(self):
        """Load persisted cost tracking data"""
        try:
            data_path = "cost_tracking_data/cost_data.json"
            if os.path.exists(data_path):
                with open(data_path, 'r') as f:
                    data = json.load(f)
                
                self.total_cost = defaultdict(float, data.get('total_cost', {}))
                self.total_tokens = defaultdict(int, data.get('total_tokens', {}))
                self.cost_per_token = defaultdict(float, data.get('cost_per_token', {}))
                
                # Restore recent history
                history_samples = data.get('history_samples', {})
                for model, samples in history_samples.items():
                    self.cost_history[model] = deque(samples, maxlen=10000)
                    
                logger.info("Loaded persisted cost tracking data")
        except Exception as e:
            logger.warning(f"Failed to load persisted cost data: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cost tracking metrics"""
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'models_tracked': list(self.cost_history.keys()),
            'total_models': len(self.cost_history),
            'average_cost_per_token': dict(self.cost_per_token),
            'total_cost_usd': dict(self.total_cost),
            'total_tokens_used': dict(self.total_tokens),
        })
        return base_metrics