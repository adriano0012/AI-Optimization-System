"""
Learning Router Module
Routes tasks to the most suitable model or strategy based on learned performance
"""

import logging
import secrets
import time
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque
import numpy as np
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)


class LearningRouter(BaseOptimizerModule):
    """
    Learns from historical performance to route tasks to the best model/strategy
    Uses reinforcement learning and bandit algorithms for exploration-exploitation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.routing_type = self.config.get('routing_type', 'model')  # model, strategy, prompt
        
        # Exploration-exploitation parameters
        self.exploration_rate = self.config.get('exploration_rate', 0.1)  # Epsilon for epsilon-greedy
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.decay_rate = self.config.get('decay_rate', 0.995)
        
        # Performance tracking
        self.performance_history = defaultdict(lambda: deque(maxlen=10000))  # (model, task_type) -> list of rewards
        self.action_counts = defaultdict(int)  # (model, task_type) -> number of times selected
        self.total_rewards = defaultdict(float)  # (model, task_type) -> total reward received
        self.average_rewards = defaultdict(float)  # (model, task_type) -> average reward
        
        # Contextual bandit features (simplified)
        self.feature_vectors = defaultdict(list)  # task_features -> list of feature vectors
        self.reward_weights = defaultdict(lambda: np.zeros(10))  # task_type -> weights for features
        
        # Available options (models or strategies)
        self.available_options = self.config.get('available_options', [])
        self.option_features = {}  # option -> feature vector
        
        # For Thompson Sampling (Bayesian)
        self.alpha = defaultdict(lambda: defaultdict(float))  # (option, context) -> alpha parameter
        self.beta = defaultdict(lambda: defaultdict(float))   # (option, context) -> beta parameter
        
        # Load persisted state if available
        self._load_persisted_state()
        
        self.logger.debug(f"LearningRouter initialized for type: {self.routing_type}")
    
    def route(self, task_context: Dict[str, Any]) -> Any:
        """
        Route a task to the best option (model/strategy) based on learned performance
        
        Args:
            task_context: Dictionary containing task information (task_type, difficulty, etc.)
            
        Returns:
            The selected option (model name, strategy name, etc.)
        """
        if not self.enabled or not self.available_options:
            # Return default option if disabled or no options available
            return self.available_options[0] if self.available_options else None
        
        # Extract task features for contextual routing
        task_features = self._extract_task_features(task_context)
        
        # Select option based on exploration-exploitation strategy
        selected_option = self._select_option(task_features, task_context)
        
        self.logger.debug(f"Selected option: {selected_option} for task: {task_context.get('task_type', 'unknown')}")
        return selected_option
    
    def update_performance(self, option: Any, task_context: Dict[str, Any], reward: float):
        """
        Update performance history for an option
        
        Args:
            option: The option that was selected (model, strategy, etc.)
            task_context: The context of the task
            reward: The reward received (higher is better)
        """
        if not self.enabled:
            return
        
        task_type = task_context.get('task_type', 'unknown')
        key = (option, task_type)
        
        # Update counts and rewards
        self.action_counts[key] += 1
        self.total_rewards[key] += reward
        self.average_rewards[key] = self.total_rewards[key] / self.action_counts[key]
        
        # Update performance history
        self.performance_history[key].append(reward)
        
        # Update contextual bandit weights (simplified)
        task_features = self._extract_task_features(task_context)
        self._update_reward_weights(task_features, reward)
        
        # Update Thompson Sampling parameters
        self._update_thompson_sampling(option, task_context, reward)
        
        # Auto-persist every 100 updates
        total_updates = sum(self.action_counts.values())
        if total_updates % 100 == 0:
            self._persist_state()
        
        self.logger.debug(f"Updated performance for {option} with reward {reward}")
    
    def _extract_task_features(self, task_context: Dict[str, Any]) -> List[float]:
        """
        Extract numerical features from task context for contextual bandit
        
        Args:
            task_context: Dictionary containing task information
            
        Returns:
            List of numerical features
        """
        features = []
        
        # Task type one-hot encoding (simplified)
        task_types = ['question_answering', 'summarization', 'translation', 
                     'code_generation', 'creative_writing', 'logical_reasoning',
                     'mathematical', 'comparison', 'opinion', 'instruction']
        task_type = task_context.get('task_type', 'unknown')
        for tt in task_types:
            features.append(1.0 if task_type == tt else 0.0)
        
        # Difficulty encoding
        difficulties = ['easy', 'medium', 'hard']
        difficulty = task_context.get('difficulty', 'medium')
        for diff in difficulties:
            features.append(1.0 if difficulty == diff else 0.0)
        
        # Prompt features
        prompt = task_context.get('prompt', '')
        features.append(min(len(prompt) / 1000.0, 1.0))  # Normalized length
        features.append(prompt.count('?') / 10.0)  # Question marks
        features.append(prompt.count('.') / 20.0)  # Sentences
        
        # Context features
        context = task_context.get('context', {})
        features.append(min(len(str(context)) / 5000.0, 1.0))  # Context size
        features.append(len(context) if isinstance(context, dict) else 0)  # Context keys
        
        # Ensure we have a fixed length feature vector
        while len(features) < 10:
            features.append(0.0)
        features = features[:10]  # Truncate to 10 features
        
        return features
    
    def _select_option(self, task_features: List[float], task_context: Dict[str, Any]) -> Any:
        """
        Select an option using epsilon-greedy strategy
        
        Args:
            task_features: Numerical features of the task
            task_context: Task context dictionary
            
        Returns:
            Selected option
        """
        # Exploration: choose random option
        if secrets.randbelow(10000) / 10000.0 < self.exploration_rate:
            return secrets.choice(self.available_options)
        
        # Exploitation: choose option with highest estimated reward
        task_type = task_context.get('task_type', 'unknown')
        best_option = None
        best_reward = -float('inf')
        
        for option in self.available_options:
            key = (option, task_type)
            estimated_reward = self.average_rewards.get(key, 0.0)
            # Add uncertainty bonus for options with few trials
            uncertainty_bonus = 0.0
            if self.action_counts[key] > 0:
                # UCB1 style bonus (simplified)
                uncertainty_bonus = np.sqrt(2.0 * np.log(sum(self.action_counts.values()) + 1) / (self.action_counts[key] + 1e-6))
            total_reward = estimated_reward + uncertainty_bonus
            
            if total_reward > best_reward:
                best_reward = total_reward
                best_option = option
        
        return best_option if best_option is not None else secrets.choice(self.available_options)
    
    def _update_reward_weights(self, task_features: List[float], reward: float):
        """
        Update reward weights for contextual bandit (simplified linear model)
        
        Args:
            task_features: Feature vector of the task
            reward: Reward received
        """
        # For simplicity, we'll update weights for the task type (in reality, per option)
        # This is a very simplified version - in practice, we'd use linear regression or neural network
        task_type = 'unknown'  # We don't have task context here, but we could pass it
        # In a full implementation, we would have weights per option and task type
        
        # Simple gradient ascent step for linear reward prediction
        prediction = np.dot(self.reward_weights[task_type], task_features)
        error = reward - prediction
        for i in range(len(task_features)):
            self.reward_weights[task_type][i] += self.learning_rate * error * task_features[i]
    
    def _update_thompson_sampling(self, option: Any, task_context: Dict[str, Any], reward: float):
        """
        Update Thompson Sampling parameters (Beta-Bernoulli)
        
        Args:
            option: The selected option
            task_context: Task context
            reward: Reward received (0 or 1 for Bernoulli, but we'll adapt for continuous)
        """
        # For continuous rewards, we can use a Gamma-Poisson or discretize
        # For simplicity, we'll treat reward as binary (success if reward > 0.5)
        task_type = task_context.get('task_type', 'unknown')
        success = 1.0 if reward > 0.5 else 0.0
        
        self.alpha[option][task_type] += success
        self.beta[option][task_type] += (1.0 - success)
    
    def _thompson_sampling_select(self, task_context: Dict[str, Any]) -> Any:
        """
        Select option using Thompson Sampling
        
        Args:
            task_context: Task context
            
        Returns:
            Selected option
        """
        task_type = task_context.get('task_type', 'unknown')
        samples = {}
        
        for option in self.available_options:
            # Sample from Beta distribution for each option
            alpha_val = self.alpha[option].get(task_type, 1.0)
            beta_val = self.beta[option].get(task_type, 1.0)
            samples[option] = np.random.beta(alpha_val, beta_val)
        
        # Return option with highest sample
        return max(samples, key=samples.get)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of learned performance
        
        Returns:
            Dictionary with performance statistics
        """
        summary = {}
        for (option, task_type), avg_reward in self.average_rewards.items():
            if option not in summary:
                summary[option] = {}
            summary[option][task_type] = {
                'average_reward': avg_reward,
                'total_rewards': self.total_rewards[(option, task_type)],
                'action_count': self.action_counts[(option, task_type)],
                'recent_performance': list(self.performance_history[(option, task_type)])[-10:] if self.performance_history[(option, task_type)] else []
            }
        return summary
    
    def _persist_state(self):
        """Persist the learning router's state to disk using atomic write"""
        from universal_ai_optimizer.core.file_utils import atomic_write_json

        state = {
            'enabled': self.enabled,
            'routing_type': self.routing_type,
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate,
            'available_options': self.available_options,
            'action_counts': {f"{k[0]}|||{k[1]}": v for k, v in self.action_counts.items()},
            'total_rewards': {f"{k[0]}|||{k[1]}": v for k, v in self.total_rewards.items()},
            'average_rewards': {f"{k[0]}|||{k[1]}": v for k, v in self.average_rewards.items()},
        }

        if atomic_write_json("learning_router", "state.json", state):
            self.logger.debug("Persisted state for learning router")
    
    def _load_persisted_state(self):
        """Load the learning router's state from disk"""
        try:
            import os, json
            data_dir = "learning_router"
            state_path = os.path.join(data_dir, "state.json")
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                self.enabled = state.get('enabled', self.enabled)
                self.routing_type = state.get('routing_type', self.routing_type)
                self.exploration_rate = state.get('exploration_rate', self.exploration_rate)
                self.learning_rate = state.get('learning_rate', self.learning_rate)
                self.available_options = state.get('available_options', self.available_options)
                
                # Convert string keys back to tuple keys
                def _parse_tuple_key(k):
                    parts = k.split('|||')
                    return (parts[0], parts[1]) if len(parts) == 2 else (k, 'unknown')
                
                self.action_counts = defaultdict(int)
                for k, v in state.get('action_counts', {}).items():
                    self.action_counts[_parse_tuple_key(k)] = v
                
                self.total_rewards = defaultdict(float)
                for k, v in state.get('total_rewards', {}).items():
                    self.total_rewards[_parse_tuple_key(k)] = v
                
                self.average_rewards = defaultdict(float)
                for k, v in state.get('average_rewards', {}).items():
                    self.average_rewards[_parse_tuple_key(k)] = v
                
                self.logger.info("Loaded persisted state for learning router")
            else:
                self.logger.debug("No persisted state found for learning router")
        except Exception as e:
            self.logger.warning(f"Failed to load persisted learning router state: {e}")
    
    def save_state(self):
        """Public method to persist state to disk"""
        self._persist_state()
    
    def process(self, prompt: str, context: Dict[str, Any], 
                model_adapter: Optional[Any] = None, 
                pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Dummy process method to satisfy BaseOptimizerModule abstract class.
        The learning router is not used in the pipeline but for internal routing decisions.
        """
        return {}

    def get_metrics(self) -> Dict[str, Any]:
        """Get learning router metrics"""
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'routing_type': self.routing_type,
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate,
            'available_options_count': len(self.available_options),
            'unique_options_routed': len(set([opt for opt, _ in self.action_counts.keys()])),
            'unique_task_types_routed': len(set([task for _, task in self.action_counts.keys()])),
            'total_actions': sum(self.action_counts.values()),
            'performance_summary': self.get_performance_summary()
        })
        return base_metrics