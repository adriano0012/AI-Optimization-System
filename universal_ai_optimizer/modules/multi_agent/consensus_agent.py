"""
Consensus Agent
Reaches consensus among multiple agent outputs
"""

from typing import Dict, Any, Optional, List
import logging
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ConsensusAgent(BaseAgent):
    """
    Agent that aggregates multiple outputs and finds consensus
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.min_agents = self.config.get('min_agents', 2)
        self.agreement_threshold = self.config.get('agreement_threshold', 0.6)
    
    def process(self, input_data: Any, context=None, model_adapter=None, pipeline_state=None) -> Dict[str, Any]:
        """
        Find consensus among multiple outputs
        
        Args:
            input_data: List of outputs from different agents, or single output
            context: Context with agent_scores, agent_outputs, etc.
        """
        context = context or {}
        
        # Get outputs from context or input
        if isinstance(input_data, list):
            outputs = input_data
        elif isinstance(input_data, dict) and 'agent_outputs' in input_data:
            outputs = list(input_data['agent_outputs'].values())
        else:
            outputs = [input_data]
        
        agent_scores = context.get('agent_scores', {})
        
        if len(outputs) < self.min_agents:
            return {
                'consensus_result': outputs[0] if outputs else None,
                'consensus_score': 0.5,
                'agents_agreed': len(outputs),
                'method': 'single_source'
            }
        
        # Find common elements
        consensus_score = self._calculate_consensus_score(outputs, agent_scores)
        
        # Select best output weighted by agent scores
        best_output = self._select_best_output(outputs, agent_scores)
        
        return {
            'consensus_result': best_output,
            'consensus_score': consensus_score,
            'agents_agreed': len(outputs),
            'method': 'weighted_selection',
            'agreement_reached': consensus_score >= self.agreement_threshold
        }
    
    def _calculate_consensus_score(self, outputs: List[Any], scores: Dict[str, float]) -> float:
        """Calculate how much agents agree"""
        if len(outputs) <= 1:
            return 1.0
        
        # Simple text similarity between outputs
        text_outputs = [str(o)[:500] for o in outputs]
        
        # Check for common keywords/phrases
        all_words = []
        for text in text_outputs:
            words = set(text.lower().split())
            all_words.append(words)
        
        if not all_words:
            return 0.0
        
        # Calculate Jaccard similarity between all pairs
        total_similarity = 0.0
        pairs = 0
        for i in range(len(all_words)):
            for j in range(i + 1, len(all_words)):
                intersection = len(all_words[i] & all_words[j])
                union = len(all_words[i] | all_words[j])
                if union > 0:
                    total_similarity += intersection / union
                    pairs += 1
        
        avg_similarity = total_similarity / pairs if pairs > 0 else 0.0
        
        # Factor in agent confidence scores
        if scores:
            avg_confidence = sum(scores.values()) / len(scores)
            return (avg_similarity + avg_confidence) / 2
        
        return avg_similarity
    
    def _select_best_output(self, outputs: List[Any], scores: Dict[str, float]) -> Any:
        """Select the best output based on agent scores with tiebreaker"""
        if not scores:
            return outputs[len(outputs) // 2]  # Return middle output
        
        # Map outputs to scores (approximate mapping)
        scored_outputs = []
        for i, output in enumerate(outputs):
            # Find matching score
            score = 0.5
            for agent_name, agent_score in scores.items():
                if isinstance(agent_score, (int, float)):
                    score = max(score, float(agent_score))
            scored_outputs.append((output, score))
        
        # Return highest scored, use index as tiebreaker (later = higher confidence)
        scored_outputs.sort(key=lambda x: (x[1], outputs.index(x[0])), reverse=True)
        return scored_outputs[0][0]
    
    def validate_input(self, input_data: Any) -> bool:
        return input_data is not None
