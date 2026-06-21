"""
Critic Agent
Evaluates and critiques outputs from other agents
"""

from typing import Dict, Any, Optional, List
import logging
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class CriticAgent(BaseAgent):
    """
    Agent that evaluates quality, correctness, and completeness of outputs
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.quality_threshold = self.config.get('quality_threshold', 0.6)
    
    def process(self, input_data: Any, context=None, model_adapter=None, pipeline_state=None) -> Dict[str, Any]:
        """Evaluate and critique the given input/output"""
        context = context or {}
        
        text = str(input_data)
        
        evaluation = {
            'strengths': self._find_strengths(text),
            'weaknesses': self._find_weaknesses(text),
            'quality_score': self._calculate_quality(text),
            'completeness_score': self._calculate_completeness(text, context),
            'clarity_score': self._calculate_clarity(text),
            'overall_confidence': 0.0,
            'recommendations': []
        }
        
        # Calculate overall confidence
        scores = [evaluation['quality_score'], evaluation['completeness_score'], evaluation['clarity_score']]
        evaluation['overall_confidence'] = sum(scores) / len(scores) if scores else 0.0
        
        # Generate recommendations
        if evaluation['quality_score'] < 0.7:
            evaluation['recommendations'].append("Improve quality of response")
        if evaluation['completeness_score'] < 0.7:
            evaluation['recommendations'].append("Add more detail or examples")
        if evaluation['clarity_score'] < 0.7:
            evaluation['recommendations'].append("Improve clarity and structure")
        
        return evaluation
    
    def _find_strengths(self, text: str) -> List[str]:
        strengths = []
        if len(text) > 100:
            strengths.append("Adequate length")
        if text.count('.') >= 3:
            strengths.append("Multiple sentences/ideas")
        if any(c.isdigit() for c in text):
            strengths.append("Contains specific data")
        if '\n' in text:
            strengths.append("Structured formatting")
        return strengths if strengths else ["Content present"]
    
    def _find_weaknesses(self, text: str) -> List[str]:
        weaknesses = []
        if len(text) < 50:
            weaknesses.append("Too brief")
        if text.lower().startswith("i don't") or text.lower().startswith("i cannot"):
            weaknesses.append("Expresses inability")
        if text.count('!') > 3:
            weaknesses.append("Excessive emphasis")
        if len(set(text.split())) / max(len(text.split()), 1) < 0.3:
            weaknesses.append("Repetitive vocabulary")
        return weaknesses if weaknesses else ["No major issues detected"]
    
    def _calculate_quality(self, text: str) -> float:
        score = 0.5
        if len(text) > 100: score += 0.1
        if len(text) > 500: score += 0.1
        if any(c.isupper() for c in text[1:]): score += 0.05
        if text.endswith('.'): score += 0.05
        return min(score, 1.0)
    
    def _calculate_completeness(self, text: str, context: Dict[str, Any]) -> float:
        score = 0.5
        if len(text) > 200: score += 0.15
        if len(text) > 500: score += 0.1
        if context.get('expected_length', 0) > 0:
            ratio = len(text) / context['expected_length']
            if 0.8 <= ratio <= 1.2: score += 0.15
        return min(score, 1.0)
    
    def _calculate_clarity(self, text: str) -> float:
        score = 0.5
        sentences = text.split('.')
        if 2 <= len(sentences) <= 20: score += 0.1
        avg_word_length = sum(len(w) for w in text.split()) / max(len(text.split()), 1)
        if 4 <= avg_word_length <= 7: score += 0.1
        if text[0:1].isupper(): score += 0.05
        return min(score, 1.0)
    
    def validate_input(self, input_data: Any) -> bool:
        return input_data is not None and len(str(input_data).strip()) > 0
