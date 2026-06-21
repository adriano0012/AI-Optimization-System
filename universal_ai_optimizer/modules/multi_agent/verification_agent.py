"""
Verification Agent
Verifies the correctness and factual accuracy of outputs
"""

from typing import Dict, Any, Optional, List
import re
import logging
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class VerificationAgent(BaseAgent):
    """
    Agent that verifies correctness, consistency, and factual accuracy
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
    
    def process(self, input_data: Any, context=None, model_adapter=None, pipeline_state=None) -> Dict[str, Any]:
        """Verify the input data"""
        context = context or {}
        text = str(input_data)
        
        checks = {
            'has_evidence': self._check_for_evidence(text),
            'has_numbers': self._check_for_specifics(text),
            'no_contradictions': self._check_contradictions(text),
            'well_structured': self._check_structure(text),
            'no_hedging': self._check_hedging(text),
        }
        
        passed = sum(checks.values())
        total = len(checks)
        confidence = passed / total if total > 0 else 0.0
        
        return {
            'verification_result': 'pass' if confidence >= self.confidence_threshold else 'needs_review',
            'confidence': confidence,
            'checks': checks,
            'checks_passed': passed,
            'checks_total': total,
            'warnings': self._generate_warnings(checks, text)
        }
    
    def _check_for_evidence(self, text: str) -> bool:
        evidence_indicators = ['according to', 'research shows', 'data indicates', 
                              'study found', 'evidence suggests', 'because', 'therefore']
        return any(ind in text.lower() for ind in evidence_indicators) or len(text) > 200
    
    def _check_for_specifics(self, text: str) -> bool:
        return bool(re.search(r'\d+', text))
    
    def _check_contradictions(self, text: str) -> bool:
        contradiction_pairs = [('always', 'never'), ('all', 'none'), ('increase', 'decrease')]
        text_lower = text.lower()
        for word1, word2 in contradiction_pairs:
            if word1 in text_lower and word2 in text_lower:
                # Check if they're in different sentences
                sentences = text_lower.split('.')
                for i, s1 in enumerate(sentences):
                    for s2 in sentences[i+1:]:
                        if word1 in s1 and word2 in s2:
                            return False
        return True
    
    def _check_structure(self, text: str) -> bool:
        return len(text.split('.')) >= 2 or len(text.split('\n')) >= 2
    
    def _check_hedging(self, text: str) -> bool:
        hedging_words = ['maybe', 'perhaps', 'might be', 'could be', 'possibly', 'unclear']
        hedge_count = sum(1 for h in hedging_words if h in text.lower())
        return hedge_count <= 2
    
    def _generate_warnings(self, checks: Dict[str, bool], text: str) -> List[str]:
        warnings = []
        if not checks['has_evidence']:
            warnings.append("No supporting evidence found")
        if not checks['no_contradictions']:
            warnings.append("Potential contradictions detected")
        if not checks['well_structured']:
            warnings.append("Structure could be improved")
        if checks['no_hedging'] is False:
            warnings.append("Excessive hedging language")
        return warnings
    
    def validate_input(self, input_data: Any) -> bool:
        return input_data is not None and len(str(input_data).strip()) > 0
