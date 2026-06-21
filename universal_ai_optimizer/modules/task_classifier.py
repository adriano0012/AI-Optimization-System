"""
Task Classifier Module
Classifies the type of task from the prompt to inform routing decisions
"""

from typing import Dict, Any, Optional, List
import logging
import re
from ..core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)

class TaskClassifier(BaseOptimizerModule):
    """
    Classifies the task type of a given prompt
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.task_categories = self.config.get('task_categories', {
            'question_answering': ['what', 'who', 'where', 'when', 'why', 'how', 'explain', 'describe', 'define'],
            'summarization': ['summarize', 'summary', 'brief', 'tl;dr', 'in short'],
            'translation': ['translate', 'translation', 'in english', 'in spanish', 'in french', 'in german'],
            'code_generation': ['code', 'program', 'function', 'class', 'algorithm', 'script', 'debug', 'fix'],
            'creative_writing': ['story', 'poem', 'creative', 'imaginative', 'fiction', 'narrative', 'character'],
            'logical_reasoning': ['prove', 'logic', 'reason', 'deduce', 'infer', 'therefore', 'because', 'if', 'then'],
            'mathematical': ['calculate', 'compute', 'solve', 'equation', 'formula', 'math', 'algebra', 'calculus'],
            'comparison': ['compare', 'contrast', 'difference', 'similar', 'better', 'worse', 'vs', 'versus'],
            'opinion': ['opinion', 'think', 'believe', 'feel', 'should', 'would', 'could', 'might'],
            'instruction': ['how to', 'steps', 'guide', 'tutorial', 'follow', 'do this', 'make sure']
        })
        self.default_task = self.config.get('default_task', 'question_answering')
        
        self.logger.debug("TaskClassifier initialized")
    
    def process(self, prompt: str, context: Dict[str, Any], 
               model_adapter: Optional[Any] = None, 
               pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classify the task type of the prompt
        
        Args:
            prompt: Input prompt
            context: Context dictionary
            model_adapter: Model adapter (unused)
            pipeline_state: Current pipeline state
            
        Returns:
            Dictionary with task classification
        """
        if not self.enabled:
            return {}
        
        self._log_processing(len(prompt), len(str(context)))
        
        # Convert prompt to lowercase for matching
        prompt_lower = prompt.lower()
        
        # Score each task category
        scores = {}
        for task_type, keywords in self.task_categories.items():
            score = 0
            for keyword in keywords:
                # Count occurrences of the keyword in the prompt
                # Using word boundaries to avoid partial matches
                matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', prompt_lower))
                score += matches
            scores[task_type] = score
        
        # Find the task type with the highest score
        if max(scores.values()) > 0:
            best_task = max(scores, key=scores.get)
            confidence = scores[best_task] / sum(scores.values()) if sum(scores.values()) > 0 else 0.0
        else:
            # No keywords matched, use default
            best_task = self.default_task
            confidence = 0.0
        
        result = {
            'task_type': best_task,
            'task_confidence': confidence,
            'task_scores': scores
        }
        
        self.logger.info(f"Classified task as: {best_task} (confidence: {confidence:.3f})")
        return result
    
    def add_task_category(self, name: str, keywords: List[str]):
        """Add a new task category with associated keywords"""
        self.task_categories[name] = keywords
        self.logger.info(f"Added task category: {name}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get task classifier metrics"""
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'task_categories': list(self.task_categories.keys()),
            'default_task': self.default_task
        })
        return base_metrics