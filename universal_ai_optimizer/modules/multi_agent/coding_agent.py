"""
Coding Agent
Specialized agent for code generation, debugging, and refactoring tasks
"""

from typing import Dict, Any, Optional, List
import re
import logging
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class CodingAgent(BaseAgent):
    """
    Agent specialized in coding tasks: generation, review, debugging, refactoring
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.supported_languages = self.config.get('supported_languages', 
            ['python', 'javascript', 'typescript', 'java', 'go', 'rust'])
    
    def process(self, input_data: Any, context=None, model_adapter=None, pipeline_state=None) -> Dict[str, Any]:
        """Process a coding task"""
        context = context or {}
        
        if not self.validate_input(input_data):
            return {'error': 'Invalid input', 'confidence': 0.0}
        
        task_type = self._detect_task_type(str(input_data))
        language = context.get('language', self._detect_language(str(input_data)))
        
        result = {
            'task_type': task_type,
            'language': language,
            'input': str(input_data)[:500],
            'analysis': self._analyze_code_request(str(input_data)),
            'suggestions': self._generate_suggestions(str(input_data), task_type),
            'confidence': 0.7
        }
        
        return result
    
    def _detect_task_type(self, text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in ['debug', 'error', 'fix', 'bug']):
            return 'debugging'
        elif any(kw in text_lower for kw in ['refactor', 'optimize', 'clean']):
            return 'refactoring'
        elif any(kw in text_lower for kw in ['review', 'check', 'analyze code']):
            return 'review'
        elif any(kw in text_lower for kw in ['write', 'create', 'implement', 'generate']):
            return 'generation'
        return 'general'
    
    def _detect_language(self, text: str) -> str:
        indicators = {
            'python': ['def ', 'import ', 'class ', 'self.', 'pip'],
            'javascript': ['const ', 'let ', 'var ', 'function', '=>', 'npm'],
            'typescript': [': string', ': number', ': boolean', 'interface ', 'type '],
            'java': ['public class', 'private ', 'System.out', 'import java'],
            'go': ['func ', 'package ', 'import (', 'fmt.'],
            'rust': ['fn ', 'let mut', 'impl ', 'pub struct'],
        }
        for lang, keywords in indicators.items():
            if any(kw in text for kw in keywords):
                return lang
        return 'unknown'
    
    def _analyze_code_request(self, text: str) -> str:
        if len(text) < 20:
            return "Brief request - may need more context"
        if '?' in text:
            return "Question about code - analysis response"
        return "Code task request identified"
    
    def _generate_suggestions(self, text: str, task_type: str) -> List[str]:
        suggestions = []
        if task_type == 'debugging':
            suggestions.extend([
                "Add logging to identify the issue",
                "Check input validation",
                "Verify edge cases"
            ])
        elif task_type == 'generation':
            suggestions.extend([
                "Define clear function signature",
                "Add error handling",
                "Include type hints"
            ])
        elif task_type == 'review':
            suggestions.extend([
                "Check for security vulnerabilities",
                "Verify error handling",
                "Assess performance"
            ])
        else:
            suggestions.append("Provide more context for better assistance")
        return suggestions
    
    def validate_input(self, input_data: Any) -> bool:
        return input_data is not None and len(str(input_data).strip()) > 0
