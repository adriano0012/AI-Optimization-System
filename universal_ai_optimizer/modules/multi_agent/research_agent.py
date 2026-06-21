"""
Research Agent
Specialized in information gathering and analysis
"""

from typing import Dict, Any, Optional, List
import logging
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """
    Agent specialized in research tasks: finding, analyzing, and synthesizing information
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.max_sources = self.config.get('max_sources', 10)
    
    def process(self, input_data: Any, context=None, model_adapter=None, pipeline_state=None) -> Dict[str, Any]:
        """Process a research task"""
        context = context or {}
        query = str(input_data)
        
        return {
            'query': query[:200],
            'research_type': self._classify_research(query),
            'key_topics': self._extract_topics(query),
            'analysis': self._analyze_query(query, context),
            'confidence': 0.65,
            'methodology': 'keyword_analysis'
        }
    
    def _classify_research(self, query: str) -> str:
        query_lower = query.lower()
        if any(kw in query_lower for kw in ['compare', 'vs', 'versus', 'difference']):
            return 'comparison'
        elif any(kw in query_lower for kw in ['how to', 'guide', 'tutorial']):
            return 'how_to'
        elif any(kw in query_lower for kw in ['what is', 'define', 'explain']):
            return 'definition'
        elif any(kw in query_lower for kw in ['why', 'reason', 'cause']):
            return 'explanatory'
        return 'general'
    
    def _extract_topics(self, text: str) -> List[str]:
        words = text.lower().split()
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 
                      'to', 'for', 'of', 'with', 'by', 'and', 'or', 'but', 'not', 'this',
                      'that', 'it', 'be', 'have', 'has', 'had', 'do', 'does', 'did'}
        meaningful = [w for w in words if w not in stop_words and len(w) > 2]
        # Deduplicate while preserving order
        seen = set()
        topics = []
        for w in meaningful:
            if w not in seen:
                seen.add(w)
                topics.append(w)
        return topics[:10]
    
    def _analyze_query(self, query: str, context: Dict[str, Any]) -> str:
        topics = self._extract_topics(query)
        return f"Research query identified with {len(topics)} key topics: {', '.join(topics[:5])}"
    
    def validate_input(self, input_data: Any) -> bool:
        return input_data is not None and len(str(input_data).strip()) > 5
