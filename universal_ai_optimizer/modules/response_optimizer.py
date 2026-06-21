"""
Response Optimizer Module
Optimizes the final output for clarity, conciseness, and relevance
"""

from typing import Dict, Any, Optional
import logging
import re
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)

class ResponseOptimizer(BaseOptimizerModule):
    """
    Response optimizer that refines the model output for better quality
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.compression = self.config.get('compression', True)
        self.rephrasing = self.config.get('rephrasing', self.config.get('disable', False))
        self.factuality_check = self.config.get('factuality_check', True)
        self.latency_target_ms = self.config.get('latency_target_ms', 1000.0)
        
        self.logger.debug("Response optimizer initialized")
    
    def process(self, prompt: str, context: Dict[str, Any], 
               model_adapter: Optional[Any] = None, 
               pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize the response from the execution engine
        
        Args:
            prompt: Original prompt
            context: Context dictionary
            model_adapter: Model adapter (unused in this module but required by interface)
            pipeline_state: Current pipeline state (should contain verification_result or execution_result)
            
        Returns:
            Dictionary with optimized response and metadata
        """
        if not self.enabled:
            return {}
        
        # Get the result to optimize from pipeline state
        # Prefer verification result if available, otherwise execution result
        result_to_optimize = None
        if pipeline_state:
            result_to_optimize = (
                pipeline_state.get('verification_result') or
                pipeline_state.get('execution_result')
            )
            # If execution_result is a dict with nested 'execution_result' key, unwrap it
            if isinstance(result_to_optimize, dict) and 'execution_result' in result_to_optimize:
                result_to_optimize = result_to_optimize['execution_result']
        
        if result_to_optimize is None:
            self.logger.warning("No result found in pipeline state for response optimization")
            return {}
        
        self._log_processing(len(prompt), len(str(context)))
        
        # Convert result to string if it's not already
        if not isinstance(result_to_optimize, str):
            # Handle objects with text attribute or convert to string
            if hasattr(result_to_optimize, 'text'):
                response_text = result_to_optimize.text
            else:
                response_text = str(result_to_optimize)
        else:
            response_text = result_to_optimize
        
        # Apply optimizations
        optimized_response = response_text
        
        if self.compression:
            optimized_response = self._compress_response(optimized_response)
        
        if self.rephrasing:
            optimized_response = self._rephrase_response(optimized_response)
        
        if self.factuality_check:
            # Note: In a full implementation, this would do actual factual checking
            # For now, it's a placeholder
            optimized_response = self._check_factuality(optimized_response, context)
        
        # Calculate optimization metrics
        original_length = len(response_text)
        optimized_length = len(optimized_response)
        compression_ratio = 1.0 - (optimized_length / max(original_length, 1)) if original_length > 0 else 0.0
        
        result = {
            'optimized_response': optimized_response,
            'response_compression_ratio': compression_ratio,
            'original_length': original_length,
            'optimized_length': optimized_length
        }
        
        self.logger.info(f"Response optimization: {compression_ratio*100:.1f}% size reduction")
        return result
    
    def _compress_response(self, response: str) -> str:
        """Compress response by removing redundancies and unnecessary words"""
        self.logger.debug("Applying response compression")
        # Remove extra whitespace
        response = re.sub(r'\s+', ' ', response).strip()
        # Remove common redundant phrases (simple examples)
        redundant_phrases = [
            r'\bIn other words\b',
            r'\bThat is to say\b',
            r'\bIt is important to note that\b',
            r'\bIt should be noted that\b'
        ]
        for phrase in redundant_phrases:
            response = re.sub(phrase, '', response, flags=re.IGNORECASE)
        # Clean up extra spaces again
        response = re.sub(r'\s+', ' ', response).strip()
        return response
    
    def _rephrase_response(self, response: str) -> str:
        """Rephrase response for clarity (placeholder)"""
        self.logger.debug("Applying response rephrasing (placeholder)")
        # In a real implementation, this would use a paraphrasing model
        # For now, just return the original
        return response
    
    def _check_factuality(self, response: str, context: Dict[str, Any]) -> str:
        """Check and potentially correct factual errors (placeholder)"""
        self.logger.debug("Applying factuality check (placeholder)")
        # In a real implementation, this would:
        # 1. Extract claims from response
        # 2. Verify against knowledge base or context
        # 3. Correct or flag inaccuracies
        # For now, just return the original
        return response
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get response optimizer metrics"""
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'compression': self.compression,
            'rephrasing': self.rephrasing,
            'factuality_check': self.factuality_check,
            'latency_target_ms': self.latency_target_ms
        })
        return base_metrics