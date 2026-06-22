"""
Anthropic Adapter
Interface for Anthropic's Claude models
"""

from typing import Dict, Any, Optional, List
import logging
import os
from .base_adapter import BaseModelAdapter, GenerationResult

logger = logging.getLogger(__name__)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic package not installed. Anthropic adapter will not work.")

class AnthropicAdapter(BaseModelAdapter):
    """
    Adapter for Anthropic's Claude models
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package is required for AnthropicAdapter")
        
        self.api_key = self.config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        self.model = self.config.get('model', 'claude-2')
        self.timeout = self.config.get('timeout', 30)
        
        self.client = None
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)
            
            masked_key = f"...{self.api_key[-4:]}" if len(self.api_key) > 4 else "***"
            self.logger.info(f"Initialized Anthropic adapter for model: {self.model} (key: {masked_key})")
        else:
            self.logger.warning("Anthropic adapter initialized without API key. generate() will fail.")
    
    def generate(self, prompt: str, 
                max_tokens: Optional[int] = None,
                temperature: float = 0.7,
                top_p: float = 1.0,
                stop: Optional[List[str]] = None,
                **kwargs) -> Any:
        """
        Generate text using Anthropic's API
        """
        if not self.client:
            raise ValueError("Anthropic API key not provided")
        
        try:
            # Prepare parameters
            params = {
                "model": self.model,
                "max_tokens": max_tokens or 1024,
                "temperature": temperature,
                "top_p": top_p,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            # Add stop sequences if provided
            if stop:
                params["stop_sequences"] = stop
            
            # Add any additional parameters
            params.update(kwargs)
            
            response = self.client.messages.create(**params)
            
            # Extract the generated text
            generated_text = response.content[0].text
            
            return GenerationResult(generated_text)
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Anthropic model"""
        return {
            'provider': 'anthropic',
            'model': self.model,
            'type': 'chat',
            'supports_streaming': False,  # Claude API doesn't support streaming in the same way
            'max_context_length': 100000 if 'claude-2' in self.model else 2048
        }
    
    def validate_config(self) -> bool:
        """Validate Anthropic configuration"""
        return bool(self.api_key)