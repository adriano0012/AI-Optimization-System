"""
OpenAI Adapter
Interface for OpenAI's GPT models
"""

from typing import Dict, Any, Optional, List
import logging
import os
from .base_adapter import BaseModelAdapter, GenerationResult

logger = logging.getLogger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed. OpenAI adapter will not work.")

class OpenAIAdapter(BaseModelAdapter):
    """
    Adapter for OpenAI's GPT models (GPT-3.5, GPT-4, etc.)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is required for OpenAIAdapter")
        
        self.api_key = self.config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        self.organization = self.config.get('organization')
        self.model = self.config.get('model', 'gpt-3.5-turbo')
        self.base_url = self.config.get('base_url')  # For custom endpoints
        self.timeout = self.config.get('timeout', 30)
        
        self.client = None
        if self.api_key:
            # Use client-based API instead of global state (SEC-001 fix)
            client_kwargs = {'api_key': self.api_key}
            if self.organization:
                client_kwargs['organization'] = self.organization
            if self.base_url:
                client_kwargs['base_url'] = self.base_url
            self.client = openai.OpenAI(**client_kwargs)
            
            masked_key = f"...{self.api_key[-4:]}" if len(self.api_key) > 4 else "***"
            self.logger.info(f"Initialized OpenAI adapter for model: {self.model} (key: {masked_key})")
        else:
            self.logger.warning("OpenAI adapter initialized without API key. generate() will fail.")
    
    def generate(self, prompt: str, 
                max_tokens: Optional[int] = None,
                temperature: float = 0.7,
                top_p: float = 1.0,
                stop: Optional[List[str]] = None,
                **kwargs) -> Any:
        """
        Generate text using OpenAI's API
        """
        if not self.client:
            raise ValueError("OpenAI API key not provided")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                timeout=self.timeout,
                **kwargs
            )
            
            # Extract the generated text
            generated_text = response.choices[0].message.content
            
            return GenerationResult(generated_text, response.usage)
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the OpenAI model"""
        return {
            'provider': 'openai',
            'model': self.model,
            'type': 'chat' if 'gpt' in self.model.lower() else 'completion',
            'supports_streaming': True,
            'max_context_length': 4096 if '3.5' in self.model else 8192 if '4' in self.model else 2048
        }
    
    def validate_config(self) -> bool:
        """Validate OpenAI configuration"""
        return bool(self.api_key)