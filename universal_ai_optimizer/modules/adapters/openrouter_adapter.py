"""
OpenRouter Adapter
Interface for OpenRouter's unified API
"""

from typing import Dict, Any, Optional, List
import logging
import requests
from .base_adapter import BaseModelAdapter, GenerationResult

logger = logging.getLogger(__name__)

class OpenRouterAdapter(BaseModelAdapter):
    """
    Adapter for OpenRouter's unified API (access to many models via one endpoint)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_key = self.config.get('api_key')
        self.base_url = self.config.get('base_url', 'https://openrouter.ai/api/v1')
        self.model = self.config.get('model', 'openai/gpt-3.5-turbo')
        
        self.logger.info(f"Initialized OpenRouter adapter for model: {self.model}")
    
    def generate(self, prompt: str, 
                max_tokens: Optional[int] = None,
                temperature: float = 0.7,
                top_p: float = 1.0,
                stop: Optional[List[str]] = None,
                **kwargs) -> Any:
        """
        Generate text using OpenRouter's API
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided")
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            
            if stop:
                payload["stop"] = stop
            
            # Add any additional parameters
            payload.update(kwargs)
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
                verify=True
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the generated text
            generated_text = result["choices"][0]["message"]["content"]
            
            return GenerationResult(generated_text, result.get("usage"))
            
        except Exception as e:
            self.logger.error(f"OpenRouter API error: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model via OpenRouter"""
        # OpenRouter doesn't have a direct model info endpoint in the same way
        # We'll return basic info based on the model string
        return {
            'provider': 'openrouter',
            'model': self.model,
            'type': 'chat' if 'chat' in self.model or 'gpt' in self.model else 'completion',
            'supports_streaming': True,
            'max_context_length': 4096  # conservative estimate, varies by model
        }
    
    def validate_config(self) -> bool:
        """Validate OpenRouter configuration"""
        if not self.api_key:
            self.logger.warning("OpenRouter API key not provided")
            return False
        return True