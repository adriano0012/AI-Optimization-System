"""
Groq Adapter
Interface for Groq's high-speed inference
"""

from typing import Dict, Any, Optional, List
import logging
import requests
from .base_adapter import BaseModelAdapter, GenerationResult

logger = logging.getLogger(__name__)

class GroqAdapter(BaseModelAdapter):
    """
    Adapter for Groq's high-speed inference API
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_key = self.config.get('api_key')
        self.base_url = self.config.get('base_url', 'https://api.groq.com/openai/v1')
        self.model = self.config.get('model', 'llama2-70b-4096')
        
        self.logger.info(f"Initialized Groq adapter for model: {self.model}")
    
    def generate(self, prompt: str, 
                max_tokens: Optional[int] = None,
                temperature: float = 0.7,
                top_p: float = 1.0,
                stop: Optional[List[str]] = None,
                **kwargs) -> Any:
        """
        Generate text using Groq's API (OpenAI-compatible)
        """
        if not self.api_key:
            raise ValueError("Groq API key not provided")
        
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
            self.logger.error(f"Groq API error: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Groq model"""
        # Groq uses OpenAI-compatible API, so we can return similar info
        return {
            'provider': 'groq',
            'model': self.model,
            'type': 'chat',
            'supports_streaming': True,
            'max_context_length': 4096 if '4096' in self.model else 8192  # varies by model
        }
    
    def validate_config(self) -> bool:
        """Validate Groq configuration"""
        if not self.api_key:
            self.logger.warning("Groq API key not provided")
            return False
        return True