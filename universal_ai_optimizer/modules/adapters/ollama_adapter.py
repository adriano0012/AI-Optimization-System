"""
Ollama Adapter
Interface for Ollama's local models
"""

from typing import Dict, Any, Optional, List
import logging
import requests
from .base_adapter import BaseModelAdapter, GenerationResult

logger = logging.getLogger(__name__)

class OllamaAdapter(BaseModelAdapter):
    """
    Adapter for Ollama's local models
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.base_url = self.config.get('base_url', 'http://localhost:11434')
        self.model = self.config.get('model', 'llama2')
        
        self.logger.info(f"Initialized Ollama adapter for model: {self.model} at {self.base_url}")
    
    def generate(self, prompt: str, 
                max_tokens: Optional[int] = None,
                temperature: float = 0.7,
                top_p: float = 1.0,
                stop: Optional[List[str]] = None,
                **kwargs) -> Any:
        """
        Generate text using Ollama's API
        """
        try:
            # Prepare the request
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            # Add options if provided
            options = {}
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            if temperature is not None:
                options["temperature"] = temperature
            if top_p is not None:
                options["top_p"] = top_p
            if stop is not None:
                options["stop"] = stop
            
            if options:
                payload["options"] = options
            
            # Add any additional parameters
            payload.update(kwargs)
            
            # Make the request
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30, verify=True)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the generated text
            generated_text = result.get("response", "")
            
            return GenerationResult(generated_text)
            
        except Exception as e:
            self.logger.error(f"Ollama API error: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Ollama model"""
        try:
            # Try to get model info from Ollama
            response = requests.get(f"{self.base_url}/api/show",
                                  json={"name": self.model}, timeout=10, verify=True)
            if response.status_code == 200:
                model_info = response.json()
                return {
                    'provider': 'ollama',
                    'model': self.model,
                    'type': 'text-generation',
                    'supports_streaming': False,  # Ollama supports streaming but we use non-streaming for simplicity
                    'max_context_length': model_info.get('context_length', 2048),
                    'details': model_info
                }
        except Exception as e:
            self.logger.debug(f"Could not fetch model info from Ollama: {str(e)}")
        
        # Fallback
        return {
            'provider': 'ollama',
            'model': self.model,
            'type': 'text-generation',
            'supports_streaming': False,
            'max_context_length': 2048  # conservative estimate
        }
    
    def validate_config(self) -> bool:
        """Validate Ollama configuration"""
        # Ollama doesn't require an API key, but we should check if the service is reachable
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5, verify=True)
            return response.status_code == 200
        except Exception:
            self.logger.warning(f"Cannot reach Ollama service at {self.base_url}")
            return False