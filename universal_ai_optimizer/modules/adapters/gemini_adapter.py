"""
Gemini Adapter
Interface for Google's Gemini models
"""

from typing import Dict, Any, Optional, List
import logging
from .base_adapter import BaseModelAdapter, GenerationResult

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google GenerativeAI package not installed. Gemini adapter will not work.")

class GeminiAdapter(BaseModelAdapter):
    """
    Adapter for Google's Gemini models
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenerativeAI package is required for GeminiAdapter")
        
        self.api_key = self.config.get('api_key')
        self.model_name = self.config.get('model', 'gemini-pro')
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        else:
            self.model = None
        
        self.logger.info(f"Initialized Gemini adapter for model: {self.model_name}")
    
    def generate(self, prompt: str, 
                max_tokens: Optional[int] = None,
                temperature: float = 0.7,
                top_p: float = 1.0,
                stop: Optional[List[str]] = None,
                **kwargs) -> Any:
        """
        Generate text using Gemini's API
        """
        if not self.model:
            raise ValueError("Gemini API key not provided")
        
        try:
            # Prepare generation config
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            if stop:
                generation_config["stop_sequences"] = stop
            
            # Add any additional parameters
            generation_config.update(kwargs)
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(**generation_config)
            )
            
            # Extract the generated text
            generated_text = response.text
            
            return GenerationResult(generated_text)
            
        except Exception as e:
            self.logger.error(f"Gemini API error: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Gemini model"""
        return {
            'provider': 'google',
            'model': self.model_name,
            'type': 'text-generation',
            'supports_streaming': False,  # Gemini API has streaming but we'll keep simple for now
            'max_context_length': 32768 if 'pro' in self.model_name else 8192  # approximate
        }
    
    def validate_config(self) -> bool:
        """Validate Gemini configuration"""
        if not self.api_key:
            self.logger.warning("Gemini API key not provided")
            return False
        return True