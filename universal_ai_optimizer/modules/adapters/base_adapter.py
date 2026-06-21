"""
Base Adapter for LLM Providers
Defines the interface that all model adapters must implement
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class GenerationResult:
    """Standardized result object returned by all adapter generate() calls."""
    def __init__(self, text, usage=None):
        self.text = text
        self.usage = usage
        if usage is None:
            self.token_count = len(text.split())
        elif hasattr(usage, "total_tokens"):
            self.token_count = usage.total_tokens
        elif isinstance(usage, dict):
            self.token_count = usage.get("total_tokens", len(text.split()))
        else:
            self.token_count = len(text.split())


class BaseModelAdapter(ABC):
    """
    Abstract base class for all LLM adapters
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def generate(self, prompt: str, 
                max_tokens: Optional[int] = None,
                temperature: float = 0.7,
                top_p: float = 1.0,
                stop: Optional[List[str]] = None,
                **kwargs) -> GenerationResult:
        """
        Generate text from the model
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: List of stop sequences
            **kwargs: Additional provider-specific parameters
            
        Returns:
            GenerationResult with generated text and usage info
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model
        
        Returns:
            Dictionary with model information (name, version, capabilities, etc.)
        """
        pass
    
    def validate_config(self) -> bool:
        """
        Validate the adapter configuration
        
        Returns:
            True if configuration is valid, False otherwise
        """
        return True