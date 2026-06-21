"""
Base Optimizer Module
Abstract base class for all optimization modules
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseOptimizerModule(ABC):
    """
    Abstract base class that all optimization modules must inherit from
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    @abstractmethod
    def process(self, prompt: str, context: Dict[str, Any], 
               model_adapter: Optional[Any] = None, 
               pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process the input and return optimization results
        
        Args:
            prompt: Input prompt string
            context: Context dictionary
            model_adapter: Model adapter for execution (if needed)
            pipeline_state: Current state of the pipeline
            
        Returns:
            Dictionary with optimization results and metadata
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for this module
        
        Returns:
            Dictionary of metrics
        """
        return {
            'module': self.__class__.__name__,
            'config': self.config
        }
    
    def _log_processing(self, prompt_length: int, context_size: int):
        """Helper to log processing details"""
        self.logger.debug(
            f"Processing prompt (length: {prompt_length}) "
            f"with context (size: {context_size})"
        )