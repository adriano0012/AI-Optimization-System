"""
Base Agent for Multi-Agent System
Defines the interface that all agents must implement
"""

from abc import abstractmethod
from typing import Dict, Any, Optional
import logging

from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)


class BaseAgent(BaseOptimizerModule):
    """
    Abstract base class for all agents in the multi-agent system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_iterations = self.config.get('max_iterations', 10)
        self.has_async = False
        self._execution_count = 0
    
    @abstractmethod
    def process(self, input_data: Any,
               context: Optional[Dict[str, Any]] = None,
               model_adapter: Optional[Any] = None,
               pipeline_state: Optional[Dict[str, Any]] = None) -> Any:
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        return input_data is not None

    def _check_max_iterations(self) -> bool:
        self._execution_count += 1
        if self._execution_count > self.max_iterations:
            self.logger.warning(f"Reached max iterations ({self.max_iterations})")
            return False
        return True

    async def process_async(self, input_data: Any,
                            context: Optional[Dict[str, Any]] = None) -> Any:
        self.logger.debug("Async call delegated to synchronous process")
        return self.process(input_data, context)