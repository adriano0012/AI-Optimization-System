"""
Base Benchmark Module
Defines the interface for all benchmark types
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)

class BaseBenchmark(ABC):
    """
    Abstract base class for all benchmarks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def run_benchmark(self, optimizer_instance: Any, 
                     test_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Run the benchmark against an optimizer instance
        
        Args:
            optimizer_instance: The UniversalAIOptimizer instance to benchmark
            test_data: Optional test data to use for benchmarking
            
        Returns:
            Dictionary with benchmark results
        """
        pass
    
    def _measure_time(self, func, *args, **kwargs):
        """Helper to measure execution time"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, (end_time - start_time) * 1000  # return result and time in ms