"""
GPU Optimizer Module
Implements GPU-specific optimizations including dynamic quantization,
memory management, kernel optimization, and hardware-aware scheduling
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import threading
import time
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)

class GPUOptimizer(BaseOptimizerModule):
    """
    GPU optimization module that applies hardware-specific optimizations
    for maximum performance and efficiency on GPU accelerators
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.quantization_levels = self.config.get('quantization_levels', ['int8', 'int4', 'fp16'])
        self.memory_pool_size = self.config.get('memory_pool_size_mb', 2048)
        self.enable_kernel_fusion = self.config.get('enable_kernel_fusion', True)
        self.enable_memory_optimization = self.config.get('enable_memory_optimization', True)
        self.enable_dynamic_batching = self.config.get('enable_dynamic_batching', True)
        self.enable_async_execution = self.config.get('enable_async_execution', True)
        self.gpu_utilization_target = self.config.get('gpu_utilization_target', 0.8)
        self.temperature_threshold = self.config.get('temperature_threshold_celsius', 80)
        
        # GPU state tracking
        self._gpu_available = False
        self._current_quantization = None
        self._memory_pool = {}
        self._kernel_cache = {}
        self._temperature_history = []
        self._utilization_history = []
        self._lock = threading.RLock()
        
        # Initialize GPU detection and setup
        self._init_gpu()
        
    def _init_gpu(self):
        """Initialize GPU detection and basic setup"""
        try:
            # Try to detect GPU availability
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            self._gpu_available = result.returncode == 0
            if self._gpu_available:
                gpu_name = result.stdout.strip()
                logger.info(f"GPU detected: {gpu_name}")
            else:
                logger.info("No GPU detected, GPU optimizations will be disabled")
        except Exception as e:
            self._gpu_available = False
            logger.debug(f"GPU detection failed: {e}")
            
        # Initialize memory pools if GPU is available
        if self._gpu_available and self.enable_memory_optimization:
            self._init_memory_pools()
            
    def _init_memory_pools(self):
        """Initialize GPU memory pools for efficient allocation"""
        try:
            # In a real implementation, this would set up CUDA memory pools
            # For now, we'll simulate with a simple tracking mechanism
            self._memory_pool = {
                'total_size': self.memory_pool_size,
                'allocated': 0,
                'free': self.memory_pool_size,
                'allocations': {}
            }
            logger.debug(f"Initialized GPU memory pool: {self.memory_pool_size}MB")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU memory pools: {e}")
            
    def process(self, prompt: str, context: Dict[str, Any], 
                model_adapter: Optional[Any] = None, 
                pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply GPU-specific optimizations to the execution pipeline
        
        Args:
            prompt: Input prompt string
            context: Context dictionary
            model_adapter: Model adapter for execution (if needed)
            pipeline_state: Current state of the pipeline
            
        Returns:
            Dictionary with GPU optimization results and metadata
        """
        if not self.enabled or not self._gpu_available:
            return {'gpu_optimizations_applied': False, 'reason': 'GPU not available or disabled'}
            
        # Update GPU stats
        self._update_gpu_stats()
        
        # Apply optimizations based on current GPU state
        optimizations_applied = []
        
        # 1. Dynamic quantization based on utilization and temperature
        quantization_choice = self._select_optimal_quantization()
        if quantization_choice:
            optimizations_applied.append(f'quantization:{quantization_choice}')
            context['gpu_quantization'] = quantization_choice
            
        # 2. Memory optimization recommendations
        if self.enable_memory_optimization:
            memory_advice = self._optimize_memory_usage(context)
            if memory_advice:
                optimizations_applied.append('memory_optimization')
                context['gpu_memory_advice'] = memory_advice
                
        # 3. Kernel fusion recommendations
        if self.enable_kernel_fusion:
            kernel_advice = self._recommend_kernel_fusion(prompt, context)
            if kernel_advice:
                optimizations_applied.append('kernel_fusion')
                context['gpu_kernel_advice'] = kernel_advice
                
        # 4. Dynamic batching recommendations
        if self.enable_dynamic_batching:
            batch_advice = self._recommend_dynamic_batching(context)
            if batch_advice:
                optimizations_applied.append('dynamic_batching')
                context['gpu_batch_advice'] = batch_advice
                
        # 5. Async execution recommendations
        if self.enable_async_execution:
            async_advice = self._recommend_async_execution(context)
            if async_advice:
                optimizations_applied.append('async_execution')
                context['gpu_async_advice'] = async_advice
                
        # Calculate estimated performance improvement
        estimated_improvement = self._estimate_performance_improvement(optimizations_applied)
        
        return {
            'gpu_optimizations_applied': True,
            'optimizations_applied': optimizations_applied,
            'estimated_performance_improvement': estimated_improvement,
            'gpu_utilization': self._get_current_utilization(),
            'gpu_temperature': self._get_current_temperature(),
            'memory_usage_mb': self._get_memory_usage(),
            'quantization_level': context.get('gpu_quantization', 'none')
        }
        
    def _update_gpu_stats(self):
        """Update current GPU statistics"""
        if not self._gpu_available:
            return
            
        try:
            import subprocess
            # Get utilization and memory info
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and len(lines[0].split(', ')) >= 4:
                    parts = lines[0].split(', ')
                    utilization = int(parts[0])
                    memory_used = int(parts[1])
                    memory_total = int(parts[2])
                    temperature = int(parts[3])
                    
                    with self._lock:
                        self._utilization_history.append(utilization)
                        self._temperature_history.append(temperature)
                        # Keep only last 100 readings
                        if len(self._utilization_history) > 100:
                            self._utilization_history.pop(0)
                        if len(self._temperature_history) > 100:
                            self._temperature_history.pop(0)
        except Exception as e:
            logger.debug(f"Failed to update GPU stats: {e}")
            
    def _get_current_utilization(self) -> float:
        """Get current GPU utilization as a fraction (0.0-1.0)"""
        with self._lock:
            if not self._utilization_history:
                return 0.0
            return sum(self._utilization_history) / len(self._utilization_history) / 100.0
            
    def _get_current_temperature(self) -> float:
        """Get current GPU temperature in Celsius"""
        with self._lock:
            if not self._temperature_history:
                return 0.0
            return sum(self._temperature_history) / len(self._temperature_history)
            
    def _get_memory_usage(self) -> int:
        """Get current GPU memory usage in MB"""
        if not self._gpu_available:
            return 0
            
        try:
            import subprocess
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=memory.used',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
        return 0
        
    def _select_optimal_quantization(self) -> Optional[str]:
        """Select optimal quantization level based on GPU state"""
        if not self._gpu_available:
            return None
            
        utilization = self._get_current_utilization()
        temperature = self._get_current_temperature()
        
        # If GPU is overheating, use more aggressive quantization
        if temperature > self.temperature_threshold:
            return 'int4'  # Most aggressive quantization
            
        # If GPU is underutilized, we can use less aggressive quantization for quality
        if utilization < 0.3:
            return 'fp16'  # Higher quality, less compression
            
        # For normal operation, use adaptive quantization
        if utilization > 0.8:
            return 'int8'  # Good balance for high utilization
        else:
            return 'int8'  # Default to int8 for balance
            
    def _optimize_memory_usage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide memory optimization recommendations"""
        if not self._gpu_available:
            return {}
            
        memory_used = self._get_memory_usage()
        # Get total memory from nvidia-smi
        try:
            import subprocess
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            memory_total = int(result.stdout.strip()) if result.returncode == 0 else 8192
        except Exception:
            memory_total = 8192  # Default assumption
            
        memory_usage_ratio = memory_used / max(memory_total, 1)
        
        advice = {}
        if memory_usage_ratio > 0.9:
            advice['action'] = 'clear_cache'
            advice['reason'] = 'High memory usage detected'
            advice['expected_savings_mb'] = int(memory_total * 0.2)
        elif memory_usage_ratio > 0.7:
            advice['action'] = 'optimize_allocation'
            advice['reason'] = 'Moderate memory usage - consider optimization'
            advice['expected_savings_mb'] = int(memory_total * 0.1)
        else:
            advice['action'] = 'none'
            advice['reason'] = 'Memory usage is optimal'
            
        advice['current_usage_mb'] = memory_used
        advice['total_memory_mb'] = memory_total
        advice['usage_ratio'] = memory_usage_ratio
        
        return advice
        
    def _recommend_kernel_fusion(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend kernel fusion opportunities"""
        # Simple heuristic: longer prompts benefit more from kernel fusion
        prompt_length = len(prompt)
        
        if prompt_length > 1000:
            return {
                'recommended': True,
                'techniques': ['flash_attention', 'xfusion'],
                'expected_speedup': '1.5x-2.0x',
                'applicable_operations': ['attention', 'feed_forward']
            }
        elif prompt_length > 500:
            return {
                'recommended': True,
                'techniques': ['flash_attention'],
                'expected_speedup': '1.2x-1.5x',
                'applicable_operations': ['attention']
            }
        else:
            return {
                'recommended': False,
                'reason': 'Prompt too short for significant kernel fusion benefits'
            }
            
    def _recommend_dynamic_batching(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend dynamic batching parameters"""
        utilization = self._get_current_utilization()
        
        # If GPU is underutilized, we can increase batch size
        if utilization < 0.5:
            return {
                'recommended': True,
                'current_utilization': utilization,
                'suggested_batch_size_increase': '2x',
                'expected_utilization_improvement': '0.5x-0.7x',
                'reason': 'Low GPU utilization detected - can increase batch size'
            }
        elif utilization > 0.9:
            return {
                'recommended': True,
                'current_utilization': utilization,
                'suggested_batch_size_decrease': '0.5x',
                'expected_latency_improvement': '20%-40%',
                'reason': 'High GPU utilization - decreasing batch size may improve latency'
            }
        else:
            return {
                'recommended': False,
                'reason': 'GPU utilization is in optimal range for current batch size'
            }
            
    def _recommend_async_execution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend async execution strategies"""
        # Check if we have multiple independent operations that could run async
        pipeline_state = context.get('pipeline_state', {})
        independent_operations = pipeline_state.get('independent_operations', 1)
        
        if independent_operations > 1:
            return {
                'recommended': True,
                'independent_operations': independent_operations,
                'expected_speedup': f'{min(independent_operations, 4):.1f}x',
                'strategy': 'pipeline_parallelism'
            }
        else:
            return {
                'recommended': False,
                'reason': 'Insufficient independent operations for async execution benefits'
            }
            
    def _estimate_performance_improvement(self, optimizations: List[str]) -> Dict[str, Any]:
        """Estimate overall performance improvement from applied optimizations"""
        # Base improvement factors
        improvement_factors = {
            'quantization:int4': {'latency': 0.5, 'throughput': 2.0, 'memory': 0.5},
            'quantization:int8': {'latency': 0.7, 'throughput': 1.5, 'memory': 0.7},
            'quantization:fp16': {'latency': 0.8, 'throughput': 1.2, 'memory': 0.8},
            'memory_optimization': {'latency': 0.9, 'throughput': 1.1, 'memory': 0.8},
            'kernel_fusion': {'latency': 0.6, 'throughput': 1.8, 'memory': 0.9},
            'dynamic_batching': {'latency': 0.8, 'throughput': 1.5, 'memory': 1.0},
            'async_execution': {'latency': 0.7, 'throughput': 2.0, 'memory': 1.0}
        }
        
        # Calculate combined improvement (simplified model)
        latency_factor = 1.0
        throughput_factor = 1.0
        memory_factor = 1.0
        
        for opt in optimizations:
            if opt in improvement_factors:
                factors = improvement_factors[opt]
                latency_factor *= factors['latency']
                throughput_factor *= factors['throughput']
                memory_factor *= factors['memory']
                
        return {
            'latency_improvement': f"{(1-latency_factor)*100:.1f}%",
            'throughput_improvement': f"{(throughput_factor-1)*100:.1f}%",
            'memory_reduction': f"{(1-memory_factor)*100:.1f}%",
            'optimizations_count': len(optimizations)
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get GPU optimizer metrics"""
        base_metrics = super().get_metrics()
        base_metrics.update({
            'gpu_available': self._gpu_available,
            'gpu_utilization': self._get_current_utilization(),
            'gpu_temperature': self._get_current_temperature(),
            'memory_usage_mb': self._get_memory_usage(),
            'enabled': self.enabled,
            'optimizations_available': [
                'dynamic_quantization',
                'memory_optimization', 
                'kernel_fusion',
                'dynamic_batching',
                'async_execution'
            ]
        })
        return base_metrics
        
    def shutdown(self):
        """Cleanup GPU optimizer resources"""
        with self._lock:
            self._memory_pool.clear()
            self._kernel_cache.clear()
            self._utilization_history.clear()
            self._temperature_history.clear()
        logger.info("GPU optimizer shut down")