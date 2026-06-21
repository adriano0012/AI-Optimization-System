"""
Optimization Pipeline
Orchestrates the sequence of optimization modules
"""

from typing import Dict, Any, List, Optional, Union
import logging
from .base import BaseOptimizerModule

logger = logging.getLogger(__name__)

class OptimizationPipeline:
    """
    Pipeline that processes input through a series of optimization modules
    """
    
    def __init__(self, modules: List[BaseOptimizerModule], fail_fast: bool = False):
        self.modules = modules
        self.fail_fast = fail_fast
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized pipeline with {len(modules)} modules")
        self._module_map: Dict[str, BaseOptimizerModule] = {}
        for module in modules:
            self._module_map[module.__class__.__name__] = module
    
    def process(self, prompt: str, context: Optional[Dict[str, Any]] = None, 
               model_adapter: Optional[Any] = None) -> Dict[str, Any]:
        """
        Process input through all modules in the pipeline
        
        Args:
            prompt: Input prompt
            context: Context dictionary
            model_adapter: Model adapter for execution
            
        Returns:
            Dictionary with processed results and metadata
        """
        self.logger.debug("Starting pipeline processing")
        
        # Initialize result dictionary
        result = {
            'original_prompt': prompt,
            'optimized_prompt': prompt,
            'compressed_context': {},
            'cached_result': None,
            'verification_score': 0.0,
            'token_savings': 0.0,
            'resource_savings': {},
            'errors': [],
            'failed_modules': []
        }
        
        # Process through each module
        current_prompt = prompt
        current_context = context.copy() if context else {}
        cache_key = None
        
        for i, module in enumerate(self.modules):
            module_name = module.__class__.__name__
            self.logger.debug(f"Processing with module {i+1}/{len(self.modules)}: {module_name}")
            
            try:
                module_result = module.process(
                    prompt=current_prompt,
                    context=current_context,
                    model_adapter=model_adapter,
                    pipeline_state=result
                )
                
                # Update result with module output (only non-empty keys)
                if module_result:
                    for key, value in module_result.items():
                        if value is not None and value != '' and value != {}:
                            result[key] = value
                
                # Capture cache key for later storage
                if 'cache_key' in (module_result or {}):
                    cache_key = module_result['cache_key']
                
                # Update prompt and context for next module
                current_prompt = result.get('optimized_prompt', current_prompt)
                current_context = result.get('compressed_context', current_context)
                
            except Exception as e:
                self.logger.exception(f"Pipeline module {module_name} failed")
                error_info = {'module': module_name, 'error': str(e)}
                result['errors'].append(error_info)
                result['failed_modules'].append(module_name)
                if self.fail_fast:
                    raise
                continue
        
        # Store result in cache after all modules have processed (O(1) lookup)
        if cache_key:
            cache_module = self._module_map.get('CacheManager')
            if cache_module is not None:
                try:
                    cache_module.store_result(cache_key, result)
                except Exception:
                    self.logger.debug("Failed to store cache result")
        
        self.logger.debug("Pipeline processing completed")
        return result