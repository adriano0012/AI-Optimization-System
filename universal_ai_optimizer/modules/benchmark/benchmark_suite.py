"""
Benchmark Suite
Standardized benchmarks for comparing optimization strategies.
"""

import logging
import random
import threading
import time
from typing import Dict, Any, Optional, List, Callable
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)


class BenchmarkTask:
    def __init__(self, name: str, prompt: str, context: Dict[str, Any],
                 expected_behavior: Optional[str] = None):
        self.name = name
        self.prompt = prompt
        self.context = context
        self.expected_behavior = expected_behavior


class BenchmarkResult:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.latency_ms = 0.0
        self.success = True
        self.output_length = 0
        self.error = None
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_name': self.task_name,
            'latency_ms': self.latency_ms,
            'success': self.success,
            'output_length': self.output_length,
            'error': self.error,
            'metadata': self.metadata,
        }


class BenchmarkSuite(BaseOptimizerModule):
    """
    Standardized benchmark suite for evaluating optimization performance.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self._lock = threading.Lock()

        self._tasks: List[BenchmarkTask] = self._get_default_tasks()
        self._results: List[BenchmarkResult] = []

    def _get_default_tasks(self) -> List[BenchmarkTask]:
        return [
            BenchmarkTask("simple_qa", "What is 2+2?", {'task_type': 'question_answering'}),
            BenchmarkTask("code_gen", "Write a function to reverse a string", {'task_type': 'code_generation'}),
            BenchmarkTask("creative", "Write a haiku about technology", {'task_type': 'creative_writing'}),
            BenchmarkTask("summarize", "Summarize the concept of machine learning in 3 sentences", {'task_type': 'summarization'}),
            BenchmarkTask("translate", "Translate 'hello world' to Spanish, French, and German", {'task_type': 'translation'}),
        ]

    def add_task(self, name: str, prompt: str, context: Dict[str, Any],
                 expected_behavior: Optional[str] = None):
        self._tasks.append(BenchmarkTask(name, prompt, context, expected_behavior))

    def run_benchmark(self, handler: Callable[[str, Dict], str],
                      tasks: Optional[List[BenchmarkTask]] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {'error': 'benchmark disabled'}

        tasks = tasks or self._tasks
        results = []
        start_time = time.time()

        for task in tasks:
            result = BenchmarkResult(task.name)
            task_start = time.time()
            try:
                output = handler(task.prompt, task.context)
                result.output_length = len(output) if output else 0
                result.success = True
            except Exception as e:
                result.success = False
                result.error = str(e)
            result.latency_ms = (time.time() - task_start) * 1000
            results.append(result)

        total_time = (time.time() - start_time) * 1000
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        latencies = [r.latency_ms for r in results]

        summary = {
            'total_tasks': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / max(len(results), 1),
            'total_time_ms': total_time,
            'avg_latency_ms': sum(latencies) / max(len(latencies), 1),
            'min_latency_ms': min(latencies) if latencies else 0,
            'max_latency_ms': max(latencies) if latencies else 0,
            'tasks': [r.to_dict() for r in results],
        }

        with self._lock:
            self._results.extend(results)

        self.logger.info(
            f"Benchmark complete: {len(successful)}/{len(results)} tasks passed, "
            f"avg latency: {summary['avg_latency_ms']:.1f}ms"
        )
        return summary

    def get_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in self._results]

    def get_comparison(self, results_a: Dict, results_b: Dict) -> Dict[str, Any]:
        return {
            'suite_a': {
                'success_rate': results_a.get('success_rate', 0),
                'avg_latency': results_a.get('avg_latency_ms', 0),
            },
            'suite_b': {
                'success_rate': results_b.get('success_rate', 0),
                'avg_latency': results_b.get('avg_latency_ms', 0),
            },
            'latency_improvement': (
                (results_a.get('avg_latency_ms', 1) - results_b.get('avg_latency_ms', 0))
                / max(results_a.get('avg_latency_ms', 1), 0.001) * 100
            ),
        }

    def process(self, prompt: str, context: Dict[str, Any],
                model_adapter=None, pipeline_state=None) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return {'benchmark_suite': 'ready', 'tasks': len(self._tasks)}

    def get_metrics(self) -> Dict[str, Any]:
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'total_tasks': len(self._tasks),
            'benchmarks_run': len(self._results),
        })
        return base_metrics
