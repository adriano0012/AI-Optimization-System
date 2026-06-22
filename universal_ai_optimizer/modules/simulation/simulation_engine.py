"""
Simulation Engine
Simulates user load for stress testing and capacity planning.
"""

import logging
import random
import threading
import time
from collections import deque
from typing import Dict, Any, Optional, List, Callable
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)


class SimulatedUser:
    """Represents a simulated user with configurable behavior."""

    def __init__(self, user_id: int, profile: Optional[Dict[str, Any]] = None):
        self.user_id = user_id
        self.profile = profile or {}
        self.request_count = 0
        self.total_latency = 0.0
        self.errors = 0
        self.created_at = time.time()

    def generate_request(self, prompt_pool: List[str]) -> Dict[str, Any]:
        prompt = random.choice(prompt_pool) if prompt_pool else f"Simulated prompt from user {self.user_id}"
        return {
            'user_id': self.user_id,
            'prompt': prompt,
            'context': {
                'task_type': random.choice(['question_answering', 'code_generation', 'creative_writing']),
                'difficulty': random.choice(['easy', 'medium', 'hard']),
            },
            'timestamp': time.time(),
        }


class SimulationResult:
    """Results from a simulation run."""

    def __init__(self):
        self.total_requests = 0
        self.successful = 0
        self.failed = 0
        self.latencies: List[float] = []
        self.errors: List[str] = []
        self.start_time = 0.0
        self.end_time = 0.0
        self.concurrent_users = 0

    def add_latency(self, latency: float):
        self.latencies.append(latency)

    def add_error(self, error: str):
        self.errors.append(error)
        self.failed += 1

    def add_success(self):
        self.successful += 1

    def get_summary(self) -> Dict[str, Any]:
        sorted_latencies = sorted(self.latencies) if self.latencies else [0]
        return {
            'total_requests': self.total_requests,
            'successful': self.successful,
            'failed': self.failed,
            'success_rate': self.successful / max(self.total_requests, 1),
            'duration_seconds': self.end_time - self.start_time,
            'requests_per_second': self.total_requests / max(self.end_time - self.start_time, 0.001),
            'latency': {
                'mean': sum(self.latencies) / max(len(self.latencies), 1),
                'median': sorted_latencies[len(sorted_latencies) // 2],
                'p95': sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0,
                'p99': sorted_latencies[int(len(sorted_latencies) * 0.99)] if sorted_latencies else 0,
                'min': min(self.latencies) if self.latencies else 0,
                'max': max(self.latencies) if self.latencies else 0,
            },
            'concurrent_users': self.concurrent_users,
            'error_count': len(self.errors),
        }


class SimulationEngine(BaseOptimizerModule):
    """
    Simulates concurrent user load for stress testing.
    Supports configurable user counts, request rates, and profiles.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self._running = False
        self._lock = threading.Lock()

        self.default_prompt_pool = [
            "What is machine learning?",
            "Write a function to sort a list",
            "Explain quantum computing",
            "Write a poem about AI",
            "How does neural networking work?",
            "Debug this Python code",
            "Summarize the concept of recursion",
            "Translate this to French",
        ]

    def run_simulation(self, num_users: int = 10,
                       duration_seconds: float = 10.0,
                       requests_per_user: int = 5,
                       prompt_pool: Optional[List[str]] = None,
                       handler: Optional[Callable] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {'error': 'simulation disabled'}

        prompt_pool = prompt_pool or self.default_prompt_pool
        result = SimulationResult()
        result.start_time = time.time()
        result.concurrent_users = num_users
        self._running = True

        users = [SimulatedUser(i) for i in range(num_users)]

        def simulate_user(user: SimulatedUser):
            for _ in range(requests_per_user):
                if not self._running:
                    break
                try:
                    request = user.generate_request(prompt_pool)
                    start = time.time()
                    if handler:
                        handler(request)
                    latency = time.time() - start
                    result.add_latency(latency)
                    result.add_success()
                    result.total_requests += 1
                    user.request_count += 1
                    user.total_latency += latency
                except Exception as e:
                    result.add_error(str(e))
                    result.total_requests += 1
                    user.errors += 1

        threads = []
        for user in users:
            t = threading.Thread(target=simulate_user, args=(user,))
            t.daemon = True
            threads.append(t)
            t.start()

        deadline = time.time() + duration_seconds
        for t in threads:
            remaining = deadline - time.time()
            if remaining > 0:
                t.join(timeout=remaining)
            else:
                self._running = False
                t.join(timeout=1.0)

        self._running = False
        result.end_time = time.time()

        self.logger.info(
            f"Simulation complete: {result.total_requests} requests, "
            f"{result.successful} success, {result.failed} failed"
        )
        return result.get_summary()

    def stop_simulation(self):
        self._running = False

    def process(self, prompt: str, context: Dict[str, Any],
                model_adapter=None, pipeline_state=None) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return {'simulation_engine': 'ready', 'running': self._running}

    def get_metrics(self) -> Dict[str, Any]:
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'running': self._running,
        })
        return base_metrics
