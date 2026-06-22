import pytest
import time
from universal_ai_optimizer.modules.simulation.simulation_engine import (
    SimulationEngine, SimulatedUser, SimulationResult
)
from universal_ai_optimizer.modules.explainability.explainability_engine import (
    ExplainabilityEngine, DecisionRecord
)
from universal_ai_optimizer.modules.benchmark.benchmark_suite import (
    BenchmarkSuite, BenchmarkTask, BenchmarkResult
)


class TestSimulatedUser:
    def test_init(self):
        user = SimulatedUser(1)
        assert user.user_id == 1
        assert user.request_count == 0

    def test_generate_request(self):
        user = SimulatedUser(42)
        req = user.generate_request(["test prompt"])
        assert req['user_id'] == 42
        assert req['prompt'] == "test prompt"

    def test_generate_request_default_pool(self):
        user = SimulatedUser(1)
        req = user.generate_request([])
        assert "Simulated prompt" in req['prompt']


class TestSimulationResult:
    def test_init(self):
        sr = SimulationResult()
        assert sr.total_requests == 0
        assert sr.latencies == []

    def test_add_latency(self):
        sr = SimulationResult()
        sr.add_latency(0.1)
        sr.add_latency(0.2)
        assert sr.latencies == [0.1, 0.2]

    def test_add_success(self):
        sr = SimulationResult()
        sr.add_success()
        assert sr.successful == 1

    def test_add_error(self):
        sr = SimulationResult()
        sr.add_error("timeout")
        assert sr.failed == 1

    def test_get_summary_empty(self):
        sr = SimulationResult()
        sr.start_time = time.time()
        sr.end_time = time.time() + 1
        summary = sr.get_summary()
        assert summary['total_requests'] == 0


class TestSimulationEngine:
    def test_init_defaults(self):
        se = SimulationEngine()
        assert se.enabled is True

    def test_disabled(self):
        se = SimulationEngine({"enabled": False})
        result = se.run_simulation(num_users=2, duration_seconds=0.1, requests_per_user=1)
        assert result.get('error') == 'simulation disabled'

    def test_run_simulation_small(self):
        se = SimulationEngine()
        result = se.run_simulation(num_users=2, duration_seconds=1.0, requests_per_user=2)
        assert result['total_requests'] > 0
        assert 'latency' in result
        assert 'requests_per_second' in result

    def test_run_simulation_with_handler(self):
        se = SimulationEngine()
        def handler(req):
            return f"Response to: {req['prompt']}"
        result = se.run_simulation(num_users=1, duration_seconds=0.5, requests_per_user=1, handler=handler)
        assert result['total_requests'] > 0

    def test_stop_simulation(self):
        se = SimulationEngine()
        se.stop_simulation()
        assert se._running is False

    def test_get_metrics(self):
        se = SimulationEngine()
        metrics = se.get_metrics()
        assert 'enabled' in metrics
        assert 'running' in metrics

    def test_process(self):
        se = SimulationEngine()
        result = se.process("prompt", {})
        assert result['simulation_engine'] == 'ready'


class TestDecisionRecord:
    def test_to_dict(self):
        record = DecisionRecord("routing", {"task": "code"}, "gpt-4", "best for code", 0.9)
        d = record.to_dict()
        assert d['decision_type'] == "routing"
        assert d['chosen_action'] == "gpt-4"
        assert d['confidence'] == 0.9


class TestExplainabilityEngine:
    def test_init_defaults(self):
        ee = ExplainabilityEngine()
        assert ee.enabled is True

    def test_disabled(self):
        ee = ExplainabilityEngine({"enabled": False})
        result = ee.process("prompt", {})
        assert result == {}

    def test_record_decision(self):
        ee = ExplainabilityEngine()
        record = ee.record_decision("routing", {"task": "code"}, "gpt-4", "best for code", 0.9)
        assert record['decision_type'] == "routing"
        assert record['chosen_action'] == "gpt-4"

    def test_get_recent_decisions(self):
        ee = ExplainabilityEngine()
        ee.record_decision("routing", {}, "gpt-4", "reason1", 0.8)
        ee.record_decision("compression", {}, "aggressive", "reason2", 0.7)
        decisions = ee.get_recent_decisions()
        assert len(decisions) == 2

    def test_get_recent_decisions_by_type(self):
        ee = ExplainabilityEngine()
        ee.record_decision("routing", {}, "gpt-4", "reason1", 0.8)
        ee.record_decision("compression", {}, "aggressive", "reason2", 0.7)
        routing = ee.get_recent_decisions(decision_type="routing")
        assert len(routing) == 1
        assert routing[0]['decision_type'] == "routing"

    def test_get_decision_tree(self):
        ee = ExplainabilityEngine()
        ee.record_decision("routing", {}, "gpt-4", "r1", 0.9)
        ee.record_decision("routing", {}, "gpt-4", "r2", 0.8)
        ee.record_decision("routing", {}, "claude-3", "r3", 0.7)
        tree = ee.get_decision_tree("routing")
        assert tree['total_decisions'] == 3
        assert tree['actions']['gpt-4'] == 2

    def test_get_confidence_trend(self):
        ee = ExplainabilityEngine()
        for i in range(5):
            ee.record_decision("routing", {}, "gpt-4", f"r{i}", 0.5 + i * 0.1)
        trend = ee.get_confidence_trend("routing")
        assert len(trend) == 5

    def test_get_summary(self):
        ee = ExplainabilityEngine()
        ee.record_decision("routing", {}, "gpt-4", "r", 0.9)
        summary = ee.get_summary()
        assert summary['total_decisions'] == 1

    def test_get_metrics(self):
        ee = ExplainabilityEngine()
        metrics = ee.get_metrics()
        assert 'enabled' in metrics
        assert 'total_decisions' in metrics


class TestBenchmarkTask:
    def test_init(self):
        bt = BenchmarkTask("test", "prompt", {"task_type": "code"})
        assert bt.name == "test"
        assert bt.prompt == "prompt"


class TestBenchmarkResult:
    def test_init(self):
        br = BenchmarkResult("test_task")
        assert br.task_name == "test_task"
        assert br.success is True

    def test_to_dict(self):
        br = BenchmarkResult("test_task")
        br.latency_ms = 100.5
        d = br.to_dict()
        assert d['task_name'] == "test_task"
        assert d['latency_ms'] == 100.5


class TestBenchmarkSuite:
    def test_init_defaults(self):
        bs = BenchmarkSuite()
        assert bs.enabled is True
        assert len(bs._tasks) > 0

    def test_disabled(self):
        bs = BenchmarkSuite({"enabled": False})
        result = bs.run_benchmark(lambda p, c: "response")
        assert result.get('error') == 'benchmark disabled'

    def test_run_benchmark(self):
        bs = BenchmarkSuite()
        def handler(prompt, context):
            return f"Response to {prompt}"
        result = bs.run_benchmark(handler)
        assert result['total_tasks'] > 0
        assert result['successful'] > 0
        assert 'avg_latency_ms' in result

    def test_run_benchmark_with_errors(self):
        bs = BenchmarkSuite()
        def failing_handler(prompt, context):
            raise ValueError("fail")
        result = bs.run_benchmark(failing_handler)
        assert result['failed'] > 0

    def test_add_task(self):
        bs = BenchmarkSuite()
        initial = len(bs._tasks)
        bs.add_task("custom", "custom prompt", {"task_type": "custom"})
        assert len(bs._tasks) == initial + 1

    def test_get_history(self):
        bs = BenchmarkSuite()
        bs.run_benchmark(lambda p, c: "ok")
        history = bs.get_history()
        assert len(history) > 0

    def test_get_comparison(self):
        bs = BenchmarkSuite()
        a = {'success_rate': 0.9, 'avg_latency_ms': 100}
        b = {'success_rate': 0.95, 'avg_latency_ms': 80}
        comp = bs.get_comparison(a, b)
        assert 'latency_improvement' in comp

    def test_get_metrics(self):
        bs = BenchmarkSuite()
        metrics = bs.get_metrics()
        assert 'enabled' in metrics
        assert 'total_tasks' in metrics
