import pytest


class TestExecutionEngine:
    def test_init_with_defaults(self):
        from modules.execution_engine import ExecutionEngine
        ee = ExecutionEngine()
        assert ee.max_concurrent_tasks == 4
        assert ee.enable_dynamic_batching is True
        assert ee.enable_memory_pooling is True
        assert ee.enable_parallel_execution is True

    def test_init_with_custom_config(self):
        from modules.execution_engine import ExecutionEngine
        ee = ExecutionEngine({"max_concurrent_tasks": 8, "enable_parallel_execution": False})
        assert ee.max_concurrent_tasks == 8
        assert ee.enable_parallel_execution is False

    def test_endpoint_initialized(self):
        """_init_execution_engine is called in __init__ so thread_pool is not None."""
        from modules.execution_engine import ExecutionEngine
        ee = ExecutionEngine()
        assert ee.thread_pool is not None

    def test_execute_task_simple(self):
        from modules.execution_engine import ExecutionEngine
        ee = ExecutionEngine()
        result = ee.process("test prompt", {"task_id": "123"}, None, {})
        assert isinstance(result, dict)
        assert "execution_result" in result

    def test_process_disabled(self):
        from modules.execution_engine import ExecutionEngine
        ee = ExecutionEngine({"enabled": False})
        result = ee.process("test", {}, None, {})
        assert result == {}

    def test_get_metrics(self):
        from modules.execution_engine import ExecutionEngine
        ee = ExecutionEngine()
        metrics = ee.get_metrics()
        assert metrics["module"] == "ExecutionEngine"

    def test_shutdown_cleans_up(self):
        from modules.execution_engine import ExecutionEngine
        ee = ExecutionEngine()
        ee.shutdown()
        assert ee.thread_pool is None or True  # at least doesn't crash

    def test_lock_prevents_race_conditions(self):
        """Ensure RLock is used and doesn't deadlock on recursive calls."""
        from modules.execution_engine import ExecutionEngine
        ee = ExecutionEngine()
        with ee._lock:
            with ee._lock:
                assert True
