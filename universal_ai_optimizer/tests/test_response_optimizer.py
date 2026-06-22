import pytest
from universal_ai_optimizer.modules.response_optimizer import ResponseOptimizer


class TestResponseOptimizer:
    def test_init_defaults(self):
        ro = ResponseOptimizer()
        assert ro.config is not None

    def test_disabled(self):
        ro = ResponseOptimizer({"enabled": False})
        result = ro.process("prompt", {}, pipeline_state={"execution_result": "hello"})
        assert result == {}

    def test_process_with_execution_result(self):
        ro = ResponseOptimizer()
        state = {"execution_result": "Hello   world"}
        result = ro.process("prompt", {}, pipeline_state=state)
        assert isinstance(result, dict)

    def test_process_with_verification_result(self):
        ro = ResponseOptimizer()
        state = {"verification_result": "Verified text"}
        result = ro.process("prompt", {}, pipeline_state=state)
        assert isinstance(result, dict)

    def test_process_no_pipeline_state(self):
        ro = ResponseOptimizer()
        result = ro.process("prompt", {}, pipeline_state=None)
        assert result == {}

    def test_process_empty_state(self):
        ro = ResponseOptimizer()
        result = ro.process("prompt", {}, pipeline_state={})
        assert result == {}

    def test_compression_removes_whitespace(self):
        ro = ResponseOptimizer({"compression": True})
        state = {"execution_result": "Hello    world   test"}
        result = ro.process("prompt", {}, pipeline_state=state)
        assert isinstance(result, dict)

    def test_compression_disabled(self):
        ro = ResponseOptimizer({"compression": False})
        state = {"execution_result": "Hello    world"}
        result = ro.process("prompt", {}, pipeline_state=state)
        assert isinstance(result, dict)

    def test_get_metrics(self):
        ro = ResponseOptimizer()
        metrics = ro.get_metrics()
        assert "enabled" in metrics
        assert "compression" in metrics

    def test_empty_response(self):
        ro = ResponseOptimizer()
        state = {"execution_result": ""}
        result = ro.process("prompt", {}, pipeline_state=state)
        assert isinstance(result, dict)
