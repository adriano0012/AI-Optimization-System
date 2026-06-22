import pytest


class TestVerificationEngine:
    def test_init_with_defaults(self):
        from universal_ai_optimizer.modules.verification_engine import VerificationEngine
        ve = VerificationEngine()
        assert ve.enabled is True
        assert ve.threshold == 0.95
        assert ve.max_iterations == 3
        assert ve.confidence_threshold == 0.8
        assert ve.consensus_threshold == 0.7

    def test_init_with_custom_config(self):
        from universal_ai_optimizer.modules.verification_engine import VerificationEngine
        ve = VerificationEngine({
            "threshold": 0.5,
            "max_iterations": 1,
            "confidence_threshold": 0.3,
        })
        assert ve.threshold == 0.5
        assert ve.max_iterations == 1
        assert ve.confidence_threshold == 0.3

    def test_process_disabled(self):
        from universal_ai_optimizer.modules.verification_engine import VerificationEngine
        ve = VerificationEngine({"enabled": False})
        result = ve.process("prompt", {}, None, {"execution_result": "test"})
        assert result == {}

    def test_process_no_execution_result(self):
        from universal_ai_optimizer.modules.verification_engine import VerificationEngine
        ve = VerificationEngine()
        result = ve.process("prompt", {}, None, {})
        assert result == {}

    def test_process_no_pipeline_state(self):
        from universal_ai_optimizer.modules.verification_engine import VerificationEngine
        ve = VerificationEngine()
        result = ve.process("prompt", {}, None, None)
        assert result == {}

    def test_get_metrics(self):
        from universal_ai_optimizer.modules.verification_engine import VerificationEngine
        ve = VerificationEngine()
        metrics = ve.get_metrics()
        assert metrics["module"] == "VerificationEngine"

    def test_verification_history_config(self):
        from universal_ai_optimizer.modules.verification_engine import VerificationEngine
        ve = VerificationEngine({"max_verification_history": 5})
        assert ve._max_verification_history == 5

    def test_calculate_text_similarity_identical(self):
        from universal_ai_optimizer.modules.verification_engine import VerificationEngine
        ve = VerificationEngine()
        sim = ve._calculate_text_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_calculate_text_similarity_different(self):
        from universal_ai_optimizer.modules.verification_engine import VerificationEngine
        ve = VerificationEngine()
        sim = ve._calculate_text_similarity("hello world", "foo bar baz")
        assert sim < 1.0

    def test_calculate_text_similarity_empty(self):
        from universal_ai_optimizer.modules.verification_engine import VerificationEngine
        ve = VerificationEngine()
        sim = ve._calculate_text_similarity("", "")
        assert sim == 0.0

    def test_calculate_text_similarity_one_empty(self):
        from universal_ai_optimizer.modules.verification_engine import VerificationEngine
        ve = VerificationEngine()
        sim = ve._calculate_text_similarity("hello", "")
        assert sim == 0.0

    def test_heuristic_fact_check(self):
        from universal_ai_optimizer.modules.verification_engine import VerificationEngine
        ve = VerificationEngine()
        score = ve._heuristic_fact_check("The Earth is round. Paris is the capital of France.")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_verification_history_maxlen_default(self):
        from universal_ai_optimizer.modules.verification_engine import VerificationEngine
        ve = VerificationEngine()
        assert ve._max_verification_history == 10000
