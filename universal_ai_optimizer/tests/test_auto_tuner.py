import pytest
from universal_ai_optimizer.modules.auto.auto_tuner import AutoTuner


class TestAutoTuner:
    def test_init_defaults(self):
        at = AutoTuner()
        assert at.config is not None

    def test_disabled(self):
        at = AutoTuner({"enabled": False})
        result = at.tune_parameters({"quality": 0.8})
        assert isinstance(result, dict)

    def test_tune_parameters_returns_dict(self):
        at = AutoTuner()
        result = at.tune_parameters({"quality": 0.8, "latency": 0.5})
        assert isinstance(result, dict)
        assert "learning_rate" in result

    def test_tune_parameters_updates_history(self):
        at = AutoTuner()
        at.tune_parameters({"quality": 0.8})
        at.tune_parameters({"quality": 0.9})
        assert len(at.outcome_history) >= 2

    def test_rule_based_latency_high(self):
        at = AutoTuner({"tuning_strategy": "rule_based"})
        result = at.tune_parameters({"latency": 2.0})
        assert isinstance(result, dict)

    def test_rule_based_quality_low(self):
        at = AutoTuner({"tuning_strategy": "rule_based"})
        result = at.tune_parameters({"quality": 0.3})
        assert isinstance(result, dict)

    def test_constraints_applied(self):
        at = AutoTuner()
        result = at.tune_parameters({"quality": 0.8})
        for key in at.tuning_parameters:
            min_val = at.tuning_parameters[key].get('min', float('-inf'))
            max_val = at.tuning_parameters[key].get('max', float('inf'))
            assert result[key] >= min_val
            assert result[key] <= max_val

    def test_smoothing(self):
        at = AutoTuner({"smoothing_factor": 0.5})
        r1 = at.tune_parameters({"quality": 0.8})
        r2 = at.tune_parameters({"quality": 0.2})
        assert isinstance(r2, dict)

    def test_get_recommendations(self):
        at = AutoTuner()
        at.tune_parameters({"quality": 0.8})
        rec = at.get_tuning_recommendations()
        assert "current_parameters" in rec
        assert "best_outcome_so_far" in rec

    def test_get_metrics(self):
        at = AutoTuner()
        metrics = at.get_metrics()
        assert "tuning_strategy" in metrics
        assert "current_parameters" in metrics
