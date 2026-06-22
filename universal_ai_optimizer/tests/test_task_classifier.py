import pytest
from universal_ai_optimizer.modules.task_classifier import TaskClassifier


class TestTaskClassifier:
    def test_init_defaults(self):
        tc = TaskClassifier()
        assert tc.config is not None
        assert tc.default_task is not None

    def test_disabled(self):
        tc = TaskClassifier({"enabled": False})
        result = tc.process("write code", {})
        assert result == {}

    def test_classify_code(self):
        tc = TaskClassifier()
        result = tc.process("write a python function to sort a list", {})
        assert "task_type" in result
        assert result["task_type"] is not None

    def test_classify_question(self):
        tc = TaskClassifier()
        result = tc.process("what is machine learning?", {})
        assert "task_type" in result

    def test_empty_prompt(self):
        tc = TaskClassifier()
        result = tc.process("", {})
        assert "task_type" in result
        assert result["task_confidence"] == 0.0

    def test_no_match_default(self):
        tc = TaskClassifier()
        result = tc.process("xyz123", {})
        assert "task_type" in result

    def test_add_category(self):
        tc = TaskClassifier()
        tc.add_task_category("custom", ["custom_keyword"])
        result = tc.process("custom_keyword test", {})
        assert "custom" in result.get("task_scores", {})

    def test_confidence_range(self):
        tc = TaskClassifier()
        result = tc.process("write code to solve the problem", {})
        assert 0.0 <= result["task_confidence"] <= 1.0

    def test_get_metrics(self):
        tc = TaskClassifier()
        metrics = tc.get_metrics()
        assert "enabled" in metrics
        assert "task_categories" in metrics
        assert "default_task" in metrics
