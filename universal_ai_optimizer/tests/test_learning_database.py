import pytest
from universal_ai_optimizer.modules.learning_database import LearningDatabase, LearningRecord


class TestLearningDatabase:
    def test_init_defaults(self):
        ldb = LearningDatabase()
        assert ldb.enabled is True

    def test_disabled(self):
        ldb = LearningDatabase({"enabled": False})
        result = ldb.process("prompt", {})
        assert result == {}

    def test_record_success(self):
        ldb = LearningDatabase()
        ldb.record_success("compress", {"task_type": "code"}, score=0.9)
        summary = ldb.get_summary()
        assert summary['total_successes'] == 1

    def test_record_failure(self):
        ldb = LearningDatabase()
        ldb.record_failure("compress", {"task_type": "code"}, score=0.1)
        summary = ldb.get_summary()
        assert summary['total_failures'] == 1

    def test_action_success_rate(self):
        ldb = LearningDatabase()
        ldb.record_success("action_a", {})
        ldb.record_success("action_a", {})
        ldb.record_failure("action_a", {})
        rate = ldb.get_action_success_rate("action_a")
        assert abs(rate - 2/3) < 0.01

    def test_action_success_rate_empty(self):
        ldb = LearningDatabase()
        rate = ldb.get_action_success_rate("nonexistent")
        assert rate == 0.0

    def test_get_best_actions(self):
        ldb = LearningDatabase()
        ldb._action_stats.clear()
        ldb._success_records.clear()
        ldb._failure_records.clear()
        ldb.record_success("good", {})
        ldb.record_success("good", {})
        ldb.record_failure("bad", {})
        best = ldb.get_best_actions(2)
        assert best[0]['action'] == "good"
        assert best[0]['success_rate'] == 1.0

    def test_get_best_actions_empty(self):
        ldb = LearningDatabase()
        ldb._action_stats.clear()
        assert ldb.get_best_actions() == []

    def test_get_recent_records(self):
        ldb = LearningDatabase()
        ldb.record_success("a", {"x": 1})
        ldb.record_failure("b", {"x": 2})
        records = ldb.get_recent_records(limit=10)
        assert len(records) == 2

    def test_get_recent_records_success_only(self):
        ldb = LearningDatabase()
        ldb.record_success("a", {})
        ldb.record_failure("b", {})
        records = ldb.get_recent_records(outcome='success')
        assert len(records) == 1
        assert records[0]['outcome'] == 'success'

    def test_get_summary(self):
        ldb = LearningDatabase()
        ldb._action_stats.clear()
        ldb._success_records.clear()
        ldb._failure_records.clear()
        ldb.record_success("a", {})
        ldb.record_failure("b", {})
        summary = ldb.get_summary()
        assert summary['total_records'] == 2
        assert summary['unique_actions'] == 2

    def test_get_metrics(self):
        ldb = LearningDatabase()
        ldb.record_success("a", {})
        metrics = ldb.get_metrics()
        assert 'enabled' in metrics
        assert 'total_successes' in metrics
        assert 'best_actions' in metrics


class TestLearningRecord:
    def test_to_dict(self):
        record = LearningRecord("compress", {"task": "code"}, "success", 0.9)
        d = record.to_dict()
        assert d['action'] == "compress"
        assert d['outcome'] == "success"
        assert d['score'] == 0.9
        assert 'timestamp' in d
