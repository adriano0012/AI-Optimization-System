import pytest
import time
from universal_ai_optimizer.modules.observability.error_tracker import ErrorTracker, ErrorCategory
from universal_ai_optimizer.modules.observability.system_monitor import SystemMonitor


class TestErrorTracker:
    def test_init_defaults(self):
        et = ErrorTracker()
        assert et.enabled is True

    def test_disabled(self):
        et = ErrorTracker({"enabled": False})
        result = et.process("prompt", {})
        assert result == {}

    def test_track_error(self):
        et = ErrorTracker()
        record = et.track_error(ValueError("test error"))
        assert record['error_type'] == 'ValueError'
        assert record['category'] == ErrorCategory.VALIDATION
        assert record['message'] == 'test error'

    def test_track_error_custom_category(self):
        et = ErrorTracker()
        record = et.track_error(Exception("auth fail"), category=ErrorCategory.AUTHENTICATION)
        assert record['category'] == ErrorCategory.AUTHENTICATION

    def test_error_counts(self):
        et = ErrorTracker()
        et.track_error(ValueError("e1"))
        et.track_error(ValueError("e2"))
        summary = et.get_summary()
        assert summary['categories'][ErrorCategory.VALIDATION] == 2

    def test_error_rate(self):
        et = ErrorTracker()
        et.track_error(TimeoutError("timeout"), category=ErrorCategory.TIMEOUT)
        rate = et.get_error_rate(ErrorCategory.TIMEOUT, window_seconds=60)
        assert rate > 0

    def test_get_errors(self):
        et = ErrorTracker()
        et.track_error(ValueError("e1"))
        et.track_error(RuntimeError("e2"))
        errors = et.get_errors()
        assert len(errors) == 2

    def test_get_errors_by_category(self):
        et = ErrorTracker()
        et.track_error(ValueError("v"))
        et.track_error(RuntimeError("r"))
        val_errors = et.get_errors(category=ErrorCategory.VALIDATION)
        assert len(val_errors) == 1

    def test_clear(self):
        et = ErrorTracker()
        et.track_error(ValueError("test"))
        et.clear()
        assert et.get_summary()['total_errors'] == 0

    def test_get_metrics(self):
        et = ErrorTracker()
        metrics = et.get_metrics()
        assert 'enabled' in metrics
        assert 'total_errors' in metrics

    def test_threshold_alert(self):
        et = ErrorTracker({"alert_threshold": 2})
        et.track_error(ValueError("e1"))
        et.track_error(ValueError("e2"))
        assert et._error_counts[ErrorCategory.VALIDATION] >= 2


class TestSystemMonitor:
    def test_init_defaults(self):
        sm = SystemMonitor()
        assert sm.enabled is True

    def test_disabled(self):
        sm = SystemMonitor({"enabled": False})
        result = sm.process("prompt", {})
        assert result == {}

    def test_get_system_stats(self):
        sm = SystemMonitor()
        stats = sm.get_system_stats()
        assert 'timestamp' in stats
        assert 'cpu' in stats
        assert 'ram' in stats
        assert 'disk' in stats
        assert 'network' in stats

    def test_get_cpu_stats(self):
        sm = SystemMonitor()
        cpu = sm.get_cpu_stats()
        assert 'available' in cpu

    def test_get_ram_stats(self):
        sm = SystemMonitor()
        ram = sm.get_ram_stats()
        assert 'available' in ram

    def test_get_disk_stats(self):
        sm = SystemMonitor()
        disk = sm.get_disk_stats()
        assert 'available' in disk

    def test_get_network_stats(self):
        sm = SystemMonitor()
        net = sm.get_network_stats()
        assert 'available' in net
        assert 'bytes_sent' in net

    def test_get_network_rates(self):
        sm = SystemMonitor()
        sm.get_network_stats()
        time.sleep(0.05)
        net = sm.get_network_stats()
        assert 'bytes_sent_rate' in net

    def test_get_gpu_stats(self):
        sm = SystemMonitor()
        gpu = sm.get_gpu_stats()
        assert 'available' in gpu

    def test_get_metrics(self):
        sm = SystemMonitor()
        metrics = sm.get_metrics()
        assert 'enabled' in metrics
        assert 'psutil_available' in metrics
        assert 'gpu_available' in metrics
