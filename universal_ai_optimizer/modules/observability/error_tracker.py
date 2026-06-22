"""
Error Tracking Module
Provides centralized error tracking, categorization, and reporting.
"""

import logging
import threading
import time
import traceback
from collections import defaultdict, deque
from typing import Dict, Any, Optional, List
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)


class ErrorCategory:
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    VALIDATION = "validation"
    NETWORK = "network"
    TIMEOUT = "timeout"
    MODEL_ERROR = "model_error"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class ErrorTracker(BaseOptimizerModule):
    """
    Centralized error tracking with categorization, rate counting,
    and alerting thresholds.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.max_history = self.config.get('max_history', 1000)
        self.alert_threshold = self.config.get('alert_threshold', 10)

        self._errors = deque(maxlen=self.max_history)
        self._error_counts = defaultdict(int)
        self._error_rates = defaultdict(lambda: deque(maxlen=100))
        self._recent_errors = deque(maxlen=50)
        self._lock = threading.RLock()

        self._error_categories = {
            'ValueError': ErrorCategory.VALIDATION,
            'TypeError': ErrorCategory.VALIDATION,
            'KeyError': ErrorCategory.VALIDATION,
            'ConnectionError': ErrorCategory.NETWORK,
            'TimeoutError': ErrorCategory.TIMEOUT,
            'PermissionError': ErrorCategory.AUTHORIZATION,
            'FileNotFoundError': ErrorCategory.INTERNAL,
            'ImportError': ErrorCategory.INTERNAL,
        }

        self.logger.info("Error tracker initialized")

    def process(self, prompt: str, context: Dict[str, Any],
                model_adapter=None, pipeline_state=None) -> Dict[str, Any]:
        if not self.enabled:
            return {}

        return {
            'error_tracking': {
                'total_errors': sum(self._error_counts.values()),
                'recent_errors': len(self._recent_errors),
                'categories': dict(self._error_counts),
            }
        }

    def track_error(self, error: Exception, context: Optional[Dict[str, Any]] = None,
                    category: Optional[str] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {}

        error_type = type(error).__name__
        if category is None:
            category = self._error_categories.get(error_type, ErrorCategory.UNKNOWN)

        error_record = {
            'timestamp': time.time(),
            'error_type': error_type,
            'message': str(error),
            'category': category,
            'traceback': traceback.format_exc(),
            'context': context or {},
        }

        with self._lock:
            self._errors.append(error_record)
            self._error_counts[category] += 1
            self._error_rates[category].append(time.time())
            self._recent_errors.append(error_record)

        if self._error_counts[category] >= self.alert_threshold:
            self.logger.warning(
                f"Error threshold reached for category '{category}': "
                f"{self._error_counts[category]} errors"
            )

        return error_record

    def get_error_rate(self, category: str, window_seconds: int = 60) -> float:
        with self._lock:
            if category not in self._error_rates:
                return 0.0
            cutoff = time.time() - window_seconds
            recent = [t for t in self._error_rates[category] if t > cutoff]
            return len(recent) / max(window_seconds, 1)

    def get_errors(self, category: Optional[str] = None,
                   limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            errors = list(self._errors)
        if category:
            errors = [e for e in errors if e['category'] == category]
        return errors[-limit:]

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            total = sum(self._error_counts.values())
            return {
                'total_errors': total,
                'categories': dict(self._error_counts),
                'recent_count': len(self._recent_errors),
                'error_rates': {
                    cat: self.get_error_rate(cat)
                    for cat in self._error_counts
                },
            }

    def get_metrics(self) -> Dict[str, Any]:
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'total_errors': sum(self._error_counts.values()),
            'categories': dict(self._error_counts),
            'recent_errors': len(self._recent_errors),
            'alert_threshold': self.alert_threshold,
        })
        return base_metrics

    def clear(self):
        with self._lock:
            self._errors.clear()
            self._error_counts.clear()
            self._error_rates.clear()
            self._recent_errors.clear()
        self.logger.info("Error tracker cleared")
