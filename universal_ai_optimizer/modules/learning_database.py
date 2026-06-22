"""
Learning Database
Tracks success/failure outcomes for adaptive optimization.
"""

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Dict, Any, Optional, List
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)


class LearningRecord:
    __slots__ = ['timestamp', 'action', 'context', 'outcome', 'score', 'metadata']

    def __init__(self, action: str, context: Dict[str, Any],
                 outcome: str, score: float, metadata: Optional[Dict] = None):
        self.timestamp = time.time()
        self.action = action
        self.context = context
        self.outcome = outcome
        self.score = score
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'action': self.action,
            'context': self.context,
            'outcome': self.outcome,
            'score': self.score,
            'metadata': self.metadata,
        }


class LearningDatabase(BaseOptimizerModule):
    """
    Tracks success/failure outcomes for adaptive optimization.
    Provides analytics for decision-making.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.max_records = self.config.get('max_records', 10000)
        self._lock = threading.RLock()

        self._success_records: deque = deque(maxlen=self.max_records)
        self._failure_records: deque = deque(maxlen=self.max_records)
        self._action_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'success': 0, 'failure': 0})
        self._context_patterns: Dict[str, List[float]] = defaultdict(list)

        self._load_persisted_state()
        self.logger.info("Learning database initialized")

    def record_success(self, action: str, context: Dict[str, Any],
                       score: float = 1.0, metadata: Optional[Dict] = None):
        record = LearningRecord(action, context, 'success', score, metadata)
        with self._lock:
            self._success_records.append(record)
            self._action_stats[action]['success'] += 1
            context_key = self._extract_context_key(context)
            self._context_patterns[context_key].append(score)
        self._persist_state()

    def record_failure(self, action: str, context: Dict[str, Any],
                       score: float = 0.0, metadata: Optional[Dict] = None):
        record = LearningRecord(action, context, 'failure', score, metadata)
        with self._lock:
            self._failure_records.append(record)
            self._action_stats[action]['failure'] += 1
            context_key = self._extract_context_key(context)
            self._context_patterns[context_key].append(score)
        self._persist_state()

    def get_action_success_rate(self, action: str) -> float:
        with self._lock:
            stats = self._action_stats.get(action, {'success': 0, 'failure': 0})
            total = stats['success'] + stats['failure']
            return stats['success'] / max(total, 1)

    def get_best_actions(self, top_n: int = 5) -> List[Dict[str, Any]]:
        with self._lock:
            results = []
            for action, stats in self._action_stats.items():
                total = stats['success'] + stats['failure']
                if total > 0:
                    results.append({
                        'action': action,
                        'success_rate': stats['success'] / total,
                        'total': total,
                        'successes': stats['success'],
                    })
            results.sort(key=lambda x: x['success_rate'], reverse=True)
            return results[:top_n]

    def get_context_recommendation(self, context: Dict[str, Any]) -> Optional[str]:
        context_key = self._extract_context_key(context)
        with self._lock:
            scores = self._context_patterns.get(context_key, [])
            if not scores:
                return None
            best_actions = self.get_best_actions(1)
            return best_actions[0]['action'] if best_actions else None

    def get_recent_records(self, outcome: Optional[str] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            records = []
            if outcome == 'success' or outcome is None:
                records.extend(r.to_dict() for r in self._success_records)
            if outcome == 'failure' or outcome is None:
                records.extend(r.to_dict() for r in self._failure_records)
            records.sort(key=lambda x: x['timestamp'], reverse=True)
            return records[:limit]

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'total_successes': len(self._success_records),
                'total_failures': len(self._failure_records),
                'total_records': len(self._success_records) + len(self._failure_records),
                'unique_actions': len(self._action_stats),
                'overall_success_rate': len(self._success_records) / max(
                    len(self._success_records) + len(self._failure_records), 1
                ),
            }

    def _extract_context_key(self, context: Dict[str, Any]) -> str:
        task_type = context.get('task_type', 'unknown')
        difficulty = context.get('difficulty', 'medium')
        return f"{task_type}:{difficulty}"

    def _persist_state(self):
        try:
            data_dir = os.path.join(os.getcwd(), 'data', 'learning_db')
            os.makedirs(data_dir, exist_ok=True)
            state = {
                'action_stats': dict(self._action_stats),
                'success_count': len(self._success_records),
                'failure_count': len(self._failure_records),
            }
            with open(os.path.join(data_dir, 'state.json'), 'w') as f:
                json.dump(state, f)
        except Exception as e:
            self.logger.debug(f"Failed to persist learning state: {e}")

    def _load_persisted_state(self):
        try:
            state_path = os.path.join(os.getcwd(), 'data', 'learning_db', 'state.json')
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                self._action_stats = defaultdict(lambda: {'success': 0, 'failure': 0})
                for action, stats in state.get('action_stats', {}).items():
                    self._action_stats[action] = stats
                self.logger.debug("Loaded persisted learning state")
        except Exception as e:
            self.logger.debug(f"Failed to load learning state: {e}")

    def process(self, prompt: str, context: Dict[str, Any],
                model_adapter=None, pipeline_state=None) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return self.get_summary()

    def get_metrics(self) -> Dict[str, Any]:
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            **self.get_summary(),
            'best_actions': self.get_best_actions(3),
        })
        return base_metrics
