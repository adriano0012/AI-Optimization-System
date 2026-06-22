"""
Explainability Engine
Provides decision tracking and reasoning visualization for optimization choices.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Dict, Any, Optional, List
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)


class DecisionRecord:
    __slots__ = ['timestamp', 'decision_type', 'input_context', 'chosen_action',
                 'reasoning', 'confidence', 'alternatives', 'metadata']

    def __init__(self, decision_type: str, input_context: Dict[str, Any],
                 chosen_action: str, reasoning: str, confidence: float,
                 alternatives: Optional[List[str]] = None,
                 metadata: Optional[Dict] = None):
        self.timestamp = time.time()
        self.decision_type = decision_type
        self.input_context = input_context
        self.chosen_action = chosen_action
        self.reasoning = reasoning
        self.confidence = confidence
        self.alternatives = alternatives or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'decision_type': self.decision_type,
            'input_context': self.input_context,
            'chosen_action': self.chosen_action,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'alternatives': self.alternatives,
            'metadata': self.metadata,
        }


class ExplainabilityEngine(BaseOptimizerModule):
    """
    Tracks and explains optimization decisions for transparency.
    Provides decision history, reasoning chains, and confidence scores.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.max_history = self.config.get('max_history', 1000)
        self._lock = threading.RLock()

        self._decisions: deque = deque(maxlen=self.max_history)
        self._decisions_by_type: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._confidence_scores: Dict[str, List[float]] = defaultdict(list)

        self.logger.info("Explainability engine initialized")

    def record_decision(self, decision_type: str, input_context: Dict[str, Any],
                        chosen_action: str, reasoning: str, confidence: float,
                        alternatives: Optional[List[str]] = None,
                        metadata: Optional[Dict] = None) -> Dict[str, Any]:
        record = DecisionRecord(
            decision_type, input_context, chosen_action,
            reasoning, confidence, alternatives, metadata
        )
        with self._lock:
            self._decisions.append(record)
            self._decisions_by_type[decision_type].append(record)
            self._confidence_scores[decision_type].append(confidence)

        self.logger.debug(f"Recorded decision: {decision_type} -> {chosen_action} (confidence: {confidence:.2f})")
        return record.to_dict()

    def get_decision_explanation(self, decision_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            if 0 <= decision_id < len(self._decisions):
                return self._decisions[decision_id].to_dict()
        return None

    def get_recent_decisions(self, decision_type: Optional[str] = None,
                             limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            if decision_type:
                records = list(self._decisions_by_type.get(decision_type, []))
            else:
                records = list(self._decisions)
            records.sort(key=lambda x: x.timestamp, reverse=True)
            return [r.to_dict() for r in records[:limit]]

    def get_decision_tree(self, decision_type: str) -> Dict[str, Any]:
        records = list(self._decisions_by_type.get(decision_type, []))
        if not records:
            return {'type': decision_type, 'decisions': [], 'avg_confidence': 0}

        action_counts = defaultdict(int)
        for r in records:
            action_counts[r.chosen_action] += 1

        return {
            'type': decision_type,
            'total_decisions': len(records),
            'actions': dict(action_counts),
            'avg_confidence': sum(r.confidence for r in records) / len(records),
            'confidence_distribution': self._get_confidence_distribution(decision_type),
        }

    def get_confidence_trend(self, decision_type: str,
                             window: int = 20) -> List[float]:
        scores = self._confidence_scores.get(decision_type, [])
        if not scores:
            return []
        return scores[-window:]

    def _get_confidence_distribution(self, decision_type: str) -> Dict[str, int]:
        scores = self._confidence_scores.get(decision_type, [])
        buckets = {'low (0-0.3)': 0, 'medium (0.3-0.7)': 0, 'high (0.7-1.0)': 0}
        for s in scores:
            if s < 0.3:
                buckets['low (0-0.3)'] += 1
            elif s < 0.7:
                buckets['medium (0.3-0.7)'] += 1
            else:
                buckets['high (0.7-1.0)'] += 1
        return buckets

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._decisions)
            all_confidences = [r.confidence for r in self._decisions]
            return {
                'total_decisions': total,
                'decision_types': list(self._decisions_by_type.keys()),
                'avg_confidence': sum(all_confidences) / max(len(all_confidences), 1),
                'type_counts': {k: len(v) for k, v in self._decisions_by_type.items()},
            }

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
        })
        return base_metrics
