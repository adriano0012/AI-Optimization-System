"""
SLA Monitoring & Alerting
Track service level objectives and trigger alerts on violations.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum


class SLAStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SLADefinition:
    sla_id: str
    name: str
    description: str
    metric: str
    target_value: float
    operator: str = ">="
    window_minutes: int = 60
    severity: AlertSeverity = AlertSeverity.HIGH
    enabled: bool = True
    organization_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sla_id': self.sla_id,
            'name': self.name,
            'description': self.description,
            'metric': self.metric,
            'target_value': self.target_value,
            'operator': self.operator,
            'window_minutes': self.window_minutes,
            'severity': self.severity.value,
            'enabled': self.enabled,
        }


@dataclass
class SLAStatusRecord:
    sla_id: str
    status: SLAStatus
    current_value: float
    target_value: float
    measured_at: float
    breach_duration_seconds: float = 0.0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sla_id': self.sla_id,
            'status': self.status.value,
            'current_value': self.current_value,
            'target_value': self.target_value,
            'measured_at': self.measured_at,
            'breach_duration_seconds': self.breach_duration_seconds,
            'message': self.message,
        }


@dataclass
class Alert:
    alert_id: str
    sla_id: str
    severity: AlertSeverity
    message: str
    triggered_at: float
    resolved_at: Optional[float] = None
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'sla_id': self.sla_id,
            'severity': self.severity.value,
            'message': self.message,
            'triggered_at': self.triggered_at,
            'resolved_at': self.resolved_at,
            'acknowledged': self.acknowledged,
        }


class SLAMonitor:
    """
    SLA monitoring with alerting and breach tracking.
    """

    def __init__(self):
        self._slas: Dict[str, SLADefinition] = {}
        self._status: Dict[str, SLAStatusRecord] = {}
        self._alerts: List[Alert] = []
        self._metric_values: Dict[str, List[tuple]] = {}
        self._alert_handlers: List[Callable] = []
        self._lock = threading.Lock()

    def register_alert_handler(self, handler: Callable):
        self._alert_handlers.append(handler)

    def create_sla(
        self,
        name: str,
        description: str,
        metric: str,
        target_value: float,
        operator: str = ">=",
        window_minutes: int = 60,
        severity: AlertSeverity = AlertSeverity.HIGH,
        organization_id: Optional[str] = None,
    ) -> SLADefinition:
        import secrets
        sla_id = f"sla-{secrets.token_hex(8)}"
        sla = SLADefinition(
            sla_id=sla_id, name=name, description=description,
            metric=metric, target_value=target_value, operator=operator,
            window_minutes=window_minutes, severity=severity,
            organization_id=organization_id,
        )
        with self._lock:
            self._slas[sla_id] = sla
        return sla

    def record_metric(self, metric_name: str, value: float):
        now = time.time()
        with self._lock:
            if metric_name not in self._metric_values:
                self._metric_values[metric_name] = []
            self._metric_values[metric_name].append((now, value))

    def evaluate_sla(self, sla_id: str) -> Optional[SLAStatusRecord]:
        sla = self._slas.get(sla_id)
        if not sla:
            return None

        now = time.time()
        window_start = now - (sla.window_minutes * 60)

        with self._lock:
            values = self._metric_values.get(sla.metric, [])

        window_values = [v for t, v in values if t >= window_start]

        if not window_values:
            record = SLAStatusRecord(
                sla_id=sla_id, status=SLAStatus.UNKNOWN,
                current_value=0.0, target_value=sla.target_value,
                measured_at=now, message="No data in window",
            )
        else:
            avg_value = sum(window_values) / len(window_values)
            status = self._check_sla(sla, avg_value)
            record = SLAStatusRecord(
                sla_id=sla_id, status=status,
                current_value=avg_value, target_value=sla.target_value,
                measured_at=now,
            )

            if status == SLAStatus.VIOLATED:
                existing = [a for a in self._alerts
                          if a.sla_id == sla_id and a.resolved_at is None]
                if not existing:
                    self._create_alert(sla, avg_value)

        self._status[sla_id] = record
        return record

    def _check_sla(self, sla: SLADefinition, value: float) -> SLAStatus:
        ops = {'>=': lambda a, b: a >= b, '<=': lambda a, b: a <= b,
               '>': lambda a, b: a > b, '<': lambda a, b: a < b,
               '==': lambda a, b: a == b}
        op_func = ops.get(sla.operator, lambda a, b: True)

        if op_func(value, sla.target_value):
            return SLAStatus.HEALTHY
        return SLAStatus.VIOLATED

    def _create_alert(self, sla: SLADefinition, current_value: float):
        import secrets
        alert = Alert(
            alert_id=f"alert-{secrets.token_hex(8)}",
            sla_id=sla.sla_id,
            severity=sla.severity,
            message=f"SLA '{sla.name}' violated: {current_value} vs target {sla.target_value}",
            triggered_at=time.time(),
        )
        self._alerts.append(alert)

        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception:
                pass

    def resolve_alert(self, alert_id: str) -> Optional[Alert]:
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved_at = time.time()
                return alert
        return None

    def acknowledge_alert(self, alert_id: str) -> Optional[Alert]:
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return alert
        return None

    def get_sla_status(self, sla_id: str) -> Optional[SLAStatusRecord]:
        if sla_id not in self._status:
            self.evaluate_sla(sla_id)
        return self._status.get(sla_id)

    def get_alerts(
        self, sla_id: Optional[str] = None, unresolved_only: bool = False
    ) -> List[Dict[str, Any]]:
        alerts = list(self._alerts)
        if sla_id:
            alerts = [a for a in alerts if a.sla_id == sla_id]
        if unresolved_only:
            alerts = [a for a in alerts if a.resolved_at is None]
        return [a.to_dict() for a in alerts]

    def list_slas(self, organization_id: Optional[str] = None) -> List[Dict[str, Any]]:
        slas = list(self._slas.values())
        if organization_id:
            slas = [s for s in slas if s.organization_id == organization_id]
        return [s.to_dict() for s in slas]

    def get_dashboard(self) -> Dict[str, Any]:
        total = len(self._slas)
        healthy = sum(1 for s in self._status.values() if s.status == SLAStatus.HEALTHY)
        violated = sum(1 for s in self._status.values() if s.status == SLAStatus.VIOLATED)
        active_alerts = sum(1 for a in self._alerts if a.resolved_at is None)
        return {
            'total_slas': total,
            'healthy': healthy,
            'violated': violated,
            'warning': total - healthy - violated,
            'active_alerts': active_alerts,
            'compliance_rate': healthy / total * 100 if total else 100.0,
        }
