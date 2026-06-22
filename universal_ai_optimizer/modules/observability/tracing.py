"""
Distributed Tracing Module
OpenTelemetry-based tracing for request tracking and performance monitoring.
"""

import logging
import os
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """Represents a single trace span."""
    trace_id: str
    span_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    parent_span_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.events.append({
            'name': name,
            'attributes': attributes or {},
            'timestamp': time.time(),
        })

    def set_status(self, status: str, description: str = ""):
        self.status = status
        if description:
            self.attributes['status.description'] = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'name': self.name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': self.duration_ms,
            'parent_span_id': self.parent_span_id,
            'attributes': self.attributes,
            'events': self.events,
            'status': self.status,
        }


class TracingManager:
    """
    In-process distributed tracing manager.
    Records trace spans for requests and operations.

    In production, this can be swapped with OpenTelemetry SDK.
    """

    def __init__(self, service_name: str = "uai-optimizer"):
        self.service_name = service_name
        self._spans: List[Span] = []
        self._active_spans: Dict[str, Span] = {}
        self._trace_counter = 0

    def _generate_trace_id(self) -> str:
        self._trace_counter += 1
        return f"trace-{self._trace_counter}-{int(time.time() * 1000)}"

    def _generate_span_id(self) -> str:
        return f"span-{int(time.time() * 1000000)}"

    def start_span(
        self,
        name: str,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        trace_id = self._generate_trace_id()
        span_id = self._generate_span_id()

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            name=name,
            start_time=time.time(),
            parent_span_id=parent_span_id,
            attributes=attributes or {},
        )
        span.set_attribute('service.name', self.service_name)

        self._active_spans[span_id] = span
        return span

    def end_span(self, span_id: str):
        if span_id in self._active_spans:
            span = self._active_spans.pop(span_id)
            span.end_time = time.time()
            self._spans.append(span)

    @contextmanager
    def trace(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        span = self.start_span(name, attributes=attributes)
        try:
            yield span
            span.set_status("OK")
        except Exception as e:
            span.set_status("ERROR", str(e))
            span.add_event('exception', {
                'exception.type': type(e).__name__,
                'exception.message': str(e),
            })
            raise
        finally:
            self.end_span(span.span_id)

    def get_trace(self, trace_id: str) -> List[Span]:
        return [s for s in self._spans if s.trace_id == trace_id]

    def get_spans_by_name(self, name: str) -> List[Span]:
        return [s for s in self._spans if s.name == name]

    def get_slow_spans(self, threshold_ms: float = 1000.0) -> List[Span]:
        return [s for s in self._spans if s.duration_ms > threshold_ms]

    def get_stats(self) -> Dict[str, Any]:
        if not self._spans:
            return {'total_spans': 0}

        durations = [s.duration_ms for s in self._spans]
        error_count = sum(1 for s in self._spans if s.status == "ERROR")

        return {
            'total_spans': len(self._spans),
            'error_count': error_count,
            'error_rate': error_count / len(self._spans) if self._spans else 0.0,
            'avg_duration_ms': sum(durations) / len(durations),
            'p50_duration_ms': sorted(durations)[len(durations) // 2],
            'p95_duration_ms': sorted(durations)[int(len(durations) * 0.95)],
            'p99_duration_ms': sorted(durations)[int(len(durations) * 0.99)],
            'max_duration_ms': max(durations),
        }

    def clear(self):
        self._spans.clear()
        self._active_spans.clear()

    def export_traces(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._spans]


class RequestTracer:
    """
    Middleware-style tracer for FastAPI requests.
    Wraps each request in a trace span with standard attributes.
    """

    def __init__(self, tracing_manager: TracingManager):
        self.tracer = tracing_manager

    def trace_request(
        self,
        method: str,
        path: str,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> Span:
        span = self.tracer.start_span(
            f"{method} {path}",
            attributes={
                'http.method': method,
                'http.url': path,
                'http.user_agent': 'api-client',
            }
        )
        if user_id:
            span.set_attribute('user.id', user_id)
        if org_id:
            span.set_attribute('organization.id', org_id)
        return span
