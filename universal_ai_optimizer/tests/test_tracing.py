"""
Tests for Distributed Tracing Module
"""

import time
import pytest
from modules.observability.tracing import (
    Span, TracingManager, RequestTracer,
)


class TestSpan:
    def test_create_span(self):
        span = Span(
            trace_id="t1", span_id="s1", name="test",
            start_time=time.time(),
        )
        assert span.trace_id == "t1"
        assert span.name == "test"
        assert span.status == "OK"

    def test_set_attribute(self):
        span = Span(
            trace_id="t1", span_id="s1", name="test",
            start_time=time.time(),
        )
        span.set_attribute("key", "value")
        assert span.attributes["key"] == "value"

    def test_add_event(self):
        span = Span(
            trace_id="t1", span_id="s1", name="test",
            start_time=time.time(),
        )
        span.add_event("click", {"x": 10})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "click"

    def test_set_status(self):
        span = Span(
            trace_id="t1", span_id="s1", name="test",
            start_time=time.time(),
        )
        span.set_status("ERROR", "something broke")
        assert span.status == "ERROR"
        assert span.attributes["status.description"] == "something broke"

    def test_duration(self):
        now = time.time()
        span = Span(
            trace_id="t1", span_id="s1", name="test",
            start_time=now, end_time=now + 0.5,
        )
        assert 490 < span.duration_ms < 510

    def test_to_dict(self):
        span = Span(
            trace_id="t1", span_id="s1", name="test",
            start_time=1000.0, end_time=1001.0,
        )
        d = span.to_dict()
        assert d['trace_id'] == "t1"
        assert d['duration_ms'] == 1000.0


class TestTracingManager:
    def test_start_and_end_span(self):
        mgr = TracingManager("test-service")
        span = mgr.start_span("my-span")
        assert span.name == "my-span"
        assert span.span_id in mgr._active_spans
        mgr.end_span(span.span_id)
        assert span.span_id not in mgr._active_spans
        assert len(mgr._spans) == 1

    def test_context_manager(self):
        mgr = TracingManager("test-service")
        with mgr.trace("op-1") as span:
            assert span.name == "op-1"
            span.set_attribute("x", 1)
        assert len(mgr._spans) == 1
        assert mgr._spans[0].attributes["x"] == 1
        assert mgr._spans[0].status == "OK"

    def test_trace_error(self):
        mgr = TracingManager("test-service")
        with pytest.raises(ValueError):
            with mgr.trace("failing-op"):
                raise ValueError("boom")
        assert len(mgr._spans) == 1
        assert mgr._spans[0].status == "ERROR"

    def test_get_trace(self):
        mgr = TracingManager()
        span1 = mgr.start_span("a")
        mgr.end_span(span1.span_id)
        span2 = mgr.start_span("b")
        mgr.end_span(span2.span_id)
        assert len(mgr.get_trace(span1.trace_id)) == 1

    def test_get_spans_by_name(self):
        mgr = TracingManager()
        for _ in range(3):
            s = mgr.start_span("fast-op")
            mgr.end_span(s.span_id)
        s = mgr.start_span("slow-op")
        mgr.end_span(s.span_id)
        assert len(mgr.get_spans_by_name("fast-op")) == 3
        assert len(mgr.get_spans_by_name("slow-op")) == 1

    def test_get_slow_spans(self):
        mgr = TracingManager()
        s = mgr.start_span("fast")
        s.end_time = s.start_time + 0.001
        mgr._spans.append(s)
        s2 = mgr.start_span("slow")
        s2.end_time = s2.start_time + 2.0
        mgr._spans.append(s2)
        assert len(mgr.get_slow_spans(threshold_ms=500)) == 1

    def test_stats(self):
        mgr = TracingManager()
        for _ in range(10):
            s = mgr.start_span("op")
            s.end_time = s.start_time + 0.01
            mgr._spans.append(s)
        stats = mgr.get_stats()
        assert stats['total_spans'] == 10
        assert stats['error_count'] == 0
        assert stats['avg_duration_ms'] > 0

    def test_clear(self):
        mgr = TracingManager()
        s = mgr.start_span("x")
        mgr.end_span(s.span_id)
        mgr.clear()
        assert len(mgr._spans) == 0
        assert len(mgr._active_spans) == 0

    def test_export(self):
        mgr = TracingManager()
        s = mgr.start_span("export-test")
        mgr.end_span(s.span_id)
        exported = mgr.export_traces()
        assert len(exported) == 1
        assert exported[0]['name'] == "export-test"

    def test_service_name(self):
        mgr = TracingManager("my-service")
        span = mgr.start_span("test")
        assert span.attributes['service.name'] == "my-service"
        mgr.end_span(span.span_id)


class TestRequestTracer:
    def test_trace_request(self):
        mgr = TracingManager("test")
        tracer = RequestTracer(mgr)
        span = tracer.trace_request("POST", "/v1/optimize", user_id="u1", org_id="o1")
        assert span.attributes['http.method'] == "POST"
        assert span.attributes['user.id'] == "u1"
        mgr.end_span(span.span_id)

    def test_trace_request_minimal(self):
        mgr = TracingManager("test")
        tracer = RequestTracer(mgr)
        span = tracer.trace_request("GET", "/health")
        assert span.attributes['http.method'] == "GET"
        mgr.end_span(span.span_id)
