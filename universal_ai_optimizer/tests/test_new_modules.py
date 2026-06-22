"""
Tests for Rate Limiter, Request Logger, Webhooks, Model Registry,
A/B Testing, Prompt Versioning, SLA Monitor, Batch Processor.
"""

import time
import pytest

# --- Rate Limiter Tests ---

from modules.security.rate_limiter import (
    RateLimiter, TokenBucket, SlidingWindowCounter,
    FixedWindowCounter, RateLimitStrategy, RateLimitResult,
)


class TestTokenBucket:
    def test_consume_within_capacity(self):
        tb = TokenBucket(capacity=5, refill_rate=10.0)
        result = tb.consume()
        assert result.allowed is True
        assert result.remaining == 4

    def test_consume_exceeds_capacity(self):
        tb = TokenBucket(capacity=2, refill_rate=1.0)
        tb.consume()
        tb.consume()
        result = tb.consume()
        assert result.allowed is False

    def test_refill(self):
        tb = TokenBucket(capacity=2, refill_rate=100.0)
        tb.consume()
        tb.consume()
        time.sleep(0.05)
        result = tb.consume()
        assert result.allowed is True


class TestSlidingWindowCounter:
    def test_within_limit(self):
        sw = SlidingWindowCounter(limit=5, window_seconds=1)
        result = sw.check()
        assert result.allowed is True
        assert result.remaining == 4

    def test_exceeds_limit(self):
        sw = SlidingWindowCounter(limit=2, window_seconds=1)
        sw.check()
        sw.check()
        result = sw.check()
        assert result.allowed is False

    def test_window_expiry(self):
        sw = SlidingWindowCounter(limit=1, window_seconds=0.1)
        sw.check()
        time.sleep(0.15)
        result = sw.check()
        assert result.allowed is True


class TestFixedWindowCounter:
    def test_within_limit(self):
        fw = FixedWindowCounter(limit=5, window_seconds=1)
        result = fw.check()
        assert result.allowed is True

    def test_exceeds_limit(self):
        fw = FixedWindowCounter(limit=2, window_seconds=1)
        fw.check()
        fw.check()
        result = fw.check()
        assert result.allowed is False

    def test_window_reset(self):
        fw = FixedWindowCounter(limit=1, window_seconds=0.1)
        fw.check()
        time.sleep(0.15)
        result = fw.check()
        assert result.allowed is True


class TestRateLimiter:
    def test_default_limiter(self):
        rl = RateLimiter(strategy=RateLimitStrategy.TOKEN_BUCKET)
        result = rl.check("user-1", "per_user")
        assert result.allowed is True

    def test_custom_limit(self):
        rl = RateLimiter()
        rl.set_limit("endpoint:/api/v1/optimize", 3, window_seconds=1)
        for _ in range(3):
            rl.check("endpoint:/api/v1/optimize")
        result = rl.check("endpoint:/api/v1/optimize")
        assert result.allowed is False

    def test_stats(self):
        rl = RateLimiter()
        rl.check("u1")
        stats = rl.get_stats()
        assert stats['active_limiters'] >= 1

    def test_clear(self):
        rl = RateLimiter()
        rl.check("u1")
        rl.clear()
        stats = rl.get_stats()
        assert stats['active_limiters'] == 0

    def test_clear_specific(self):
        rl = RateLimiter()
        rl.check("u1")
        rl.check("u2")
        rl.clear("u1")
        stats = rl.get_stats()
        assert stats['active_limiters'] == 1

    def test_result_headers(self):
        rl = RateLimiter()
        result = rl.check("u1")
        headers = result.to_headers()
        assert 'X-RateLimit-Limit' in headers
        assert 'X-RateLimit-Remaining' in headers

    def test_retry_after_header(self):
        rl = RateLimiter()
        rl.set_limit("u1", 1, window_seconds=1)
        rl.check("u1")
        result = rl.check("u1")
        headers = result.to_headers()
        assert 'Retry-After' in headers

    def test_set_default_limit(self):
        rl = RateLimiter()
        rl.set_default_limit('global', 5000)
        assert rl._default_limits['global'] == 5000


# --- Request Logger Tests ---

from modules.observability.request_logger import (
    RequestLogger, RequestLog, redact_pii,
)


class TestRedactPII:
    def test_email(self):
        assert '[EMAIL]' in redact_pii('Contact me at test@example.com')

    def test_phone(self):
        assert '[PHONE]' in redact_pii('Call 555-123-4567')

    def test_api_key(self):
        assert '[REDACTED]' in redact_pii('api_key=sk_12345')

    def test_none(self):
        assert redact_pii(None) is None


class TestRequestLogger:
    def test_start_end_request(self):
        rl = RequestLogger(max_logs=100)
        req_id = rl.start_request("POST", "/v1/optimize", user_id="u1")
        rl.end_request(req_id, 200, response_size=1024)
        logs = rl.get_logs()
        assert len(logs) == 1
        assert logs[0]['status_code'] == 200

    def test_stats(self):
        rl = RequestLogger()
        req_id = rl.start_request("GET", "/health")
        rl.end_request(req_id, 200)
        stats = rl.get_stats()
        assert stats['total_requests'] == 1
        assert stats['error_count'] == 0

    def test_error_counting(self):
        rl = RequestLogger()
        req_id = rl.start_request("POST", "/v1/optimize")
        rl.end_request(req_id, 500)
        stats = rl.get_stats()
        assert stats['error_count'] == 1

    def test_filter_by_status(self):
        rl = RequestLogger()
        r1 = rl.start_request("GET", "/a")
        rl.end_request(r1, 200)
        r2 = rl.start_request("GET", "/b")
        rl.end_request(r2, 404)
        assert len(rl.get_logs(status_code=200)) == 1
        assert len(rl.get_logs(status_code=404)) == 1

    def test_filter_by_path(self):
        rl = RequestLogger()
        r1 = rl.start_request("GET", "/v1/optimize")
        rl.end_request(r1, 200)
        r2 = rl.start_request("GET", "/health")
        rl.end_request(r2, 200)
        assert len(rl.get_logs(path_pattern="optimize")) == 1

    def test_filter_by_user(self):
        rl = RequestLogger()
        r1 = rl.start_request("GET", "/a", user_id="u1")
        rl.end_request(r1, 200)
        r2 = rl.start_request("GET", "/b", user_id="u2")
        rl.end_request(r2, 200)
        assert len(rl.get_logs(user_id="u1")) == 1

    def test_active_requests(self):
        rl = RequestLogger()
        req_id = rl.start_request("GET", "/a")
        assert rl.get_stats()['active_requests'] == 1
        rl.end_request(req_id, 200)
        assert rl.get_stats()['active_requests'] == 0

    def test_log_dict(self):
        rl = RequestLogger()
        r1 = rl.start_request("POST", "/test", user_id="u1", metadata={'key': 'val'})
        rl.end_request(r1, 201)
        logs = rl.get_logs()
        assert logs[0]['user_id'] == 'u1'
        assert logs[0]['metadata'] == {'key': 'val'}


# --- Webhook Tests ---

from modules.webhooks.webhook_manager import (
    WebhookManager, WebhookEvent, WebhookStatus,
)


class TestWebhookManager:
    def test_create_webhook(self):
        wm = WebhookManager()
        wh = wm.create_webhook(
            url="http://example.com/hook",
            events=[WebhookEvent.OPTIMIZATION_COMPLETED],
        )
        assert wh.id.startswith("wh-")
        assert wh.status == WebhookStatus.ACTIVE

    def test_list_webhooks(self):
        wm = WebhookManager()
        wm.create_webhook(url="http://a.com", events=["a"])
        wm.create_webhook(url="http://b.com", events=["b"], organization_id="org1")
        assert len(wm.list_webhooks()) == 2
        assert len(wm.list_webhooks(organization_id="org1")) == 1

    def test_delete_webhook(self):
        wm = WebhookManager()
        wh = wm.create_webhook(url="http://a.com", events=["a"])
        assert wm.delete_webhook(wh.id) is True
        assert wm.get_webhook(wh.id) is None

    def test_toggle_webhook(self):
        wm = WebhookManager()
        wh = wm.create_webhook(url="http://a.com", events=["a"])
        wm.toggle_webhook(wh.id, False)
        assert wh.status == WebhookStatus.INACTIVE

    def test_signature(self):
        wm = WebhookManager()
        secret = "test-secret-123"
        payload = '{"event": "test"}'
        sig = wm.generate_signature(secret, payload)
        assert wm.verify_signature(secret, payload, sig) is True

    def test_stats(self):
        wm = WebhookManager()
        wm.create_webhook(url="http://a.com", events=["a"])
        stats = wm.get_stats()
        assert stats['total_webhooks'] == 1
        assert stats['active_webhooks'] == 1


# --- Model Registry Tests ---

from modules.model_registry.model_registry import (
    ModelRegistry, ModelStatus, ModelStage,
)


class TestModelRegistry:
    def test_register_model(self):
        reg = ModelRegistry()
        m = reg.register_model("gpt-custom", description="Custom GPT")
        assert m.model_id.startswith("model-")
        assert len(m.versions) == 1

    def test_add_version(self):
        reg = ModelRegistry()
        m = reg.register_model("test-model", version="1.0.0")
        v = reg.add_version(m.model_id, "2.0.0")
        assert v.version == "2.0.0"
        assert len(m.versions) == 2

    def test_promote_version(self):
        reg = ModelRegistry()
        m = reg.register_model("test-model")
        v = reg.promote_version(m.model_id, "1.0.0", ModelStage.PRODUCTION)
        assert v.stage == ModelStage.PRODUCTION
        assert v.status == ModelStatus.ACTIVE

    def test_deprecate_version(self):
        reg = ModelRegistry()
        m = reg.register_model("test-model")
        v = reg.deprecate_version(m.model_id, "1.0.0")
        assert v.status == ModelStatus.DEPRECATED

    def test_compare_versions(self):
        reg = ModelRegistry()
        m = reg.register_model("test-model")
        reg.add_version(m.model_id, "2.0.0", metrics={'accuracy': 0.95})
        comp = reg.compare_versions(m.model_id, "1.0.0", "2.0.0")
        assert comp is not None
        assert 'accuracy' in comp['metrics_comparison']

    def test_list_models(self):
        reg = ModelRegistry()
        reg.register_model("a", organization_id="org1")
        reg.register_model("b")
        assert len(reg.list_models()) == 2
        assert len(reg.list_models(organization_id="org1")) == 1

    def test_stats(self):
        reg = ModelRegistry()
        reg.register_model("a")
        reg.register_model("b")
        stats = reg.get_stats()
        assert stats['total_models'] == 2

    def test_latest_version(self):
        reg = ModelRegistry()
        m = reg.register_model("test-model", version="1.0.0")
        reg.add_version(m.model_id, "2.0.0")
        assert m.latest_version.version == "2.0.0"


# --- A/B Testing Tests ---

from modules.ab_testing.ab_testing import (
    ABTestManager, ExperimentStatus, VariantStatus,
)


class TestABTestManager:
    def test_create_experiment(self):
        abm = ABTestManager()
        exp = abm.create_experiment(
            name="Test Exp", description="Testing",
            variants=[
                {'name': 'control', 'weight': 0.5},
                {'name': 'treatment', 'weight': 0.5},
            ],
        )
        assert exp.experiment_id.startswith("exp-")
        assert len(exp.variants) == 2

    def test_start_stop(self):
        abm = ABTestManager()
        exp = abm.create_experiment(
            name="Test", description="",
            variants=[{'name': 'a'}, {'name': 'b'}],
        )
        abm.start_experiment(exp.experiment_id)
        assert exp.status == ExperimentStatus.RUNNING
        abm.stop_experiment(exp.experiment_id)
        assert exp.status == ExperimentStatus.COMPLETED

    def test_assign_variant_deterministic(self):
        abm = ABTestManager()
        exp = abm.create_experiment(
            name="Test", description="",
            variants=[{'name': 'a'}, {'name': 'b'}],
        )
        abm.start_experiment(exp.experiment_id)
        v1 = abm.assign_variant(exp.experiment_id, "user-123")
        v2 = abm.assign_variant(exp.experiment_id, "user-123")
        assert v1.variant_id == v2.variant_id

    def test_record_metrics(self):
        abm = ABTestManager()
        exp = abm.create_experiment(
            name="Test", description="",
            variants=[{'name': 'a'}, {'name': 'b'}],
        )
        abm.start_experiment(exp.experiment_id)
        v = abm.assign_variant(exp.experiment_id, "u1")
        abm.record_impression(exp.experiment_id, v.variant_id)
        abm.record_conversion(exp.experiment_id, v.variant_id, revenue=10.0)
        assert v.impressions == 1
        assert v.conversions == 1

    def test_significance(self):
        abm = ABTestManager()
        exp = abm.create_experiment(
            name="Test", description="",
            variants=[{'name': 'a'}, {'name': 'b'}],
        )
        abm.start_experiment(exp.experiment_id)
        va = exp.variants[0]
        vb = exp.variants[1]
        for _ in range(100):
            abm.record_impression(exp.experiment_id, va.variant_id)
            abm.record_impression(exp.experiment_id, vb.variant_id)
            if _ < 40:
                abm.record_conversion(exp.experiment_id, va.variant_id)
            else:
                abm.record_conversion(exp.experiment_id, vb.variant_id)
        result = abm.calculate_significance(exp.experiment_id)
        assert result is not None
        assert 'p_value' in result

    def test_declare_winner(self):
        abm = ABTestManager()
        exp = abm.create_experiment(
            name="Test", description="",
            variants=[{'name': 'a'}, {'name': 'b'}],
        )
        abm.start_experiment(exp.experiment_id)
        abm.declare_winner(exp.experiment_id, exp.variants[1].variant_id)
        assert exp.winner_variant_id == exp.variants[1].variant_id
        assert exp.status == ExperimentStatus.COMPLETED

    def test_list_experiments(self):
        abm = ABTestManager()
        abm.create_experiment("a", "", [{'name': 'a'}, {'name': 'b'}], organization_id="org1")
        abm.create_experiment("b", "", [{'name': 'a'}, {'name': 'b'}])
        assert len(abm.list_experiments()) == 2
        assert len(abm.list_experiments(organization_id="org1")) == 1

    def test_stats(self):
        abm = ABTestManager()
        abm.create_experiment("a", "", [{'name': 'a'}])
        stats = abm.get_stats()
        assert stats['total_experiments'] == 1


# --- Prompt Versioning Tests ---

from modules.prompt_versioning.prompt_versioning import (
    PromptVersionManager, PromptStatus,
)


class TestPromptVersionManager:
    def test_create_prompt(self):
        pvm = PromptVersionManager()
        pt = pvm.create_prompt("greeting", "Hello {{name}}!")
        assert pt.prompt_id.startswith("prompt-")
        assert pt.active_version.content == "Hello {{name}}!"

    def test_create_version(self):
        pvm = PromptVersionManager()
        pt = pvm.create_prompt("greeting", "Hello {{name}}!")
        pv = pvm.create_version(pt.prompt_id, "Hi {{name}}!", changelog="Changed greeting")
        assert pv.version == 2
        assert pv.changelog == "Changed greeting"
        assert pt.active_version.version == 2
        assert pt.versions[0].status == PromptStatus.DEPRECATED

    def test_rollback(self):
        pvm = PromptVersionManager()
        pt = pvm.create_prompt("greeting", "v1 content")
        pvm.create_version(pt.prompt_id, "v2 content")
        pvm.create_version(pt.prompt_id, "v3 content")
        rolled = pvm.rollback(pt.prompt_id, 1)
        assert rolled.status == PromptStatus.ACTIVE
        assert pt.active_version.version == 1

    def test_render(self):
        pvm = PromptVersionManager()
        pt = pvm.create_prompt("greeting", "Hello {{name}}!")
        rendered = pvm.render(pt.prompt_id, {'name': 'World'})
        assert rendered == "Hello World!"

    def test_record_usage(self):
        pvm = PromptVersionManager()
        pt = pvm.create_prompt("test", "content")
        pvm.record_usage(pt.prompt_id, score=0.8)
        pvm.record_usage(pt.prompt_id, score=0.9)
        assert pt.active_version.usage_count == 2
        assert abs(pt.active_version.avg_score - 0.85) < 0.01

    def test_diff(self):
        pvm = PromptVersionManager()
        pt = pvm.create_prompt("test", "line1\nline2")
        pvm.create_version(pt.prompt_id, "line1\nline3")
        diff = pvm.get_diff(pt.prompt_id, 1, 2)
        assert diff['total_changes'] == 1
        assert diff['changes'][0]['old'] == 'line2'
        assert diff['changes'][0]['new'] == 'line3'

    def test_list_prompts(self):
        pvm = PromptVersionManager()
        pvm.create_prompt("a", "content", organization_id="org1")
        pvm.create_prompt("b", "content")
        assert len(pvm.list_prompts()) == 2
        assert len(pvm.list_prompts(organization_id="org1")) == 1

    def test_stats(self):
        pvm = PromptVersionManager()
        pvm.create_prompt("a", "content")
        pvm.create_prompt("b", "content")
        stats = pvm.get_stats()
        assert stats['total_prompts'] == 2


# --- SLA Monitor Tests ---

from modules.observability.sla_monitor import (
    SLAMonitor, SLAStatus, AlertSeverity,
)


class TestSLAMonitor:
    def test_create_sla(self):
        sm = SLAMonitor()
        sla = sm.create_sla(
            name="Latency", description="Avg latency < 500ms",
            metric="latency_ms", target_value=500.0, operator="<=",
        )
        assert sla.sla_id.startswith("sla-")

    def test_record_and_evaluate(self):
        sm = SLAMonitor()
        sla = sm.create_sla(
            name="Uptime", description="> 99%",
            metric="uptime", target_value=99.0, operator=">=",
        )
        sm.record_metric("uptime", 99.5)
        record = sm.evaluate_sla(sla.sla_id)
        assert record.status == SLAStatus.HEALTHY

    def test_breach(self):
        sm = SLAMonitor()
        sla = sm.create_sla(
            name="Latency", description="< 100ms",
            metric="latency_ms", target_value=100.0, operator="<=",
        )
        sm.record_metric("latency_ms", 200.0)
        record = sm.evaluate_sla(sla.sla_id)
        assert record.status == SLAStatus.VIOLATED

    def test_alert_triggered(self):
        sm = SLAMonitor()
        sla = sm.create_sla(
            name="Test", description="",
            metric="err_rate", target_value=0.01, operator="<=",
            severity=AlertSeverity.CRITICAL,
        )
        sm.record_metric("err_rate", 0.5)
        sm.evaluate_sla(sla.sla_id)
        alerts = sm.get_alerts(unresolved_only=True)
        assert len(alerts) == 1

    def test_resolve_alert(self):
        sm = SLAMonitor()
        sla = sm.create_sla("t", "", "err_rate", 0.01, "<=")
        sm.record_metric("err_rate", 0.5)
        sm.evaluate_sla(sla.sla_id)
        alerts = sm.get_alerts(unresolved_only=True)
        sm.resolve_alert(alerts[0]['alert_id'])
        assert len(sm.get_alerts(unresolved_only=True)) == 0

    def test_alert_handler(self):
        sm = SLAMonitor()
        triggered = []
        sm.register_alert_handler(lambda a: triggered.append(a))
        sla = sm.create_sla("t", "", "err", 0.01, "<=")
        sm.record_metric("err", 0.5)
        sm.evaluate_sla(sla.sla_id)
        assert len(triggered) == 1

    def test_dashboard(self):
        sm = SLAMonitor()
        sla = sm.create_sla("t", "", "uptime", 99.0, ">=")
        sm.record_metric("uptime", 99.5)
        sm.evaluate_sla(sla.sla_id)
        dash = sm.get_dashboard()
        assert dash['healthy'] == 1
        assert dash['compliance_rate'] == 100.0

    def test_list_slas(self):
        sm = SLAMonitor()
        sm.create_sla("a", "", "m", 1, ">=", organization_id="org1")
        sm.create_sla("b", "", "m", 1, ">=")
        assert len(sm.list_slas()) == 2
        assert len(sm.list_slas(organization_id="org1")) == 1


# --- Batch Processor Tests ---

from modules.optimization_brain.batch_processor import (
    BatchProcessor, BatchStatus, JobStatus,
)


class TestBatchProcessor:
    def test_create_batch(self):
        bp = BatchProcessor()
        batch = bp.create_batch(
            prompts=["prompt1", "prompt2", "prompt3"],
            model_adapter="test",
        )
        assert batch.batch_id.startswith("batch-")
        assert batch.total_jobs == 3

    def test_execute_batch(self):
        bp = BatchProcessor()

        def mock_optimize(prompt, context=None, model_adapter=None):
            class Result:
                optimized_prompt = f"optimized-{prompt}"
                compression_ratio = 0.5
                tokens_saved = 10
            return Result()

        batch = bp.create_batch(prompts=["p1", "p2"])
        result = bp.execute_batch(batch.batch_id, mock_optimize)
        assert result.status == BatchStatus.COMPLETED
        assert result.completed_jobs == 2
        assert all(j.status == JobStatus.COMPLETED for j in result.jobs)

    def test_progress(self):
        bp = BatchProcessor()
        batch = bp.create_batch(prompts=["p1", "p2", "p3"])

        def mock_optimize(**kwargs):
            class R:
                optimized_prompt = "x"
                compression_ratio = 1.0
                tokens_saved = 0
            return R()

        bp.execute_batch(batch.batch_id, mock_optimize)
        assert batch.progress == 1.0

    def test_cancel_batch(self):
        bp = BatchProcessor()
        batch = bp.create_batch(prompts=["p1"])
        bp.cancel_batch(batch.batch_id)
        assert batch.status == BatchStatus.CANCELLED

    def test_list_batches(self):
        bp = BatchProcessor()
        bp.create_batch(prompts=["p1"], organization_id="org1")
        bp.create_batch(prompts=["p2"])
        assert len(bp.list_batches()) == 2
        assert len(bp.list_batches(organization_id="org1")) == 1

    def test_stats(self):
        bp = BatchProcessor()
        bp.create_batch(prompts=["p1", "p2"])
        stats = bp.get_stats()
        assert stats['total_batches'] == 1
        assert stats['total_jobs'] == 2
