"""
Webhook System
Create, manage, and trigger webhooks with retry logic and signature verification.
"""

import time
import hmac
import hashlib
import json
import logging
import threading
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class WebhookEvent(str, Enum):
    OPTIMIZATION_COMPLETED = "optimization.completed"
    OPTIMIZATION_FAILED = "optimization.failed"
    BATCH_COMPLETED = "batch.completed"
    QUOTA_WARNING = "quota.warning"
    QUOTA_EXCEEDED = "quota.exceeded"
    MODEL_PROMOTED = "model.promoted"
    ALERT_TRIGGERED = "alert.triggered"
    USER_CREATED = "user.created"
    API_KEY_CREATED = "api_key.created"


class WebhookStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"


@dataclass
class WebhookDelivery:
    id: str
    webhook_id: str
    event: str
    payload: Dict[str, Any]
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None
    attempt: int = 1
    timestamp: float = 0.0
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'webhook_id': self.webhook_id,
            'event': self.event,
            'status_code': self.status_code,
            'attempt': self.attempt,
            'timestamp': self.timestamp,
            'duration_ms': self.duration_ms,
            'error': self.error,
        }


@dataclass
class Webhook:
    id: str
    url: str
    events: List[str]
    secret: str
    organization_id: Optional[str] = None
    user_id: Optional[str] = None
    status: WebhookStatus = WebhookStatus.ACTIVE
    max_retries: int = 3
    retry_delay_seconds: int = 5
    created_at: float = 0.0
    last_triggered_at: Optional[float] = None
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'url': self.url,
            'events': self.events,
            'organization_id': self.organization_id,
            'user_id': self.user_id,
            'status': self.status.value,
            'max_retries': self.max_retries,
            'created_at': self.created_at,
            'last_triggered_at': self.last_triggered_at,
            'failure_count': self.failure_count,
        }


class WebhookManager:
    """
    Manages webhooks with delivery, retry, and signature verification.
    """

    def __init__(self):
        self._webhooks: Dict[str, Webhook] = {}
        self._deliveries: List[WebhookDelivery] = []
        self._delivery_queue: List[tuple] = []
        self._lock = threading.Lock()

    def create_webhook(
        self,
        url: str,
        events: List[str],
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        max_retries: int = 3,
    ) -> Webhook:
        import secrets
        webhook_id = f"wh-{secrets.token_hex(8)}"
        secret = secrets.token_hex(32)

        webhook = Webhook(
            id=webhook_id, url=url, events=events, secret=secret,
            organization_id=organization_id, user_id=user_id,
            max_retries=max_retries, created_at=time.time(),
        )
        with self._lock:
            self._webhooks[webhook_id] = webhook
        return webhook

    def get_webhook(self, webhook_id: str) -> Optional[Webhook]:
        return self._webhooks.get(webhook_id)

    def list_webhooks(self, organization_id: Optional[str] = None) -> List[Webhook]:
        with self._lock:
            hooks = list(self._webhooks.values())
        if organization_id:
            hooks = [h for h in hooks if h.organization_id == organization_id]
        return hooks

    def delete_webhook(self, webhook_id: str) -> bool:
        with self._lock:
            return self._webhooks.pop(webhook_id, None) is not None

    def toggle_webhook(self, webhook_id: str, active: bool) -> Optional[Webhook]:
        webhook = self._webhooks.get(webhook_id)
        if webhook:
            webhook.status = WebhookStatus.ACTIVE if active else WebhookStatus.INACTIVE
        return webhook

    def generate_signature(self, secret: str, payload: str) -> str:
        return hmac.new(
            secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

    def verify_signature(self, secret: str, payload: str, signature: str) -> bool:
        expected = self.generate_signature(secret, payload)
        return hmac.compare_digest(expected, signature)

    def _deliver(self, webhook: Webhook, event: str, payload: Dict[str, Any]) -> WebhookDelivery:
        import secrets
        delivery_id = f"del-{secrets.token_hex(8)}"
        body = json.dumps({
            'event': event,
            'data': payload,
            'webhook_id': webhook.id,
            'timestamp': time.time(),
        })

        signature = self.generate_signature(webhook.secret, body)
        headers = {
            'Content-Type': 'application/json',
            'X-Webhook-Signature': f'sha256={signature}',
            'X-Webhook-Event': event,
            'X-Webhook-ID': webhook.id,
        }

        delivery = WebhookDelivery(
            id=delivery_id, webhook_id=webhook.id,
            event=event, payload=payload, timestamp=time.time(),
        )

        for attempt in range(1, webhook.max_retries + 1):
            delivery.attempt = attempt
            start = time.time()
            try:
                req = urllib.request.Request(
                    webhook.url,
                    data=body.encode(),
                    headers=headers,
                    method='POST',
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    delivery.status_code = resp.status
                    delivery.response_body = resp.read().decode()
                    delivery.duration_ms = (time.time() - start) * 1000
                    webhook.last_triggered_at = time.time()
                    webhook.failure_count = 0
                    logger.info(f"Webhook delivered: {webhook.id} -> {delivery.status_code}")
                    break
            except Exception as e:
                delivery.status_code = None
                delivery.error = str(e)
                delivery.duration_ms = (time.time() - start) * 1000
                webhook.failure_count += 1
                logger.warning(f"Webhook delivery attempt {attempt} failed: {e}")
                if attempt < webhook.max_retries:
                    time.sleep(webhook.retry_delay_seconds)

        if delivery.status_code and delivery.status_code >= 400:
            webhook.failure_count += 1
        if webhook.failure_count >= 5:
            webhook.status = WebhookStatus.FAILED

        with self._lock:
            self._deliveries.append(delivery)
        return delivery

    def trigger(self, event: str, payload: Dict[str, Any]) -> List[WebhookDelivery]:
        deliveries = []
        with self._lock:
            active_hooks = [
                h for h in self._webhooks.values()
                if h.status == WebhookStatus.ACTIVE and event in h.events
            ]

        for webhook in active_hooks:
            delivery = self._deliver(webhook, event, payload)
            deliveries.append(delivery)
        return deliveries

    def trigger_async(self, event: str, payload: Dict[str, Any]):
        thread = threading.Thread(
            target=self.trigger, args=(event, payload), daemon=True
        )
        thread.start()

    def get_deliveries(
        self,
        webhook_id: Optional[str] = None,
        event: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            deliveries = list(self._deliveries)
        if webhook_id:
            deliveries = [d for d in deliveries if d.webhook_id == webhook_id]
        if event:
            deliveries = [d for d in deliveries if d.event == event]
        return [d.to_dict() for d in deliveries[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._deliveries)
            success = sum(1 for d in self._deliveries if d.status_code and 200 <= d.status_code < 300)
            failed = sum(1 for d in self._deliveries if d.error)
            return {
                'total_webhooks': len(self._webhooks),
                'active_webhooks': sum(1 for h in self._webhooks.values() if h.status == WebhookStatus.ACTIVE),
                'total_deliveries': total,
                'successful_deliveries': success,
                'failed_deliveries': failed,
                'success_rate': success / total if total else 0.0,
            }
