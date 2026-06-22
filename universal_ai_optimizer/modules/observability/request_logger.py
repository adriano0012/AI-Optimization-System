"""
Request/Response Logging Middleware
Structured logging for all API traffic with PII redaction.
"""

import time
import uuid
import json
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from collections import deque

logger = logging.getLogger(__name__)

PII_PATTERNS = [
    (re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b'), '[EMAIL]'),
    (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), '[PHONE]'),
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN]'),
    (re.compile(r'\b(?:\d[ -]*?){13,16}\b'), '[CARD]'),
    (re.compile(r'(api[_-]?key|token|secret|password)[\s:=]+\S+', re.IGNORECASE), r'\1=[REDACTED]'),
]


def redact_pii(text: str, patterns: Optional[List] = None) -> str:
    if not text:
        return text
    for pattern, replacement in (patterns or PII_PATTERNS):
        text = pattern.sub(replacement, text)
    return text


@dataclass
class RequestLog:
    request_id: str
    method: str
    path: str
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    start_time: float = 0.0
    end_time: Optional[float] = None
    status_code: Optional[int] = None
    response_size: Optional[int] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def to_dict(self, redact: bool = True) -> Dict[str, Any]:
        d = {
            'request_id': self.request_id,
            'method': self.method,
            'path': self.path,
            'user_id': self.user_id,
            'org_id': self.org_id,
            'client_ip': self.client_ip,
            'status_code': self.status_code,
            'duration_ms': round(self.duration_ms, 2),
            'response_size': self.response_size,
            'error': self.error,
            'timestamp': self.start_time,
        }
        if self.metadata:
            d['metadata'] = self.metadata
        return d


class RequestLogger:
    """
    Structured request/response logger.
    Captures all API traffic with configurable retention.
    """

    def __init__(self, max_logs: int = 10000, redact_pii: bool = True):
        self.max_logs = max_logs
        self.redact_pii = redact_pii
        self._logs: deque = deque(maxlen=max_logs)
        self._active_requests: Dict[str, RequestLog] = {}
        self._stats = {
            'total_requests': 0,
            'error_count': 0,
            'total_duration_ms': 0.0,
        }

    def start_request(
        self,
        method: str,
        path: str,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        log = RequestLog(
            request_id=request_id,
            method=method,
            path=path,
            user_id=user_id,
            org_id=org_id,
            client_ip=client_ip,
            user_agent=user_agent,
            start_time=time.time(),
            metadata=metadata or {},
        )
        self._active_requests[request_id] = log
        self._stats['total_requests'] += 1
        return request_id

    def end_request(
        self,
        request_id: str,
        status_code: int,
        response_size: Optional[int] = None,
        error: Optional[str] = None,
    ):
        if request_id not in self._active_requests:
            return

        log = self._active_requests.pop(request_id)
        log.end_time = time.time()
        log.status_code = status_code
        log.response_size = response_size
        log.error = error

        if status_code >= 400:
            self._stats['error_count'] += 1
        self._stats['total_duration_ms'] += log.duration_ms

        self._logs.append(log)

        log_dict = log.to_dict(redact=self.redact_pii)
        if self.redact_pii:
            log_str = redact_pii(json.dumps(log_dict))
            try:
                log_dict = json.loads(log_str)
            except (json.JSONDecodeError, ValueError):
                pass

        if status_code >= 500:
            logger.error("Request completed", extra=log_dict)
        elif status_code >= 400:
            logger.warning("Request completed", extra=log_dict)
        else:
            logger.info("Request completed", extra=log_dict)

    def get_logs(
        self,
        user_id: Optional[str] = None,
        status_code: Optional[int] = None,
        path_pattern: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        logs = list(self._logs)

        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        if status_code:
            logs = [l for l in logs if l.status_code == status_code]
        if path_pattern:
            logs = [l for l in logs if path_pattern in l.path]

        return [l.to_dict(redact=self.redact_pii) for l in logs[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        total = self._stats['total_requests']
        return {
            'total_requests': total,
            'error_count': self._stats['error_count'],
            'error_rate': self._stats['error_count'] / total if total else 0.0,
            'avg_duration_ms': (
                self._stats['total_duration_ms'] / total if total else 0.0
            ),
            'active_requests': len(self._active_requests),
            'log_buffer_size': len(self._logs),
        }
