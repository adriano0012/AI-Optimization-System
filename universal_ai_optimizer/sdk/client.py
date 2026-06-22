"""
Universal AI Optimizer - Python SDK Client
Production-grade client for the UAI Optimizer API.
"""

import json
import time
import threading
from typing import Optional, Dict, Any, List, Iterator
from dataclasses import dataclass, field


@dataclass
class SDKConfig:
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    jwt_token: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class OptimizationResult:
    id: str
    original_prompt: str
    optimized_prompt: str
    compression_ratio: float
    tokens_saved: int
    latency_ms: float
    task_type: Optional[str] = None
    model_used: Optional[str] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class UAIOptimizerClient:
    """
    Python SDK client for Universal AI Optimizer API.
    Supports JWT and API key authentication with automatic retries.
    """

    def __init__(self, config: Optional[SDKConfig] = None, **kwargs):
        self.config = config or SDKConfig(**kwargs)
        self._session = None
        self._lock = threading.Lock()

    def _get_headers(self) -> Dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self.config.jwt_token:
            headers['Authorization'] = f'Bearer {self.config.jwt_token}'
        elif self.config.api_key:
            headers['X-API-Key'] = self.config.api_key
        return headers

    def _request(
        self, method: str, path: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        import urllib.request
        import urllib.error
        import urllib.parse

        url = f"{self.config.base_url}{path}"
        if params:
            url += '?' + urllib.parse.urlencode(params)

        body = json.dumps(data).encode() if data else None
        headers = self._get_headers()

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                req = urllib.request.Request(
                    url, data=body, headers=headers, method=method,
                )
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                error_body = e.read().decode() if e.fp else str(e)
                last_error = {
                    'status_code': e.code, 'error': error_body,
                }
                if e.code < 500:
                    raise UAIClientError(
                        f"HTTP {e.code}: {error_body}",
                        status_code=e.code, response=last_error,
                    )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
            except urllib.error.URLError as e:
                last_error = {'error': str(e)}
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        raise UAIClientError(
            f"Request failed after {self.config.max_retries} retries",
            response=last_error,
        )

    # --- Auth ---

    def register(
        self, email: str, password: str, name: str
    ) -> Dict[str, Any]:
        result = self._request('POST', '/v1/auth/register', {
            'email': email, 'password': password, 'name': name,
        })
        self.config.jwt_token = result.get('access_token')
        return result

    def login(self, email: str, password: str) -> Dict[str, Any]:
        result = self._request('POST', '/v1/auth/login', {
            'email': email, 'password': password,
        })
        self.config.jwt_token = result.get('access_token')
        return result

    def create_api_key(self, name: str, rate_limit: int = 1000) -> Dict[str, Any]:
        return self._request('POST', '/v1/auth/api-keys', {
            'name': name, 'rate_limit': rate_limit,
        })

    def list_api_keys(self) -> List[Dict[str, Any]]:
        return self._request('GET', '/v1/auth/api-keys')

    # --- Optimize ---

    def optimize(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        model_adapter: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> OptimizationResult:
        result = self._request('POST', '/v1/optimize', {
            'prompt': prompt,
            'context': context or {},
            'model_adapter': model_adapter,
            'task_type': task_type,
        })
        return OptimizationResult(
            id=result.get('id', ''),
            original_prompt=result.get('original_prompt', prompt),
            optimized_prompt=result.get('optimized_prompt', prompt),
            compression_ratio=result.get('compression_ratio', 1.0),
            tokens_saved=result.get('tokens_saved', 0),
            latency_ms=result.get('latency_ms', 0),
            task_type=result.get('task_type'),
            model_used=result.get('model_used'),
            success=result.get('success', True),
            metadata=result.get('metadata', {}),
        )

    def optimize_batch(
        self,
        prompts: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None,
        model_adapter: Optional[str] = None,
        task_type: Optional[str] = None,
        max_concurrency: int = 4,
    ) -> Dict[str, Any]:
        return self._request('POST', '/v1/optimize/batch', {
            'prompts': prompts,
            'contexts': contexts or [],
            'model_adapter': model_adapter,
            'task_type': task_type,
            'max_concurrency': max_concurrency,
        })

    # --- Organizations ---

    def create_organization(self, name: str, plan: str = "free") -> Dict[str, Any]:
        return self._request('POST', '/v1/organizations', {
            'name': name, 'plan': plan,
        })

    def get_organization(self, org_id: str) -> Dict[str, Any]:
        return self._request('GET', f'/v1/organizations/{org_id}')

    # --- Webhooks ---

    def create_webhook(
        self, url: str, events: List[str], max_retries: int = 3
    ) -> Dict[str, Any]:
        return self._request('POST', '/v1/webhooks', {
            'url': url, 'events': events, 'max_retries': max_retries,
        })

    def list_webhooks(self) -> List[Dict[str, Any]]:
        return self._request('GET', '/v1/webhooks')

    def delete_webhook(self, webhook_id: str) -> bool:
        self._request('DELETE', f'/v1/webhooks/{webhook_id}')
        return True

    # --- Model Registry ---

    def register_model(
        self, name: str, description: str = "", version: str = "1.0.0",
        artifact_uri: Optional[str] = None, metrics: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        return self._request('POST', '/v1/models', {
            'name': name, 'description': description,
            'version': version, 'artifact_uri': artifact_uri,
            'metrics': metrics or {},
        })

    def list_models(self) -> List[Dict[str, Any]]:
        return self._request('GET', '/v1/models')

    def promote_model(
        self, model_id: str, version: str, stage: str
    ) -> Dict[str, Any]:
        return self._request('POST', f'/v1/models/{model_id}/promote', {
            'version': version, 'stage': stage,
        })

    # --- A/B Testing ---

    def create_experiment(
        self, name: str, description: str,
        variants: List[Dict[str, Any]],
        target_metric: str = "conversion_rate",
    ) -> Dict[str, Any]:
        return self._request('POST', '/v1/experiments', {
            'name': name, 'description': description,
            'variants': variants, 'target_metric': target_metric,
        })

    def list_experiments(self) -> List[Dict[str, Any]]:
        return self._request('GET', '/v1/experiments')

    # --- Prompt Versioning ---

    def create_prompt(
        self, name: str, content: str, description: str = "",
    ) -> Dict[str, Any]:
        return self._request('POST', '/v1/prompts', {
            'name': name, 'content': content, 'description': description,
        })

    def list_prompts(self) -> List[Dict[str, Any]]:
        return self._request('GET', '/v1/prompts')

    def render_prompt(
        self, prompt_id: str, variables: Optional[Dict[str, str]] = None
    ) -> str:
        result = self._request('POST', f'/v1/prompts/{prompt_id}/render', {
            'variables': variables or {},
        })
        return result.get('content', '')

    # --- SLA ---

    def create_sla(
        self, name: str, description: str, metric: str,
        target_value: float, operator: str = ">=",
    ) -> Dict[str, Any]:
        return self._request('POST', '/v1/slas', {
            'name': name, 'description': description,
            'metric': metric, 'target_value': target_value,
            'operator': operator,
        })

    def get_sla_dashboard(self) -> Dict[str, Any]:
        return self._request('GET', '/v1/slas/dashboard')

    # --- Admin ---

    def health_check(self) -> Dict[str, Any]:
        return self._request('GET', '/health')

    def get_metrics(self) -> Dict[str, Any]:
        return self._request('GET', '/v1/admin/metrics')

    def get_audit_logs(self, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        return self._request('GET', '/v1/admin/audit-logs', params={
            'page': page, 'page_size': page_size,
        })

    def get_tracing_stats(self) -> Dict[str, Any]:
        return self._request('GET', '/v1/admin/tracing/stats')


class UAIClientError(Exception):
    def __init__(
        self, message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response = response
