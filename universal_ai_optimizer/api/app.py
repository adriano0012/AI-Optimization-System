"""
FastAPI Application
REST API with OpenAPI, streaming, authentication.
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request, Header, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from api.schemas import (
    OptimizeRequest, OptimizeResponse, OptimizeStreamChunk,
    UserCreate, UserLogin, TokenResponse,
    ApiKeyCreate, ApiKeyResponse,
    OrganizationCreate, OrganizationResponse,
    HealthResponse, ErrorResponse, PaginatedResponse,
    AuditLogResponse, QuotaResponse,
)
from api.models import DatabaseManager
from api.auth import AuthManager
from modules.observability.tracing import (
    TracingManager, RequestTracer,
)

logger = logging.getLogger(__name__)

DB_URL = os.environ.get("UAI_DATABASE_URL", "sqlite:///data/uai_optimizer.db")
START_TIME = time.time()
VERSION = "1.0.0"

security = HTTPBearer(auto_error=False)

db_manager: Optional[DatabaseManager] = None
auth_manager: Optional[AuthManager] = None
optimizer_instance = None
tracing_manager: Optional[TracingManager] = None
request_tracer: Optional[RequestTracer] = None
webhook_manager = None
model_registry = None
ab_test_manager = None
prompt_version_manager = None
sla_monitor = None
batch_processor = None
rate_limiter = None
request_logger = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_manager, auth_manager, optimizer_instance, tracing_manager, request_tracer
    global webhook_manager, model_registry, ab_test_manager, prompt_version_manager
    global sla_monitor, batch_processor, rate_limiter, request_logger

    os.makedirs("data", exist_ok=True)
    db_manager = DatabaseManager(DB_URL)
    tracing_manager = TracingManager()
    request_tracer = RequestTracer(tracing_manager)

    from modules.webhooks.webhook_manager import WebhookManager
    from modules.model_registry.model_registry import ModelRegistry
    from modules.ab_testing.ab_testing import ABTestManager
    from modules.prompt_versioning.prompt_versioning import PromptVersionManager
    from modules.observability.sla_monitor import SLAMonitor
    from modules.optimization_brain.batch_processor import BatchProcessor
    from modules.security.rate_limiter import RateLimiter, RateLimitStrategy
    from modules.observability.request_logger import RequestLogger

    webhook_manager = WebhookManager()
    model_registry = ModelRegistry()
    ab_test_manager = ABTestManager()
    prompt_version_manager = PromptVersionManager()
    sla_monitor = SLAMonitor()
    batch_processor = BatchProcessor()
    rate_limiter = RateLimiter(strategy=RateLimitStrategy.TOKEN_BUCKET)
    request_logger = RequestLogger()

    # Run Alembic migrations on startup if UAI_AUTO_MIGRATE=true
    if os.environ.get("UAI_AUTO_MIGRATE", "false").lower() == "true":
        from alembic.config import Config as AlembicConfig
        from alembic import command as alembic_cmd
        alembic_cfg = AlembicConfig("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", DB_URL)
        alembic_cmd.upgrade(alembic_cfg, "head")
        logger.info("Alembic migrations applied")
    else:
        db_manager.create_tables()

    auth_manager = AuthManager(db_manager)

    from universal_ai_optimizer.core.optimizer import UniversalAIOptimizer
    optimizer_instance = UniversalAIOptimizer()

    logger.info("UAI Optimizer API started")
    yield

    logger.info("UAI Optimizer API shutting down")


app = FastAPI(
    title="Universal AI Optimizer API",
    description="Production-grade LLM inference optimization middleware",
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("UAI_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Dependency Injection ---

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    if x_api_key and auth_manager:
        key_info = auth_manager.validate_api_key(x_api_key)
        if key_info:
            return {'type': 'api_key', **key_info}

    if credentials and auth_manager:
        payload = auth_manager.verify_jwt_token(credentials.credentials)
        if payload:
            return {'type': 'jwt', **payload}

    return {'type': 'anonymous', 'role': 'member'}


def require_role(role: str):
    def checker(user=Depends(get_current_user)):
        if user.get('role') not in [role, 'admin']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {role}"
            )
        return user
    return checker


def log_audit(action: str, user: Dict, resource_type: str = None,
              resource_id: str = None, details: Dict = None,
              ip_address: str = None):
    if db_manager:
        session = db_manager.get_session()
        try:
            from api.models import AuditLog
            log = AuditLog(
                user_id=user.get('sub') or user.get('id'),
                organization_id=user.get('org_id') or user.get('organization_id'),
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details,
                ip_address=ip_address,
            )
            session.add(log)
            session.commit()
        except Exception as e:
            logger.warning(f"Audit log failed: {e}")
        finally:
            session.close()


# --- Health & Metrics ---

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    checks = {}
    if db_manager:
        try:
            session = db_manager.get_session()
            session.execute("SELECT 1")
            session.close()
            checks['database'] = 'healthy'
        except Exception:
            checks['database'] = 'unhealthy'

    overall = 'healthy' if all(v == 'healthy' for v in checks.values()) else 'degraded'
    return HealthResponse(
        status=overall,
        version=VERSION,
        uptime_seconds=time.time() - START_TIME,
        checks=checks,
    )


@app.get("/ready", tags=["System"])
async def readiness_check():
    return {"status": "ready"}


@app.get("/version", tags=["System"])
async def get_version():
    return {"version": VERSION, "name": "universal-ai-optimizer"}


# --- Auth Endpoints ---

@app.post("/v1/auth/register", response_model=TokenResponse, tags=["Auth"])
async def register(user: UserCreate):
    if not auth_manager:
        raise HTTPException(status_code=503, detail="Auth not configured")
    try:
        created = auth_manager.create_user(user.email, user.password, user.name)
        token = auth_manager.create_jwt_token(created['id'], role=created['role'])
        return TokenResponse(
            access_token=token, user_id=created['id'],
            organization_id=created.get('organization_id'),
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/v1/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(credentials: UserLogin):
    if not auth_manager:
        raise HTTPException(status_code=503, detail="Auth not configured")
    user = auth_manager.authenticate_user(credentials.email, credentials.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = auth_manager.create_jwt_token(user['id'], role=user['role'])
    return TokenResponse(
        access_token=token, user_id=user['id'],
        organization_id=user.get('organization_id'),
    )


@app.post("/v1/auth/api-keys", response_model=ApiKeyResponse, tags=["Auth"])
async def create_api_key(
    req: ApiKeyCreate,
    user=Depends(get_current_user),
):
    if not auth_manager:
        raise HTTPException(status_code=503, detail="Auth not configured")
    key_data = auth_manager.generate_api_key(
        name=req.name,
        user_id=user.get('sub') or user.get('id'),
        rate_limit=req.rate_limit,
    )
    log_audit("api_key.created", user, "api_key", key_data['id'])
    return ApiKeyResponse(**key_data)


@app.get("/v1/auth/api-keys", tags=["Auth"])
async def list_api_keys(user=Depends(get_current_user)):
    if not db_manager:
        return []
    session = db_manager.get_session()
    try:
        from api.models import ApiKey
        user_id = user.get('sub') or user.get('id')
        keys = session.query(ApiKey).filter_by(user_id=user_id, is_active=True).all()
        return [k.to_dict() for k in keys]
    finally:
        session.close()


# --- Core Optimization Endpoint ---

@app.post("/v1/optimize", response_model=OptimizeResponse, tags=["Optimize"])
async def optimize(
    req: OptimizeRequest,
    user=Depends(get_current_user),
    request: Request = None,
):
    if not optimizer_instance:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")

    user_id = user.get('sub') or user.get('id')

    # Start trace span
    span = None
    if request_tracer:
        span = request_tracer.trace_request("POST", "/v1/optimize", user_id=user_id)

    start = time.time()
    try:
        result = optimizer_instance.optimize(
            prompt=req.prompt,
            context=req.context,
            model_adapter=req.model_adapter,
        )
        latency_ms = (time.time() - start) * 1000

        # Record span attributes
        if span:
            span.set_attribute('task_type', req.task_type or 'unknown')
            span.set_attribute('compression_ratio',
                               getattr(result, 'compression_ratio', 1.0))
            span.set_attribute('tokens_saved',
                               getattr(result, 'tokens_saved', 0))
            span.set_attribute('latency_ms', latency_ms)
            span.set_status("OK")
            tracing_manager.end_span(span.span_id)

        log_audit(
            "optimize", user, "optimization", None,
            {'task_type': req.task_type, 'prompt_length': len(req.prompt)},
            ip_address=request.client.host if request else None,
        )

        return OptimizeResponse(
            id=str(uuid.uuid4()),
            original_prompt=result.original_prompt,
            optimized_prompt=result.optimized_prompt or result.original_prompt,
            compression_ratio=getattr(result, 'compression_ratio', 1.0),
            tokens_saved=getattr(result, 'tokens_saved', 0),
            latency_ms=latency_ms,
            task_type=req.task_type,
            model_used=req.model_adapter,
            success=True,
        )
    except Exception as e:
        if span:
            span.set_status("ERROR", str(e))
            tracing_manager.end_span(span.span_id)
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/optimize/stream", tags=["Optimize"])
async def optimize_stream(
    req: OptimizeRequest,
    user=Depends(get_current_user),
):
    if not optimizer_instance:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")

    async def generate():
        import asyncio
        start = time.time()

        yield OptimizeStreamChunk(
            chunk_type="start",
            data={'prompt_length': len(req.prompt)},
            timestamp=time.time(),
        ).model_dump_json() + "\n"

        try:
            result = await asyncio.to_thread(
                optimizer_instance.optimize, req.prompt, req.context, req.model_adapter
            )
            latency_ms = (time.time() - start) * 1000

            yield OptimizeStreamChunk(
                chunk_type="result",
                data={
                    'original_prompt': result.original_prompt,
                    'optimized_prompt': getattr(result, 'optimized_prompt', result.original_prompt),
                    'compression_ratio': getattr(result, 'compression_ratio', 1.0),
                    'tokens_saved': getattr(result, 'tokens_saved', 0),
                    'latency_ms': latency_ms,
                },
                timestamp=time.time(),
            ).model_dump_json() + "\n"

        except Exception as e:
            yield OptimizeStreamChunk(
                chunk_type="error",
                data={'error': str(e)},
                timestamp=time.time(),
            ).model_dump_json() + "\n"

        yield OptimizeStreamChunk(
            chunk_type="done",
            data={},
            timestamp=time.time(),
        ).model_dump_json() + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# --- Organization Endpoints ---

@app.post("/v1/organizations", response_model=OrganizationResponse, tags=["Organizations"])
async def create_organization(
    req: OrganizationCreate,
    user=Depends(require_role('admin')),
):
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not configured")
    session = db_manager.get_session()
    try:
        from api.models import Organization
        org = Organization(name=req.name, plan=req.plan.value)
        session.add(org)
        session.commit()
        return OrganizationResponse(**org.to_dict())
    finally:
        session.close()


@app.get("/v1/organizations/{org_id}", response_model=OrganizationResponse, tags=["Organizations"])
async def get_organization(org_id: str, user=Depends(get_current_user)):
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not configured")
    session = db_manager.get_session()
    try:
        from api.models import Organization
        org = session.query(Organization).filter_by(id=org_id).first()
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        return OrganizationResponse(**org.to_dict())
    finally:
        session.close()


# --- Admin Endpoints ---

@app.get("/v1/admin/audit-logs", response_model=PaginatedResponse, tags=["Admin"])
async def list_audit_logs(
    page: int = 1, page_size: int = 20,
    user=Depends(require_role('admin')),
):
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not configured")
    session = db_manager.get_session()
    try:
        from api.models import AuditLog
        query = session.query(AuditLog)
        total = query.count()
        logs = query.order_by(AuditLog.created_at.desc()).offset(
            (page - 1) * page_size
        ).limit(page_size).all()
        return PaginatedResponse(
            items=[l.to_dict() for l in logs],
            total=total, page=page, page_size=page_size,
            total_pages=(total + page_size - 1) // page_size,
        )
    finally:
        session.close()


@app.get("/v1/admin/quotas/{org_id}", tags=["Admin"])
async def get_quotas(org_id: str, user=Depends(require_role('admin'))):
    if not db_manager:
        return []
    session = db_manager.get_session()
    try:
        from api.models import Quota
        quotas = session.query(Quota).filter_by(organization_id=org_id).all()
        return [q.to_dict() for q in quotas]
    finally:
        session.close()


@app.get("/v1/admin/tracing/stats", tags=["Admin"])
async def get_tracing_stats(user=Depends(require_role('admin'))):
    if not tracing_manager:
        return {}
    stats = tracing_manager.get_stats()
    slow_spans = tracing_manager.get_slow_spans(threshold_ms=1000)
    stats['slow_spans_count'] = len(slow_spans)
    stats['slow_span_names'] = list({s.name for s in slow_spans})
    return stats


# --- Webhook Endpoints ---

@app.post("/v1/webhooks", tags=["Webhooks"])
async def create_webhook(
    req: Request,
    user=Depends(get_current_user),
):
    body = await req.json()
    if not webhook_manager:
        raise HTTPException(status_code=503, detail="Webhook manager not configured")
    wh = webhook_manager.create_webhook(
        url=body['url'],
        events=body.get('events', []),
        organization_id=user.get('org_id'),
        user_id=user.get('sub') or user.get('id'),
        max_retries=body.get('max_retries', 3),
    )
    return wh.to_dict()


@app.get("/v1/webhooks", tags=["Webhooks"])
async def list_webhooks(user=Depends(get_current_user)):
    if not webhook_manager:
        return []
    org_id = user.get('org_id')
    return [w.to_dict() for w in webhook_manager.list_webhooks(organization_id=org_id)]


@app.delete("/v1/webhooks/{webhook_id}", tags=["Webhooks"])
async def delete_webhook(webhook_id: str, user=Depends(get_current_user)):
    if not webhook_manager:
        raise HTTPException(status_code=503, detail="Not configured")
    deleted = webhook_manager.delete_webhook(webhook_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return {"deleted": True}


# --- Model Registry Endpoints ---

@app.post("/v1/models", tags=["Models"])
async def register_model(
    req: Request,
    user=Depends(get_current_user),
):
    body = await req.json()
    if not model_registry:
        raise HTTPException(status_code=503, detail="Not configured")
    m = model_registry.register_model(
        name=body['name'],
        description=body.get('description', ''),
        version=body.get('version', '1.0.0'),
        artifact_uri=body.get('artifact_uri'),
        metrics=body.get('metrics', {}),
        owner_id=user.get('sub') or user.get('id'),
        organization_id=user.get('org_id'),
    )
    return m.to_dict()


@app.get("/v1/models", tags=["Models"])
async def list_models(user=Depends(get_current_user)):
    if not model_registry:
        return []
    return model_registry.list_models(organization_id=user.get('org_id'))


@app.post("/v1/models/{model_id}/promote", tags=["Models"])
async def promote_model(
    model_id: str,
    req: Request,
    user=Depends(require_role('admin')),
):
    body = await req.json()
    if not model_registry:
        raise HTTPException(status_code=503, detail="Not configured")
    from modules.model_registry.model_registry import ModelStage
    mv = model_registry.promote_version(model_id, body['version'], ModelStage(body['stage']))
    if not mv:
        raise HTTPException(status_code=404, detail="Model or version not found")
    return mv.to_dict()


# --- A/B Testing Endpoints ---

@app.post("/v1/experiments", tags=["Experiments"])
async def create_experiment(
    req: Request,
    user=Depends(get_current_user),
):
    body = await req.json()
    if not ab_test_manager:
        raise HTTPException(status_code=503, detail="Not configured")
    exp = ab_test_manager.create_experiment(
        name=body['name'],
        description=body.get('description', ''),
        variants=body['variants'],
        target_metric=body.get('target_metric', 'conversion_rate'),
        organization_id=user.get('org_id'),
    )
    return exp.to_dict()


@app.get("/v1/experiments", tags=["Experiments"])
async def list_experiments(user=Depends(get_current_user)):
    if not ab_test_manager:
        return []
    return ab_test_manager.list_experiments(organization_id=user.get('org_id'))


@app.post("/v1/experiments/{experiment_id}/start", tags=["Experiments"])
async def start_experiment(experiment_id: str, user=Depends(require_role('admin'))):
    if not ab_test_manager:
        raise HTTPException(status_code=503, detail="Not configured")
    exp = ab_test_manager.start_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp.to_dict()


@app.post("/v1/experiments/{experiment_id}/assign", tags=["Experiments"])
async def assign_variant(
    experiment_id: str,
    req: Request,
    user=Depends(get_current_user),
):
    body = await req.json()
    if not ab_test_manager:
        raise HTTPException(status_code=503, detail="Not configured")
    variant = ab_test_manager.assign_variant(experiment_id, body['user_id'])
    if not variant:
        raise HTTPException(status_code=404, detail="No variant available")
    return variant.to_dict()


@app.get("/v1/experiments/{experiment_id}/significance", tags=["Experiments"])
async def get_significance(experiment_id: str, user=Depends(require_role('admin'))):
    if not ab_test_manager:
        raise HTTPException(status_code=503, detail="Not configured")
    result = ab_test_manager.calculate_significance(experiment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Insufficient data")
    return result


# --- Prompt Versioning Endpoints ---

@app.post("/v1/prompts", tags=["Prompts"])
async def create_prompt(
    req: Request,
    user=Depends(get_current_user),
):
    body = await req.json()
    if not prompt_version_manager:
        raise HTTPException(status_code=503, detail="Not configured")
    pt = prompt_version_manager.create_prompt(
        name=body['name'],
        content=body['content'],
        description=body.get('description', ''),
        variables=body.get('variables'),
        system_prompt=body.get('system_prompt'),
        model=body.get('model'),
        organization_id=user.get('org_id'),
        created_by=user.get('sub') or user.get('id'),
    )
    return pt.to_dict()


@app.get("/v1/prompts", tags=["Prompts"])
async def list_prompts(user=Depends(get_current_user)):
    if not prompt_version_manager:
        return []
    return prompt_version_manager.list_prompts(organization_id=user.get('org_id'))


@app.post("/v1/prompts/{prompt_id}/versions", tags=["Prompts"])
async def create_prompt_version(
    prompt_id: str,
    req: Request,
    user=Depends(get_current_user),
):
    body = await req.json()
    if not prompt_version_manager:
        raise HTTPException(status_code=503, detail="Not configured")
    pv = prompt_version_manager.create_version(
        prompt_id, body['content'],
        changelog=body.get('changelog'),
        variables=body.get('variables'),
        created_by=user.get('sub') or user.get('id'),
    )
    if not pv:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return pv.to_dict()


@app.post("/v1/prompts/{prompt_id}/rollback", tags=["Prompts"])
async def rollback_prompt(
    prompt_id: str,
    req: Request,
    user=Depends(require_role('admin')),
):
    body = await req.json()
    if not prompt_version_manager:
        raise HTTPException(status_code=503, detail="Not configured")
    pv = prompt_version_manager.rollback(prompt_id, body['version'])
    if not pv:
        raise HTTPException(status_code=404, detail="Version not found")
    return pv.to_dict()


@app.post("/v1/prompts/{prompt_id}/render", tags=["Prompts"])
async def render_prompt(
    prompt_id: str,
    req: Request,
    user=Depends(get_current_user),
):
    body = await req.json()
    if not prompt_version_manager:
        raise HTTPException(status_code=503, detail="Not configured")
    content = prompt_version_manager.render(prompt_id, body.get('variables'))
    if content is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"content": content}


# --- SLA Endpoints ---

@app.post("/v1/slas", tags=["SLA"])
async def create_sla(
    req: Request,
    user=Depends(require_role('admin')),
):
    body = await req.json()
    if not sla_monitor:
        raise HTTPException(status_code=503, detail="Not configured")
    from modules.observability.sla_monitor import AlertSeverity
    sla = sla_monitor.create_sla(
        name=body['name'],
        description=body.get('description', ''),
        metric=body['metric'],
        target_value=body['target_value'],
        operator=body.get('operator', '>='),
        severity=AlertSeverity(body.get('severity', 'high')),
        organization_id=user.get('org_id'),
    )
    return sla.to_dict()


@app.get("/v1/slas", tags=["SLA"])
async def list_slas(user=Depends(get_current_user)):
    if not sla_monitor:
        return []
    return sla_monitor.list_slas(organization_id=user.get('org_id'))


@app.get("/v1/slas/dashboard", tags=["SLA"])
async def sla_dashboard(user=Depends(get_current_user)):
    if not sla_monitor:
        return {}
    return sla_monitor.get_dashboard()


@app.get("/v1/slas/alerts", tags=["SLA"])
async def sla_alerts(
    unresolved_only: bool = False,
    user=Depends(get_current_user),
):
    if not sla_monitor:
        return []
    return sla_monitor.get_alerts(unresolved_only=unresolved_only)


@app.get("/v1/admin/rate-limiter/stats", tags=["Admin"])
async def rate_limiter_stats(user=Depends(require_role('admin'))):
    if not rate_limiter:
        return {}
    return rate_limiter.get_stats()


@app.get("/v1/admin/request-logger/stats", tags=["Admin"])
async def request_logger_stats(user=Depends(require_role('admin'))):
    if not request_logger:
        return {}
    return request_logger.get_stats()


# --- WebSocket Real-time Optimization ---

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)

    async def send_json(self, client_id: str, data: dict):
        ws = self.active_connections.get(client_id)
        if ws:
            await ws.send_json(data)

ws_manager = ConnectionManager()


@app.websocket("/ws/optimize/{client_id}")
async def websocket_optimize(websocket: WebSocket, client_id: str):
    await ws_manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get('prompt', '')
            context = data.get('context', {})
            model_adapter = data.get('model_adapter')

            if not optimizer_instance:
                await ws_manager.send_json(client_id, {
                    'type': 'error', 'error': 'Optimizer not initialized'
                })
                continue

            try:
                start = time.time()
                import asyncio
                result = await asyncio.to_thread(
                    optimizer_instance.optimize, prompt, context, model_adapter
                )
                latency_ms = (time.time() - start) * 1000
                await ws_manager.send_json(client_id, {
                    'type': 'result',
                    'optimized_prompt': getattr(result, 'optimized_prompt', result.original_prompt),
                    'compression_ratio': getattr(result, 'compression_ratio', 1.0),
                    'tokens_saved': getattr(result, 'tokens_saved', 0),
                    'latency_ms': latency_ms,
                })
            except Exception as e:
                await ws_manager.send_json(client_id, {
                    'type': 'error', 'error': str(e)
                })
    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)


# --- Error Handlers ---

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=str(exc.detail),
            code=f"HTTP_{exc.status_code}",
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            code="INTERNAL_ERROR",
        ).model_dump(),
    )
