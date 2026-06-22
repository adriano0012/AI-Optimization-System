"""
Pydantic Schemas
Request/response validation for the API layer.
"""

from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import Optional, List, Dict, Any
from enum import Enum


class UserRole(str, Enum):
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class PlanTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


# --- Auth Schemas ---

class UserCreate(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    name: str = Field(..., min_length=1, max_length=255)

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if '@' not in v or '.' not in v:
            raise ValueError('Invalid email format')
        return v.lower().strip()


class UserLogin(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    user_id: str
    organization_id: Optional[str] = None


class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    rate_limit: int = Field(default=1000, ge=1, le=100000)


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    key: Optional[str] = None
    key_prefix: str
    is_active: bool
    rate_limit: int
    created_at: str


# --- Organization Schemas ---

class OrganizationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    plan: PlanTier = PlanTier.FREE


class OrganizationResponse(BaseModel):
    id: str
    name: str
    plan: str
    created_at: str


# --- Optimization Schemas ---

class OptimizeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=100000)
    context: Dict[str, Any] = Field(default_factory=dict)
    model_adapter: Optional[str] = None
    task_type: Optional[str] = None
    stream: bool = False

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v.strip()


class OptimizeResponse(BaseModel):
    id: str
    original_prompt: str
    optimized_prompt: str
    compression_ratio: float
    tokens_saved: int
    latency_ms: float
    task_type: Optional[str] = None
    model_used: Optional[str] = None
    success: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OptimizeStreamChunk(BaseModel):
    chunk_type: str
    data: Any
    timestamp: float


# --- Health/Safety Schemas ---

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    checks: Dict[str, Any]


class MetricsResponse(BaseModel):
    total_optimizations: int
    total_tokens_saved: int
    avg_compression_ratio: float
    avg_latency_ms: float
    error_rate: float
    active_users: int


# --- Error Schemas ---

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: str
    request_id: Optional[str] = None


class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int


# --- Admin Schemas ---

class AuditLogResponse(BaseModel):
    id: str
    user_id: Optional[str]
    action: str
    resource_type: Optional[str]
    resource_id: Optional[str]
    details: Optional[Dict[str, Any]]
    ip_address: Optional[str]
    created_at: str


class QuotaResponse(BaseModel):
    quota_type: str
    limit_value: int
    used_value: int
    period: str
    remaining: int
