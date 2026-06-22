"""
Model Registry
Register, version, and promote AI models with metadata tracking.
"""

import time
import secrets
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class ModelStatus(str, Enum):
    REGISTERED = "registered"
    VALIDATING = "validating"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ModelStage(str, Enum):
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ModelVersion:
    version_id: str
    model_id: str
    version: str
    stage: ModelStage
    status: ModelStatus
    artifact_uri: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    promoted_at: Optional[float] = None
    deprecated_at: Optional[float] = None
    created_by: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'version_id': self.version_id,
            'model_id': self.model_id,
            'version': self.version,
            'stage': self.stage.value,
            'status': self.status.value,
            'artifact_uri': self.artifact_uri,
            'metrics': self.metrics,
            'parameters': self.parameters,
            'tags': self.tags,
            'created_at': self.created_at,
            'promoted_at': self.promoted_at,
            'description': self.description,
        }


@dataclass
class Model:
    model_id: str
    name: str
    description: str = ""
    owner_id: Optional[str] = None
    organization_id: Optional[str] = None
    versions: List[ModelVersion] = field(default_factory=list)
    created_at: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def latest_version(self) -> Optional[ModelVersion]:
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: (v.created_at, v.version))

    @property
    def production_version(self) -> Optional[ModelVersion]:
        prod = [v for v in self.versions
                if v.stage == ModelStage.PRODUCTION and v.status == ModelStatus.ACTIVE]
        return prod[0] if prod else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'name': self.name,
            'description': self.description,
            'owner_id': self.owner_id,
            'organization_id': self.organization_id,
            'version_count': len(self.versions),
            'latest_version': self.latest_version.version if self.latest_version else None,
            'production_version': self.production_version.version if self.production_version else None,
            'created_at': self.created_at,
            'tags': self.tags,
        }


class ModelRegistry:
    """
    Centralized model registry with versioning and lifecycle management.
    """

    def __init__(self):
        self._models: Dict[str, Model] = {}
        self._lock = threading.Lock()

    def register_model(
        self,
        name: str,
        description: str = "",
        owner_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        version: str = "1.0.0",
        artifact_uri: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Model:
        model_id = f"model-{secrets.token_hex(8)}"
        version_id = f"ver-{secrets.token_hex(8)}"
        now = time.time()

        model_version = ModelVersion(
            version_id=version_id, model_id=model_id, version=version,
            stage=ModelStage.DEV, status=ModelStatus.REGISTERED,
            artifact_uri=artifact_uri, metrics=metrics or {},
            parameters=parameters or {}, tags=tags or {},
            created_at=now, updated_at=now, created_by=owner_id,
        )

        model = Model(
            model_id=model_id, name=name, description=description,
            owner_id=owner_id, organization_id=organization_id,
            versions=[model_version], created_at=now, tags=tags or {},
        )

        with self._lock:
            self._models[model_id] = model
        return model

    def add_version(
        self,
        model_id: str,
        version: str,
        artifact_uri: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> Optional[ModelVersion]:
        model = self._models.get(model_id)
        if not model:
            return None

        version_id = f"ver-{secrets.token_hex(8)}"
        now = time.time()
        mv = ModelVersion(
            version_id=version_id, model_id=model_id, version=version,
            stage=ModelStage.DEV, status=ModelStatus.REGISTERED,
            artifact_uri=artifact_uri, metrics=metrics or {},
            parameters=parameters or {}, created_at=now, updated_at=now,
            created_by=created_by, description=description,
        )
        model.versions.append(mv)
        return mv

    def promote_version(
        self,
        model_id: str,
        version: str,
        stage: ModelStage,
    ) -> Optional[ModelVersion]:
        model = self._models.get(model_id)
        if not model:
            return None

        for mv in model.versions:
            if mv.version == version:
                mv.stage = stage
                mv.status = ModelStatus.ACTIVE
                mv.promoted_at = time.time()
                mv.updated_at = time.time()
                return mv
        return None

    def deprecate_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        model = self._models.get(model_id)
        if not model:
            return None
        for mv in model.versions:
            if mv.version == version:
                mv.status = ModelStatus.DEPRECATED
                mv.deprecated_at = time.time()
                mv.updated_at = time.time()
                return mv
        return None

    def get_model(self, model_id: str) -> Optional[Model]:
        return self._models.get(model_id)

    def get_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        model = self._models.get(model_id)
        if not model:
            return None
        for mv in model.versions:
            if mv.version == version:
                return mv
        return None

    def list_models(
        self, organization_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        models = list(self._models.values())
        if organization_id:
            models = [m for m in models if m.organization_id == organization_id]
        return [m.to_dict() for m in models]

    def compare_versions(
        self, model_id: str, version_a: str, version_b: str
    ) -> Optional[Dict[str, Any]]:
        va = self.get_version(model_id, version_a)
        vb = self.get_version(model_id, version_b)
        if not va or not vb:
            return None
        all_keys = set(va.metrics.keys()) | set(vb.metrics.keys())
        comparison = {}
        for k in all_keys:
            a_val = va.metrics.get(k)
            b_val = vb.metrics.get(k)
            comparison[k] = {
                'version_a': a_val, 'version_b': b_val,
                'delta': (b_val - a_val) if a_val is not None and b_val is not None else None,
            }
        return {
            'model_id': model_id,
            'version_a': version_a, 'version_b': version_b,
            'metrics_comparison': comparison,
        }

    def get_stats(self) -> Dict[str, Any]:
        total_models = len(self._models)
        total_versions = sum(len(m.versions) for m in self._models.values())
        active = sum(
            1 for m in self._models.values()
            for v in m.versions if v.status == ModelStatus.ACTIVE
        )
        prod = sum(
            1 for m in self._models.values()
            for v in m.versions if v.stage == ModelStage.PRODUCTION
        )
        return {
            'total_models': total_models,
            'total_versions': total_versions,
            'active_versions': active,
            'production_versions': prod,
        }
