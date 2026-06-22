"""
Prompt Versioning System
Version control for prompts with diff tracking and rollback.
"""

import time
import secrets
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class PromptStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class PromptVersion:
    version_id: str
    prompt_id: str
    version: int
    content: str
    status: PromptStatus
    variables: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    tags: Dict[str, str] = field(default_factory=dict)
    hash: str = ""
    created_at: float = 0.0
    created_by: Optional[str] = None
    changelog: Optional[str] = None
    usage_count: int = 0
    avg_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'version_id': self.version_id,
            'prompt_id': self.prompt_id,
            'version': self.version,
            'content': self.content,
            'status': self.status.value,
            'variables': self.variables,
            'system_prompt': self.system_prompt,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'tags': self.tags,
            'hash': self.hash,
            'created_at': self.created_at,
            'changelog': self.changelog,
            'usage_count': self.usage_count,
            'avg_score': self.avg_score,
        }


@dataclass
class PromptTemplate:
    prompt_id: str
    name: str
    description: str = ""
    organization_id: Optional[str] = None
    owner_id: Optional[str] = None
    versions: List[PromptVersion] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def active_version(self) -> Optional[PromptVersion]:
        active = [v for v in self.versions if v.status == PromptStatus.ACTIVE]
        return active[0] if active else None

    @property
    def latest_version(self) -> Optional[PromptVersion]:
        return max(self.versions, key=lambda v: v.version) if self.versions else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'prompt_id': self.prompt_id,
            'name': self.name,
            'description': self.description,
            'organization_id': self.organization_id,
            'version_count': len(self.versions),
            'active_version': self.active_version.version if self.active_version else None,
            'latest_version': self.latest_version.version if self.latest_version else None,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }


class PromptVersionManager:
    """
    Prompt version control system with diff tracking and rollback.
    """

    def __init__(self):
        self._prompts: Dict[str, PromptTemplate] = {}
        self._lock = threading.Lock()

    def create_prompt(
        self,
        name: str,
        content: str,
        description: str = "",
        variables: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        organization_id: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> PromptTemplate:
        prompt_id = f"prompt-{secrets.token_hex(8)}"
        version_id = f"ver-{secrets.token_hex(8)}"
        now = time.time()
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        pv = PromptVersion(
            version_id=version_id, prompt_id=prompt_id,
            version=1, content=content, status=PromptStatus.ACTIVE,
            variables=variables or [], system_prompt=system_prompt,
            model=model, temperature=temperature, max_tokens=max_tokens,
            hash=content_hash, created_at=now, created_by=created_by,
        )

        pt = PromptTemplate(
            prompt_id=prompt_id, name=name, description=description,
            organization_id=organization_id, versions=[pv],
            created_at=now, updated_at=now,
        )
        with self._lock:
            self._prompts[prompt_id] = pt
        return pt

    def create_version(
        self,
        prompt_id: str,
        content: str,
        changelog: Optional[str] = None,
        variables: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        created_by: Optional[str] = None,
    ) -> Optional[PromptVersion]:
        pt = self._prompts.get(prompt_id)
        if not pt:
            return None

        latest = pt.latest_version
        next_version = (latest.version + 1) if latest else 1
        version_id = f"ver-{secrets.token_hex(8)}"
        now = time.time()
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        pv = PromptVersion(
            version_id=version_id, prompt_id=prompt_id,
            version=next_version, content=content,
            status=PromptStatus.ACTIVE,
            variables=variables or (latest.variables if latest else []),
            system_prompt=system_prompt or (latest.system_prompt if latest else None),
            model=model or (latest.model if latest else None),
            temperature=temperature,
            max_tokens=max_tokens,
            hash=content_hash, created_at=now, created_by=created_by,
            changelog=changelog,
        )

        if latest:
            latest.status = PromptStatus.DEPRECATED

        pt.versions.append(pv)
        pt.updated_at = now
        return pv

    def rollback(self, prompt_id: str, target_version: int) -> Optional[PromptVersion]:
        pt = self._prompts.get(prompt_id)
        if not pt:
            return None

        target = None
        for v in pt.versions:
            if v.version == target_version:
                target = v
                break
        if not target:
            return None

        for v in pt.versions:
            if v.status == PromptStatus.ACTIVE:
                v.status = PromptStatus.DEPRECATED

        target.status = PromptStatus.ACTIVE
        pt.updated_at = time.time()
        return target

    def get_prompt(self, prompt_id: str) -> Optional[PromptTemplate]:
        return self._prompts.get(prompt_id)

    def get_version(
        self, prompt_id: str, version: int
    ) -> Optional[PromptVersion]:
        pt = self._prompts.get(prompt_id)
        if not pt:
            return None
        for v in pt.versions:
            if v.version == version:
                return v
        return None

    def get_diff(
        self, prompt_id: str, version_a: int, version_b: int
    ) -> Optional[Dict[str, Any]]:
        va = self.get_version(prompt_id, version_a)
        vb = self.get_version(prompt_id, version_b)
        if not va or not vb:
            return None

        a_lines = va.content.splitlines()
        b_lines = vb.content.splitlines()
        max_lines = max(len(a_lines), len(b_lines))

        changes = []
        for i in range(max_lines):
            a_line = a_lines[i] if i < len(a_lines) else None
            b_line = b_lines[i] if i < len(b_lines) else None
            if a_line != b_line:
                changes.append({
                    'line': i + 1,
                    'old': a_line,
                    'new': b_line,
                })

        return {
            'prompt_id': prompt_id,
            'version_a': version_a,
            'version_b': version_b,
            'changes': changes,
            'total_changes': len(changes),
        }

    def render(
        self, prompt_id: str, variables: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        pt = self._prompts.get(prompt_id)
        if not pt or not pt.active_version:
            return None

        content = pt.active_version.content
        if variables:
            for key, value in variables.items():
                content = content.replace(f"{{{{{key}}}}}", value)
        return content

    def record_usage(self, prompt_id: str, score: Optional[float] = None):
        pt = self._prompts.get(prompt_id)
        if pt and pt.active_version:
            pt.active_version.usage_count += 1
            if score is not None:
                if pt.active_version.avg_score is None:
                    pt.active_version.avg_score = score
                else:
                    n = pt.active_version.usage_count
                    old_avg = pt.active_version.avg_score
                    pt.active_version.avg_score = old_avg + (score - old_avg) / n

    def list_prompts(
        self, organization_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        prompts = list(self._prompts.values())
        if organization_id:
            prompts = [p for p in prompts if p.organization_id == organization_id]
        return [p.to_dict() for p in prompts]

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._prompts)
        total_versions = sum(len(p.versions) for p in self._prompts.values())
        return {
            'total_prompts': total,
            'total_versions': total_versions,
            'total_usage': sum(
                v.usage_count
                for p in self._prompts.values()
                for v in p.versions
            ),
        }
