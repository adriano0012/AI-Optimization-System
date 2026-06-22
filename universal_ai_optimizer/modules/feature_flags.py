"""
Feature Flags
Runtime feature toggles with percentage rollouts and user targeting.
"""

import hashlib
import logging
import threading
import time
from typing import Dict, Any, Optional, List
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)


class FeatureFlag:
    """Represents a single feature flag."""

    def __init__(self, name: str, enabled: bool = True,
                 description: str = "",
                 rollout_percentage: float = 100.0,
                 target_users: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.enabled = enabled
        self.description = description
        self.rollout_percentage = rollout_percentage
        self.target_users = target_users or []
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.updated_at = time.time()


class FeatureFlagManager(BaseOptimizerModule):
    """
    Manages feature flags with support for:
    - Global enable/disable
    - Percentage-based rollouts
    - User targeting
    - Environment-based flags
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self._flags: Dict[str, FeatureFlag] = {}
        self._lock = threading.Lock()

        self._environment = self.config.get('environment', 'production')
        self._default_flags = self.config.get('flags', {})

        for name, flag_config in self._default_flags.items():
            if isinstance(flag_config, dict):
                self._flags[name] = FeatureFlag(name, **flag_config)
            else:
                self._flags[name] = FeatureFlag(name, enabled=bool(flag_config))

        self.logger.info(f"Feature flag manager initialized with {len(self._flags)} flags")

    def is_enabled(self, flag_name: str, user_id: Optional[str] = None,
                   context: Optional[Dict[str, Any]] = None) -> bool:
        if not self.enabled:
            return True

        flag = self._flags.get(flag_name)
        if flag is None:
            return self._default_flags.get(flag_name, {}).get('enabled', False) if isinstance(self._default_flags.get(flag_name), dict) else bool(self._default_flags.get(flag_name, False))

        if not flag.enabled:
            return False

        if user_id and flag.target_users:
            if user_id in flag.target_users:
                return True
            return False

        if flag.rollout_percentage < 100.0:
            hash_val = int(hashlib.md5(f"{flag_name}:{user_id or 'anonymous'}".encode()).hexdigest()[:8], 16)
            user_bucket = (hash_val % 100)
            return user_bucket < flag.rollout_percentage

        return True

    def set_flag(self, name: str, enabled: bool, **kwargs):
        with self._lock:
            if name in self._flags:
                flag = self._flags[name]
                flag.enabled = enabled
                flag.updated_at = time.time()
                for k, v in kwargs.items():
                    if hasattr(flag, k):
                        setattr(flag, k, v)
            else:
                self._flags[name] = FeatureFlag(name, enabled=enabled, **kwargs)
            self.logger.info(f"Feature flag '{name}' set to {enabled}")

    def remove_flag(self, name: str):
        with self._lock:
            if name in self._flags:
                del self._flags[name]
                self.logger.info(f"Removed feature flag: {name}")

    def get_flag(self, name: str) -> Optional[Dict[str, Any]]:
        flag = self._flags.get(name)
        if flag is None:
            return None
        return {
            'name': flag.name,
            'enabled': flag.enabled,
            'description': flag.description,
            'rollout_percentage': flag.rollout_percentage,
            'target_users': flag.target_users,
            'metadata': flag.metadata,
            'created_at': flag.created_at,
            'updated_at': flag.updated_at,
        }

    def get_all_flags(self) -> Dict[str, bool]:
        return {name: flag.enabled for name, flag in self._flags.items()}

    def get_enabled_flags(self) -> List[str]:
        return [name for name, flag in self._flags.items() if flag.enabled]

    def process(self, prompt: str, context: Dict[str, Any],
                model_adapter=None, pipeline_state=None) -> Dict[str, Any]:
        if not self.enabled:
            return {}

        user_id = context.get('user_id')
        active_flags = {
            name: self.is_enabled(name, user_id, context)
            for name in self._flags
        }
        return {'feature_flags': active_flags}

    def get_metrics(self) -> Dict[str, Any]:
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'total_flags': len(self._flags),
            'enabled_flags': len(self.get_enabled_flags()),
            'flags': self.get_all_flags(),
            'environment': self._environment,
        })
        return base_metrics
