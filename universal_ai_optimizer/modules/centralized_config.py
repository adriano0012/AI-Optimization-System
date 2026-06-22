"""
Centralized Configuration
Provides a single source of truth for all module configurations.
"""

import json
import logging
import os
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)


class ConfigurationProvider:
    """Base class for configuration providers."""

    def load(self) -> Dict[str, Any]:
        raise NotImplementedError

    def save(self, config: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def watch(self, callback: Callable[[Dict[str, Any]], None]):
        pass


class FileConfigProvider(ConfigurationProvider):
    """Loads configuration from a JSON file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._last_modified = 0

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load config from {self.file_path}: {e}")
            return {}

    def save(self, config: Dict[str, Any]) -> bool:
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except IOError as e:
            logger.error(f"Failed to save config to {self.file_path}: {e}")
            return False

    def has_changed(self) -> bool:
        if not os.path.exists(self.file_path):
            return False
        current_mtime = os.path.getmtime(self.file_path)
        if current_mtime > self._last_modified:
            self._last_modified = current_mtime
            return True
        return False


class EnvironmentConfigProvider(ConfigurationProvider):
    """Loads configuration from environment variables with prefix."""

    def __init__(self, prefix: str = "UAO_"):
        self.prefix = prefix

    def load(self) -> Dict[str, Any]:
        config = {}
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                config_key = key[len(self.prefix):].lower()
                try:
                    config[config_key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    config[config_key] = value
        return config

    def save(self, config: Dict[str, Any]) -> bool:
        for key, value in config.items():
            env_key = f"{self.prefix}{key.upper()}"
            os.environ[env_key] = json.dumps(value) if not isinstance(value, str) else value
        return True


class CentralizedConfig(BaseOptimizerModule):
    """
    Centralized configuration manager that:
    - Aggregates configs from multiple providers
    - Supports runtime overrides
    - Provides module-specific config access
    - Supports hot-reloading
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self._providers: List[ConfigurationProvider] = []
        self._config_cache: Dict[str, Any] = {}
        self._overrides: Dict[str, Any] = {}
        self._listeners: List[Callable] = []
        self._lock = threading.RLock()
        self._last_reload = 0

        self._reload_interval = self.config.get('reload_interval', 300)

        self.logger.info("Centralized configuration initialized")

    def add_provider(self, provider: ConfigurationProvider, priority: int = 0):
        with self._lock:
            self._providers.append((priority, provider))
            self._providers.sort(key=lambda x: x[0], reverse=True)
            self._reload_config()

    def set_override(self, key: str, value: Any):
        with self._lock:
            self._overrides[key] = value
            self._notify_listeners()

    def remove_override(self, key: str):
        with self._lock:
            if key in self._overrides:
                del self._overrides[key]
                self._notify_listeners()

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._overrides:
            return self._overrides[key]
        return self._config_cache.get(key, default)

    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        full_config = self.get(module_name, {})
        if not isinstance(full_config, dict):
            full_config = {'value': full_config}
        return full_config

    def set_module_config(self, module_name: str, config: Dict[str, Any]):
        self.set_override(module_name, config)

    def _reload_config(self):
        new_config = {}
        for _, provider in self._providers:
            try:
                provider_config = provider.load()
                new_config.update(provider_config)
            except Exception as e:
                self.logger.warning(f"Provider load error: {e}")

        with self._lock:
            self._config_cache = new_config
            self._last_reload = time.time()
            self._notify_listeners()

    def _notify_listeners(self):
        for listener in self._listeners:
            try:
                listener(self._config_cache)
            except Exception as e:
                self.logger.warning(f"Config listener error: {e}")

    def add_listener(self, callback: Callable[[Dict[str, Any]], None]):
        self._listeners.append(callback)

    def reload_if_stale(self):
        if time.time() - self._last_reload > self._reload_interval:
            self._reload_config()

    def get_all(self) -> Dict[str, Any]:
        with self._lock:
            result = self._config_cache.copy()
            result.update(self._overrides)
            return result

    def process(self, prompt: str, context: Dict[str, Any],
                model_adapter=None, pipeline_state=None) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        self.reload_if_stale()
        return {'config_loaded': True, 'config_keys': list(self.get_all().keys())}

    def get_metrics(self) -> Dict[str, Any]:
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'providers': len(self._providers),
            'config_keys': len(self._config_cache),
            'overrides': len(self._overrides),
            'last_reload': self._last_reload,
        })
        return base_metrics
