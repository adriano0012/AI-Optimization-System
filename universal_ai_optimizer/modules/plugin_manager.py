"""
Plugin System
Enables dynamic loading and management of plugins at runtime.
"""

import importlib
import logging
import os
import threading
from typing import Dict, Any, Optional, List, Callable, Type
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)


class PluginInterface:
    """Base interface that all plugins must implement."""

    def get_name(self) -> str:
        raise NotImplementedError

    def get_version(self) -> str:
        return "0.1.0"

    def initialize(self, config: Dict[str, Any]) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data


class PluginMeta:
    """Metadata for a registered plugin."""

    def __init__(self, name: str, version: str, plugin_class: Type,
                 config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.version = version
        self.plugin_class = plugin_class
        self.config = config or {}
        self.instance = None
        self.enabled = True

    def instantiate(self):
        self.instance = self.plugin_class()
        self.instance.initialize(self.config)
        return self.instance


class PluginManager(BaseOptimizerModule):
    """
    Manages plugin lifecycle: discovery, loading, initialization, and execution.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self._plugins: Dict[str, PluginMeta] = {}
        self._plugin_dirs: List[str] = self.config.get('plugin_dirs', [])
        self._lock = threading.Lock()

        if self.enabled and self._plugin_dirs:
            self._discover_plugins()

        self.logger.info(f"Plugin manager initialized with {len(self._plugins)} plugins")

    def _discover_plugins(self):
        for plugin_dir in self._plugin_dirs:
            if not os.path.isdir(plugin_dir):
                continue
            for filename in os.listdir(plugin_dir):
                if filename.endswith('.py') and not filename.startswith('_'):
                    module_name = filename[:-3]
                    try:
                        spec = importlib.util.spec_from_file_location(
                            module_name,
                            os.path.join(plugin_dir, filename)
                        )
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            if hasattr(module, 'register_plugin'):
                                module.register_plugin(self)
                    except Exception as e:
                        self.logger.warning(f"Failed to load plugin {module_name}: {e}")

    def register_plugin(self, name: str, plugin_class: Type,
                        config: Optional[Dict[str, Any]] = None,
                        version: str = "0.1.0"):
        with self._lock:
            self._plugins[name] = PluginMeta(name, version, plugin_class, config)
            self.logger.info(f"Registered plugin: {name} v{version}")

    def unregister_plugin(self, name: str):
        with self._lock:
            if name in self._plugins:
                meta = self._plugins[name]
                if meta.instance:
                    try:
                        meta.instance.shutdown()
                    except Exception as e:
                        self.logger.warning(f"Error shutting down plugin {name}: {e}")
                del self._plugins[name]
                self.logger.info(f"Unregistered plugin: {name}")

    def enable_plugin(self, name: str):
        with self._lock:
            if name in self._plugins:
                self._plugins[name].enabled = True

    def disable_plugin(self, name: str):
        with self._lock:
            if name in self._plugins:
                self._plugins[name].enabled = False

    def get_plugin(self, name: str) -> Optional[Any]:
        with self._lock:
            meta = self._plugins.get(name)
            if meta and meta.enabled and meta.instance:
                return meta.instance
            return None

    def get_all_plugins(self) -> Dict[str, bool]:
        with self._lock:
            return {name: meta.enabled for name, meta in self._plugins.items()}

    def initialize_all(self):
        with self._lock:
            for name, meta in self._plugins.items():
                if meta.enabled and not meta.instance:
                    try:
                        meta.instantiate()
                        self.logger.info(f"Initialized plugin: {name}")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize plugin {name}: {e}")

    def shutdown_all(self):
        with self._lock:
            for name, meta in self._plugins.items():
                if meta.instance:
                    try:
                        meta.instance.shutdown()
                    except Exception as e:
                        self.logger.warning(f"Error shutting down plugin {name}: {e}")
                    meta.instance = None

    def process(self, prompt: str, context: Dict[str, Any],
                model_adapter=None, pipeline_state=None) -> Dict[str, Any]:
        if not self.enabled:
            return {}

        results = {}
        with self._lock:
            active_plugins = {n: m for n, m in self._plugins.items() if m.enabled and m.instance}

        for name, meta in active_plugins.items():
            try:
                result = meta.instance.process({
                    'prompt': prompt,
                    'context': context,
                    'pipeline_state': pipeline_state,
                })
                results[name] = result
            except Exception as e:
                self.logger.error(f"Plugin {name} processing error: {e}")

        return {'plugin_results': results} if results else {}

    def get_metrics(self) -> Dict[str, Any]:
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'total_plugins': len(self._plugins),
            'active_plugins': sum(1 for m in self._plugins.values() if m.enabled),
            'plugins': self.get_all_plugins(),
        })
        return base_metrics
