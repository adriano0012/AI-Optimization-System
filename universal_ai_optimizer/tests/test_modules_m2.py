import pytest
import tempfile
import os
import json
from universal_ai_optimizer.modules.plugin_manager import PluginManager, PluginInterface, PluginMeta
from universal_ai_optimizer.modules.feature_flags import FeatureFlagManager, FeatureFlag
from universal_ai_optimizer.modules.centralized_config import (
    CentralizedConfig, FileConfigProvider, EnvironmentConfigProvider
)


class SamplePlugin(PluginInterface):
    def get_name(self):
        return "sample"

    def initialize(self, config):
        self.data = config.get('data', 'default')

    def process(self, data):
        return {'processed': True, 'input': data.get('prompt', '')}


class TestPluginManager:
    def test_init_defaults(self):
        pm = PluginManager()
        assert pm.enabled is True
        assert pm.get_all_plugins() == {}

    def test_disabled(self):
        pm = PluginManager({"enabled": False})
        result = pm.process("prompt", {})
        assert result == {}

    def test_register_plugin(self):
        pm = PluginManager()
        pm.register_plugin("test", SamplePlugin)
        assert "test" in pm.get_all_plugins()

    def test_unregister_plugin(self):
        pm = PluginManager()
        pm.register_plugin("test", SamplePlugin)
        pm.unregister_plugin("test")
        assert "test" not in pm.get_all_plugins()

    def test_enable_disable_plugin(self):
        pm = PluginManager()
        pm.register_plugin("test", SamplePlugin)
        pm.disable_plugin("test")
        assert pm.get_all_plugins()["test"] is False
        pm.enable_plugin("test")
        assert pm.get_all_plugins()["test"] is True

    def test_get_plugin(self):
        pm = PluginManager()
        pm.register_plugin("test", SamplePlugin)
        pm.initialize_all()
        plugin = pm.get_plugin("test")
        assert plugin is not None
        assert plugin.get_name() == "sample"

    def test_get_plugin_disabled(self):
        pm = PluginManager()
        pm.register_plugin("test", SamplePlugin)
        pm.initialize_all()
        pm.disable_plugin("test")
        assert pm.get_plugin("test") is None

    def test_process(self):
        pm = PluginManager()
        pm.register_plugin("test", SamplePlugin)
        pm.initialize_all()
        result = pm.process("hello", {})
        assert "plugin_results" in result

    def test_shutdown_all(self):
        pm = PluginManager()
        pm.register_plugin("test", SamplePlugin)
        pm.initialize_all()
        pm.shutdown_all()
        assert pm.get_plugin("test") is None

    def test_get_metrics(self):
        pm = PluginManager()
        pm.register_plugin("test", SamplePlugin)
        metrics = pm.get_metrics()
        assert metrics['total_plugins'] == 1
        assert metrics['active_plugins'] == 1


class TestFeatureFlagManager:
    def test_init_defaults(self):
        ffm = FeatureFlagManager()
        assert ffm.enabled is True

    def test_disabled(self):
        ffm = FeatureFlagManager({"enabled": False})
        result = ffm.process("prompt", {})
        assert result == {}

    def test_set_flag(self):
        ffm = FeatureFlagManager()
        ffm.set_flag("new_feature", True)
        assert ffm.is_enabled("new_feature") is True

    def test_disable_flag(self):
        ffm = FeatureFlagManager()
        ffm.set_flag("new_feature", True)
        ffm.set_flag("new_feature", False)
        assert ffm.is_enabled("new_feature") is False

    def test_is_enabled_default(self):
        ffm = FeatureFlagManager()
        assert ffm.is_enabled("nonexistent") is False

    def test_rollout_percentage(self):
        ffm = FeatureFlagManager()
        ffm.set_flag("gradual", True, rollout_percentage=0.0)
        assert ffm.is_enabled("gradual", user_id="user1") is False

    def test_rollout_100(self):
        ffm = FeatureFlagManager()
        ffm.set_flag("full", True, rollout_percentage=100.0)
        assert ffm.is_enabled("full", user_id="user1") is True

    def test_target_users(self):
        ffm = FeatureFlagManager()
        ffm.set_flag("targeted", True, target_users=["user1", "user2"])
        assert ffm.is_enabled("targeted", user_id="user1") is True
        assert ffm.is_enabled("targeted", user_id="user3") is False

    def test_remove_flag(self):
        ffm = FeatureFlagManager()
        ffm.set_flag("temp", True)
        ffm.remove_flag("temp")
        assert ffm.get_flag("temp") is None

    def test_get_flag(self):
        ffm = FeatureFlagManager()
        ffm.set_flag("test", True, description="Test flag")
        info = ffm.get_flag("test")
        assert info['name'] == "test"
        assert info['enabled'] is True
        assert info['description'] == "Test flag"

    def test_get_all_flags(self):
        ffm = FeatureFlagManager()
        ffm.set_flag("a", True)
        ffm.set_flag("b", False)
        flags = ffm.get_all_flags()
        assert flags == {"a": True, "b": False}

    def test_init_with_config_flags(self):
        ffm = FeatureFlagManager({"flags": {"feat1": True, "feat2": {"enabled": False}}})
        assert ffm.is_enabled("feat1") is True
        assert ffm.is_enabled("feat2") is False

    def test_get_metrics(self):
        ffm = FeatureFlagManager()
        ffm.set_flag("a", True)
        metrics = ffm.get_metrics()
        assert metrics['total_flags'] == 1
        assert metrics['enabled_flags'] == 1


class TestCentralizedConfig:
    def test_init_defaults(self):
        cc = CentralizedConfig()
        assert cc.enabled is True

    def test_disabled(self):
        cc = CentralizedConfig({"enabled": False})
        result = cc.process("prompt", {})
        assert result == {}

    def test_set_override(self):
        cc = CentralizedConfig()
        cc.set_override("key1", "value1")
        assert cc.get("key1") == "value1"

    def test_remove_override(self):
        cc = CentralizedConfig()
        cc.set_override("key1", "value1")
        cc.remove_override("key1")
        assert cc.get("key1") is None

    def test_get_default(self):
        cc = CentralizedConfig()
        assert cc.get("nonexistent", "default") == "default"

    def test_file_provider(self):
        tmp_path = os.path.join(tempfile.gettempdir(), f"test_config_{os.getpid()}.json")
        try:
            with open(tmp_path, 'w') as f:
                json.dump({"setting1": "value1"}, f)
            cc = CentralizedConfig()
            cc.add_provider(FileConfigProvider(tmp_path))
            assert cc.get("setting1") == "value1"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def test_file_provider_save(self):
        tmp_path = os.path.join(tempfile.gettempdir(), f"test_save_{os.getpid()}.json")
        try:
            provider = FileConfigProvider(tmp_path)
            provider.save({"new_key": "new_value"})
            loaded = provider.load()
            assert loaded["new_key"] == "new_value"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def test_file_provider_nonexistent(self):
        provider = FileConfigProvider("/nonexistent/path.json")
        assert provider.load() == {}

    def test_module_config(self):
        cc = CentralizedConfig()
        cc.set_module_config("cache", {"enabled": True, "ttl": 300})
        config = cc.get_module_config("cache")
        assert config['enabled'] is True
        assert config['ttl'] == 300

    def test_get_all(self):
        cc = CentralizedConfig()
        cc.set_override("a", 1)
        cc.set_override("b", 2)
        all_config = cc.get_all()
        assert all_config['a'] == 1
        assert all_config['b'] == 2

    def test_add_listener(self):
        cc = CentralizedConfig()
        changes = []
        cc.add_listener(lambda c: changes.append(c))
        cc.set_override("x", 1)
        assert len(changes) > 0

    def test_get_metrics(self):
        cc = CentralizedConfig()
        metrics = cc.get_metrics()
        assert 'enabled' in metrics
        assert 'providers' in metrics
        assert 'config_keys' in metrics
