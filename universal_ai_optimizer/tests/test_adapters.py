import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any


@pytest.fixture(autouse=True)
def _mock_external_packages():
    """Mock openai / anthropic modules so adapters can be imported without real SDKs."""
    import sys

    class MockResponse:
        class Choice:
            class Message:
                content = "openai mocked"
            message = Message()
        class Usage:
            total_tokens = 10
        choices = [Choice()]
        usage = Usage()

    class MockAnthropicResponse:
        class Content:
            text = "anthropic mocked"
        content = [Content()]
        usage = None

    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MockResponse()

    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MockAnthropicResponse()

    # Patch at the module level so the adapters' import checks find them
    patcher_openai = patch.dict("sys.modules", {"openai": mock_openai})
    patcher_anthropic = patch.dict("sys.modules", {"anthropic": mock_anthropic})
    patcher_openai.start()
    patcher_anthropic.start()
    yield
    patcher_openai.stop()
    patcher_anthropic.stop()


class TestOpenAIAdapter:
    def test_import_and_init(self, adapters_config):
        from universal_ai_optimizer.modules.adapters.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter(adapters_config)
        assert adapter.model == "test-model"
        assert adapter.api_key == "test-key-12345"
        assert adapter.timeout == 5

    def test_generate_returns_text(self, adapters_config):
        from universal_ai_optimizer.modules.adapters.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter(adapters_config)
        result = adapter.generate("hello")
        assert result.text == "openai mocked"

    def test_get_model_info(self, adapters_config):
        from universal_ai_optimizer.modules.adapters.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter(adapters_config)
        info = adapter.get_model_info()
        assert info["provider"] == "openai"
        assert info["model"] == "test-model"

    def test_validate_config_missing_key(self):
        from universal_ai_optimizer.modules.adapters.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter({"model": "gpt-4"})
        assert adapter.validate_config() is False

    def test_validate_config_with_key(self, adapters_config):
        from universal_ai_optimizer.modules.adapters.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter(adapters_config)
        assert adapter.validate_config() is True


class TestAnthropicAdapter:
    def test_import_and_init(self, adapters_config):
        from universal_ai_optimizer.modules.adapters.anthropic_adapter import AnthropicAdapter
        adapter = AnthropicAdapter(adapters_config)
        assert adapter.model == "test-model"
        assert adapter.api_key == "test-key-12345"

    def test_generate_returns_text(self, adapters_config):
        from universal_ai_optimizer.modules.adapters.anthropic_adapter import AnthropicAdapter
        adapter = AnthropicAdapter(adapters_config)
        result = adapter.generate("hello")
        assert result.text == "anthropic mocked"

    def test_generate_raises_without_key(self):
        from universal_ai_optimizer.modules.adapters.anthropic_adapter import AnthropicAdapter
        adapter = AnthropicAdapter({"model": "claude-2"})
        with pytest.raises(ValueError, match="API key not provided"):
            adapter.generate("hello")

    def test_get_model_info(self, adapters_config):
        from universal_ai_optimizer.modules.adapters.anthropic_adapter import AnthropicAdapter
        adapter = AnthropicAdapter(adapters_config)
        info = adapter.get_model_info()
        assert info["provider"] == "anthropic"

    def test_validate_config(self, adapters_config):
        from universal_ai_optimizer.modules.adapters.anthropic_adapter import AnthropicAdapter
        adapter = AnthropicAdapter(adapters_config)
        assert adapter.validate_config() is True


class TestOpenRouterAdapter:
    def test_import_and_init(self, adapters_config):
        from universal_ai_optimizer.modules.adapters.openrouter_adapter import OpenRouterAdapter
        adapter = OpenRouterAdapter(adapters_config)
        assert adapter.model == "test-model"

    def test_validate_config_missing_key(self):
        from universal_ai_optimizer.modules.adapters.openrouter_adapter import OpenRouterAdapter
        adapter = OpenRouterAdapter({"model": "test"})
        assert adapter.validate_config() is False


class TestOllamaAdapter:
    def test_import_and_init(self):
        from universal_ai_optimizer.modules.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter({"model": "llama2", "base_url": "http://localhost:11434"})
        assert adapter.model == "llama2"
        assert adapter.base_url == "http://localhost:11434"

    def test_get_model_info_defaults(self):
        from universal_ai_optimizer.modules.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter({})
        assert adapter.model == "llama2"
        assert adapter.base_url == "http://localhost:11434"


class TestGroqAdapter:
    def test_import_and_init(self, adapters_config):
        from universal_ai_optimizer.modules.adapters.groq_adapter import GroqAdapter
        adapter = GroqAdapter(adapters_config)
        assert adapter.model == "test-model"
        assert adapter.base_url == "http://test.local/api"
