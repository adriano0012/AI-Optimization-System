import pytest
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# Shared configs
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config():
    return {"enabled": True}

@pytest.fixture
def adapters_config():
    return {
        "api_key": "test-key-12345",
        "model": "test-model",
        "base_url": "http://test.local/api",
        "timeout": 5,
    }

@pytest.fixture
def routing_config():
    return {
        "enabled": True,
        "routing_type": "model",
        "exploration_rate": 0.1,
        "learning_rate": 0.01,
        "available_options": ["gpt-4", "claude-3", "gemini-pro"],
    }

@pytest.fixture
def simple_context():
    return {"task_type": "code_generation", "difficulty": "medium", "prompt": "def hello(): pass"}

@pytest.fixture
def mock_adapter():
    """A minimal model adapter stub that returns canned responses."""
    class MockGenerationResult:
        def __init__(self, text):
            self.text = text
            self.token_count = len(text.split())
    class MockAdapter:
        def generate(self, prompt="", **kwargs):
            return MockGenerationResult("mocked response from adapter")
        def get_model_info(self):
            return {"provider": "mock", "model": "mock-model", "type": "chat", "supports_streaming": False}
        def validate_config(self):
            return True
    return MockAdapter()
