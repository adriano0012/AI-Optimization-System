"""Adapters for different LLM providers"""

from .base_adapter import BaseModelAdapter, GenerationResult
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .gemini_adapter import GeminiAdapter
from .groq_adapter import GroqAdapter
from .ollama_adapter import OllamaAdapter
from .openrouter_adapter import OpenRouterAdapter

__all__ = [
    'BaseModelAdapter',
    'OpenAIAdapter',
    'AnthropicAdapter',
    'GeminiAdapter',
    'GroqAdapter',
    'OllamaAdapter',
    'OpenRouterAdapter',
]