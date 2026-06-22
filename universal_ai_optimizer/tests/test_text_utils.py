import pytest
from universal_ai_optimizer.modules.common.text_utils import (
    TextSimilarityCache, calculate_text_similarity
)


class TestTextSimilarityCache:
    def test_identical_texts(self):
        cache = TextSimilarityCache()
        assert cache.similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        cache = TextSimilarityCache()
        assert cache.similarity("cat", "dog") == 0.0

    def test_empty_inputs(self):
        cache = TextSimilarityCache()
        assert cache.similarity("", "") == 0.0

    def test_one_empty(self):
        cache = TextSimilarityCache()
        assert cache.similarity("hello", "") == 0.0
        assert cache.similarity("", "hello") == 0.0

    def test_case_insensitive(self):
        cache = TextSimilarityCache()
        assert cache.similarity("Hello World", "hello world") == 1.0

    def test_punctuation_stripped(self):
        cache = TextSimilarityCache()
        assert cache.similarity("hello.", "hello") == 1.0

    def test_partial_overlap(self):
        cache = TextSimilarityCache()
        score = cache.similarity("hello world", "hello there")
        assert 0.0 < score < 1.0

    def test_whitespace_only(self):
        cache = TextSimilarityCache()
        assert cache.similarity("   ", "   ") == 0.0

    def test_single_word(self):
        cache = TextSimilarityCache()
        assert cache.similarity("test", "test") == 1.0


class TestCalculateTextSimilarity:
    def test_identical(self):
        assert calculate_text_similarity("foo bar", "foo bar") == 1.0

    def test_empty(self):
        assert calculate_text_similarity("", "") == 0.0

    def test_different(self):
        assert calculate_text_similarity("aaa", "bbb") == 0.0
