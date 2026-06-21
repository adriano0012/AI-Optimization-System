"""Shared text utility functions"""
import re
import functools
from typing import Tuple, List, Dict, Optional


class TextSimilarityCache:
    """LRU-cached word set extraction for O(1) Jaccard similarity"""

    def __init__(self, maxsize: int = 2048):
        self._word_sets = functools.lru_cache(maxsize=maxsize)(self._compute_word_set)

    @staticmethod
    def _compute_word_set(text: str) -> frozenset:
        return frozenset(re.findall(r'\b\w+\b', text.lower()))

    def similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        words1 = self._word_sets(text1)
        words2 = self._word_sets(text2)
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / max(union, 1)


_similarity_cache = TextSimilarityCache()


def calculate_text_similarity(text1: str, text2: str) -> float:
    return _similarity_cache.similarity(text1, text2)
