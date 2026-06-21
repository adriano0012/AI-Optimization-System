"""
Semantic Router Module
Responsible for semantic-based routing of tasks to appropriate model categories.
"""

import re
import math
import hashlib
import logging
from typing import Dict, Any, Optional, List, Union
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)


class SemanticRouter(BaseOptimizerModule):
    """
    Routes tasks semantically by computing similarity with predefined categories.
    Uses sentence-transformers for real embeddings when available, falls back to
    TF-IDF-style hashing when not installed.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get("enabled", True)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
        self.categories = self.config.get("categories", {})
        self._model = None
        self._embedding_dim = 384
        self._init_model()

    def _init_model(self):
        """Try to load sentence-transformers model, fall back to hash-based."""
        try:
            from sentence_transformers import SentenceTransformer
            model_name = self.config.get("model_name", "all-MiniLM-L6-v2")
            self._model = SentenceTransformer(model_name)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded sentence-transformers model: {model_name} (dim={self._embedding_dim})")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Using hash-based embeddings. "
                "Install with: pip install sentence-transformers"
            )
            self._model = None
        except Exception as e:
            logger.warning(f"Failed to load sentence-transformers: {e}. Using hash fallback.")
            self._model = None

    def add_category(self, name: str, description: str, preferred_models: Optional[List[str]] = None) -> None:
        """Adds a semantic routing category"""
        self.categories[name] = {
            "description": description,
            "preferred_models": preferred_models
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for the semantic router"""
        return {
            "enabled": self.enabled,
            "similarity_threshold": self.similarity_threshold,
            "model_loaded": self._model is not None,
            "embedding_dim": self._embedding_dim,
            "categories_count": len(self.categories)
        }

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Computes the cosine similarity between two vectors"""
        dot_product = sum(x * y for x, y in zip(v1, v2))
        norm1 = math.sqrt(sum(x * x for x in v1))
        norm2 = math.sqrt(sum(x * x for x in v2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _get_embedding(self, text: str) -> List[float]:
        """
        Generates a semantic embedding for the given text.
        Uses sentence-transformers if available, otherwise falls back to hash-based.
        """
        if not text:
            return [0.0] * self._embedding_dim

        # Use real sentence-transformers model if available
        if self._model is not None:
            try:
                embedding = self._model.encode(text, convert_to_numpy=False)
                return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            except Exception as e:
                logger.warning(f"Embedding computation failed, falling back to hash: {e}")

        # Fallback: TF-IDF-inspired hash-based embedding
        return self._hash_embedding(text)

    def _hash_embedding(self, text: str) -> List[float]:
        """Deterministic hash-based embedding fallback."""
        words = re.findall(r"\w+", text.lower())
        if not words:
            words = [text.lower()]

        vector = [0.0] * self._embedding_dim
        for word in words:
            h = hashlib.sha256(word.encode("utf-8")).digest()
            for i in range(self._embedding_dim):
                byte_idx = i % len(h)
                val = (h[byte_idx] / 127.5) - 1.0
                vector[i] += val

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return [0.0] * self._embedding_dim
        return [v / norm for v in vector]

    def process(
        self,
        prompt: str,
        context: Dict[str, Any],
        model_adapter: Optional[Any] = None,
        pipeline_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Determines the semantic category for the prompt and routes accordingly"""
        if not self.enabled:
            return {}

        self._log_processing(len(prompt), len(str(context)))

        if not self.categories:
            return {"semantic_route": None, "semantic_similarity": 0.0}

        prompt_emb = self._get_embedding(prompt)
        best_category = None
        best_similarity = -1.0

        for cat_name, cat_info in self.categories.items():
            desc = cat_info.get("description", "")
            desc_emb = self._get_embedding(desc)
            sim = self._cosine_similarity(prompt_emb, desc_emb)
            if sim > best_similarity:
                best_similarity = sim
                best_category = (cat_name, cat_info)

        if best_category and best_similarity >= self.similarity_threshold:
            cat_name, cat_info = best_category
            return {
                "semantic_route": cat_name,
                "semantic_similarity": best_similarity,
                "preferred_models": cat_info.get("preferred_models")
            }

        return {"semantic_route": None, "semantic_similarity": max(0.0, best_similarity)}
