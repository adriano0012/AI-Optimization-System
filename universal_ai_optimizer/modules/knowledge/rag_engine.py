"""RAG Engine - Hybrid retrieval with chunking, embedding similarity, and re-ranking"""
import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


_tokenize_cache: Dict[str, Tuple[str, ...]] = {}
_tokenize_cache_order: List[str] = []
_TOKENIZE_CACHE_MAXSIZE = 2048

def _tokenize(text: str) -> Tuple[str, ...]:
    key = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if key in _tokenize_cache:
        return _tokenize_cache[key]
    result = tuple(re.findall(r'\b[a-z0-9]+\b', text.lower()))
    if len(_tokenize_cache) >= _TOKENIZE_CACHE_MAXSIZE:
        old = _tokenize_cache_order.pop(0)
        _tokenize_cache.pop(old, None)
    _tokenize_cache[key] = result
    _tokenize_cache_order.append(key)
    return result


class Chunk:
    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.metadata = metadata or {}
        self.tokens = _tokenize(text)
        self.embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {'text': self.text, 'metadata': self.metadata, 'tokens_count': len(self.tokens)}


class RAGEngine:
    def __init__(self, embedding_dim: int = 384):
        self.chunks: List[Chunk] = []
        self.embedding_dim = embedding_dim
        self.document_counter = 0
        self._total_retrievals = 0
        self._cache_hits = 0

    def index_document(self, text: str, metadata: Optional[Dict[str, Any]] = None, 
                       chunk_size: int = 512, overlap: int = 64) -> int:
        chunks = self._chunk_text(text, chunk_size, overlap)
        for chunk_text in chunks:
            chunk = Chunk(chunk_text, metadata)
            if NUMPY_AVAILABLE:
                chunk.embedding = self._compute_embedding(chunk_text)
            self.chunks.append(chunk)
        count = len(chunks)
        self.document_counter += 1
        logger.debug(f"Indexed document {self.document_counter} into {count} chunks")
        return count

    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        current = []
        current_len = 0
        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > chunk_size and current:
                chunks.append(' '.join(current))
                overlap_text = []
                overlap_len = 0
                for s in reversed(current):
                    sl = len(s)
                    if overlap_len + sl > overlap:
                        break
                    overlap_text.insert(0, s)
                    overlap_len += sl
                current = overlap_text
                current_len = overlap_len
            current.append(sent)
            current_len += sent_len
        if current:
            chunks.append(' '.join(current))
        return chunks or [text[:chunk_size]]

    def _compute_embedding(self, text: str) -> List[float]:
        if not NUMPY_AVAILABLE:
            return []
        tokens = _tokenize(text)
        if not tokens:
            return [0.0] * self.embedding_dim
        vocab = set(tokens)
        arr = np.zeros(self.embedding_dim, dtype=np.float32)
        for i, token in enumerate(vocab):
            idx = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.embedding_dim
            arr[idx] += 1.0
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        return arr.tolist()

    def _keyword_score(self, query_tokens: set, chunk: Chunk) -> float:
        if not chunk.tokens:
            return 0.0
        overlap = len(query_tokens & set(chunk.tokens))
        return overlap / max(len(query_tokens), len(chunk.tokens))

    def _embedding_score(self, query_emb: List[float], chunk: Chunk) -> float:
        if not chunk.embedding or not query_emb or len(query_emb) != self.embedding_dim:
            return 0.0
        if not NUMPY_AVAILABLE:
            return 0.0
        q = np.array(query_emb, dtype=np.float32)
        c = np.array(chunk.embedding, dtype=np.float32)
        dot = float(np.dot(q, c))
        return max(0.0, min(1.0, dot))

    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []
        self._total_retrievals += 1
        query_tokens = set(_tokenize(query))
        query_emb = self._compute_embedding(query) if NUMPY_AVAILABLE else []
        scored = []
        for chunk in self.chunks:
            kw_score = self._keyword_score(query_tokens, chunk)
            emb_score = self._embedding_score(query_emb, chunk) if query_emb else 0.0
            combined = alpha * kw_score + (1.0 - alpha) * emb_score
            scored.append((combined, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        if len(top) > 1:
            top = self._rerank(query_tokens, query_emb, top)
        results = []
        for score, chunk in top:
            results.append({'text': chunk.text, 'metadata': chunk.metadata, 'score': round(score, 4)})
        return results

    def _rerank(self, query_tokens: set, query_emb: List[float],
                candidates: List[Tuple[float, Chunk]]) -> List[Tuple[float, Chunk]]:
        reranked = []
        for orig_score, chunk in candidates:
            overlap = len(query_tokens & set(chunk.tokens))
            chunk_len_factor = 1.0 / max(1, len(chunk.tokens) / 10)
            rerank_score = orig_score * 0.7 + (overlap * chunk_len_factor) * 0.3
            reranked.append((rerank_score, chunk))
        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked

    def get_stats(self) -> Dict[str, Any]:
        return {
            'chunks': len(self.chunks),
            'documents': self.document_counter,
            'total_retrievals': self._total_retrievals,
            'average_chunk_tokens': sum(len(c.tokens) for c in self.chunks) / max(1, len(self.chunks))
        }