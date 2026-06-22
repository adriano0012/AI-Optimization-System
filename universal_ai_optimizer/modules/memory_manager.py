"""
Memory Manager Module
Implements advanced memory systems for context retention and retrieval
"""

from typing import Dict, Any, Optional, List, Tuple, Callable
import logging
import math
import heapq
import time
import re
import threading
try:
    import numpy as np
except ImportError:
    np = None
from collections import deque, defaultdict
from universal_ai_optimizer.core.base import BaseOptimizerModule
from universal_ai_optimizer.modules.common.text_utils import TextSimilarityCache

logger = logging.getLogger(__name__)

_COMPRESSION_OPTIONAL_FIELDS = {'metadata', 'tags', 'embedding', 'raw_text', 'source'}
_COMPRESSION_MAX_TEXT_LENGTH = 500


def rank_memories(memories, context):
    return sorted(memories, key=lambda x: x.get('importance_score', 0), reverse=True)


def compress_memory(memory_data):
    if not isinstance(memory_data, dict):
        return memory_data
    compressed = {}
    for key, value in memory_data.items():
        if key in _COMPRESSION_OPTIONAL_FIELDS:
            continue
        if key in ('content', 'text', 'summary') and isinstance(value, str):
            if len(value) > _COMPRESSION_MAX_TEXT_LENGTH:
                value = value[:_COMPRESSION_MAX_TEXT_LENGTH] + '...'
        if key.endswith('_score') and isinstance(value, (int, float)):
            value = round(value, 2)
        compressed[key] = value
    return compressed


def compress_memory_batch(memories):
    return [compress_memory(m) for m in memories]


def estimate_memory_size(memories):
    import json
    return len(json.dumps(memories, default=str).encode('utf-8'))


def apply_memory_expiration(memories, current_time, decay_rate):
    for mem in memories:
        age = current_time - mem.get('timestamp', current_time)
        mem['importance_score'] = mem.get('importance_score', 1.0) * (decay_rate ** (age / 86400))
    return [m for m in memories if m.get('importance_score', 0) > 0.1]

class WorkingMemory:
    def __init__(self, capacity=100):
        self.memories = deque(maxlen=capacity)
    def add(self, memory): self.memories.append(memory)
    def get_all(self): return list(self.memories)

class EpisodicMemory:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.memories = deque(maxlen=capacity)
    def add(self, memory):
        self.memories.append(memory)
    def get_all(self): return list(self.memories)

class SemanticMemory:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memories = []
    def add(self, memory):
        self.memories.append(memory)
        if len(self.memories) > self.capacity:
            self.memories.pop(0)
    def get_all(self): return self.memories

class LongTermMemory:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.memories = []
    def add(self, memory):
        self.memories.append(memory)
        if len(self.memories) > self.capacity:
            self.memories.pop(0)
    def get_all(self): return self.memories

class MemoryManager(BaseOptimizerModule):
    """
    Advanced memory manager that handles different types of memory systems
    with ranking, scoring, and intelligent retrieval
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.memory_types = self.config.get('memory_types', [
            'working', 'episodic', 'semantic', 'long_term', 'hierarchical'
        ])
        self.vector_db = self.config.get('vector_db', 'faiss')
        self.embedding_model = self.config.get('embedding_model', 
                                               'sentence-transformers/all-MiniLM-L6-v2')
        self.max_memories = self.config.get('max_memories', 10000)
        self.memory_threshold = self.config.get('memory_threshold', 0.7)
        self.consolidation_threshold = self.config.get('consolidation_threshold', 0.8)
        self.forgetting_curve_rate = self.config.get('forgetting_curve_rate', 0.1)
        
        # For hierarchical memory - define this BEFORE calling _init_memory_systems
        self.memory_hierarchy = {
            'working': {'capacity': 100, 'decay_rate': 0.9},
            'episodic': {'capacity': 1000, 'decay_rate': 0.95},
            'semantic': {'capacity': 10000, 'decay_rate': 0.99},
            'long_term': {'capacity': 50000, 'decay_rate': 0.999}
        }
        
        # Pre-computed embedding cache to avoid O(n²) recomputation
        self._embedding_cache = TextSimilarityCache(maxsize=4096)
        
        # Initialize memory systems with enhanced structures
        self._init_memory_systems()
        
        # Memory ranking and scoring components
        self.memory_scores = defaultdict(float)  # memory_id -> score
        self.access_counts = defaultdict(int)    # memory_id -> access count
        self.last_accessed = defaultdict(float)  # memory_id -> timestamp
        self.creation_time = defaultdict(float)  # memory_id -> timestamp

    def _get_timestamp(self):
        """Get current timestamp"""
        return time.time()

    def _calculate_initial_importance(self, prompt: str, context: Dict[str, Any]) -> float:
        """
        Calculate initial importance score for a new memory
        Based on prompt length, context richness, and other heuristics
        """
        # Simple heuristic: longer prompts and richer context are more important
        prompt_score = min(len(prompt) / 1000.0, 1.0)  # Normalize to 0-1
        context_score = min(len(str(context)) / 5000.0, 1.0)  # Normalize to 0-1
        
        # Combine scores (weighted average)
        importance = 0.6 * prompt_score + 0.4 * context_score
        
        # Ensure in range [0, 1]
        return max(0.0, min(1.0, importance))
    
    def _init_memory_systems(self):
        """Initialize enhanced memory systems"""
        self.logger.debug(f"Initializing advanced memory systems: {self.memory_types}")
        
        # Different storage structures for different memory types
        self.memories = {
            'working': deque(maxlen=self.memory_hierarchy['working']['capacity']),
            'episodic': [],
            'semantic': [],  # Would be backed by vector DB in reality
            'long_term': [],  # Would be backed by persistent storage
            'hierarchical': {}  # Organized by importance/access patterns
        }
        
        # Indexing for fast retrieval
        self.memory_index = {}  # memory_id -> memory object
        self.semantic_index = {}  # For semantic search (placeholder)
        self.episodic_index = defaultdict(list)  # By timestamp/tags
        
        # Memory counters
        self.next_memory_id = 0
        self.total_memories_stored = 0
        self.total_memories_retrieved = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.hits = 0
        self.misses = 0
        
        # Thread safety
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # In a real implementation, we would set up vector databases, etc.
        self.embeddings = {}  # Cache for embeddings {memory_id: embedding}
        self.access_patterns = defaultdict(list)  # Track access patterns for prediction
        
        # Background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def process(self, prompt: str, context: Dict[str, Any], 
               model_adapter: Optional[Any] = None, 
               pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input with advanced memory systems - store, retrieve, and manage memories
        
        Args:
            prompt: Input prompt
            context: Context dictionary
            model_adapter: Model adapter (unused in this module but required by interface)
            pipeline_state: Current pipeline state
            
        Returns:
            Dictionary with updated context including retrieved memories and memory stats
        """
        if not self.enabled:
            return {}
        
        self._log_processing(len(prompt), len(str(context)))
        
        # Store current interaction in working memory with enhanced metadata
        memory_id = self._store_in_memory_enhanced('working', {
            'prompt': prompt,
            'context': context,
            'timestamp': self._get_timestamp(),
            'importance_score': self._calculate_initial_importance(prompt, context)
        })
        
        # Retrieve relevant memories using multiple strategies
        relevant_memories = self._retrieve_relevant_memories_advanced(prompt, context)
        
        # Track cache hit/miss for metrics
        if relevant_memories:
            self.cache_hits += 1
            self.hits += 1
        else:
            self.cache_misses += 1
            self.misses += 1
        
        # Update context with retrieved memories
        updated_context = context.copy()
        if relevant_memories:
            updated_context['retrieved_memories'] = relevant_memories
            updated_context['memory_ids'] = [mem.get('id') for mem in relevant_memories if mem.get('id')]
        
        # Perform memory consolidation and maintenance
        self._consolidate_memories_intelligent()
        self._update_memory_rankings()
        
        # Update pipeline state for downstream modules
        memory_stats = self._get_enhanced_memory_stats()
        
        result = {
            'compressed_context': updated_context,
            'memory_stats': memory_stats,
            'stored_memory_id': memory_id
        }
        
        self.logger.debug(f"Retrieved {len(relevant_memories)} relevant memories")
        return result
    
    def _store_in_memory_enhanced(self, memory_type: str, data: Dict[str, Any]) -> str:
        """Store data in specified memory type with enhanced metadata"""
        if memory_type not in self.memories:
            self.logger.warning(f"Unknown memory type: {memory_type}")
            return ""
        
        # Generate unique memory ID
        memory_id = f"{memory_type}_{self.next_memory_id}"
        self.next_memory_id += 1
        
        # Enhance data with metadata
        enhanced_data = {
            'id': memory_id,
            'type': memory_type,
            'timestamp': data.get('timestamp', self._get_timestamp()),
            'importance_score': data.get('importance_score', 0.5),
            'access_count': 0,
            'last_accessed': self._get_timestamp(),
            'created_at': self._get_timestamp(),
            'data': data
        }
        
        # Store in appropriate structure
        if memory_type == 'working':
            # Working memory uses deque with automatic overflow
            self.memories[memory_type].append(enhanced_data)
        else:
            # Other memory types use lists
            self.memories[memory_type].append(enhanced_data)
            
            # Enforce capacity limits
            capacity = self.memory_hierarchy.get(memory_type, {}).get('capacity', self.max_memories)
            if len(self.memories[memory_type]) > capacity:
                # Remove lowest scoring memories
                self._prune_memory_by_score(memory_type, capacity)
        
        # Update indices and tracking
        self.memory_index[memory_id] = enhanced_data
        self.creation_time[memory_id] = enhanced_data['timestamp']
        self.memory_scores[memory_id] = enhanced_data['importance_score']
        self.access_counts[memory_id] = 0
        self.last_accessed[memory_id] = enhanced_data['timestamp']
        
        # Add to specialized indices
        if memory_type == 'episodic':
            timestamp = enhanced_data['timestamp']
            self.episodic_index[int(timestamp)].append(memory_id)
        elif memory_type == 'semantic':
            # Would add to semantic index in reality
            pass
        
        self.total_memories_stored += 1
        self.logger.debug(f"Stored enhanced memory {memory_id} in {memory_type} memory. "
                         f"Total: {len(self.memories[memory_type])}")
        
        return memory_id

    def _prune_memory_by_score(self, memory_type: str, capacity: int):
        memories = self.memories.get(memory_type)
        if not memories or len(memories) <= capacity:
            return
        scored = [(mem.get('importance_score', 0.0), i, mem) for i, mem in enumerate(memories)]
        scored.sort(key=lambda x: x[0])
        to_remove = scored[:len(memories) - capacity]
        ids_to_remove = set()
        for score, idx, mem in to_remove:
            mem_id = mem.get('id', '')
            if mem_id in self.memory_index:
                del self.memory_index[mem_id]
            ids_to_remove.add(mem_id)
        new_memories = [mem for mem in memories if mem.get('id', '') not in ids_to_remove]
        if memory_type == 'working':
            self.memories[memory_type] = deque(new_memories, maxlen=self.memory_hierarchy['working']['capacity'])
        else:
            self.memories[memory_type] = new_memories
        self.logger.debug(f"Pruned {len(to_remove)} memories from {memory_type} (capacity={capacity})")
    
    def _retrieve_relevant_memories_advanced(self, prompt: str, 
                                           context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve memories using advanced scoring and ranking"""
        # Calculate query importance/prompt characteristics
        query_features = self._extract_query_features(prompt)
        
        # Retrieve from each memory type with appropriate strategy
        retrieved = []
        
        # 1. Working memory - recent items get boost
        with self._lock:
            working_memories = list(self.memories['working'])[-10:]  # Recent working memories
        for mem in working_memories:
            relevance = self._calculate_advanced_relevance(prompt, context, mem, query_features)
            if relevance > self.memory_threshold:
                mem_copy = mem['data'].copy()
                mem_copy['_memory_id'] = mem['id']
                mem_copy['_relevance_score'] = relevance
                mem_copy['_memory_type'] = mem['type']
                retrieved.append(mem_copy)
        
        # 2. Episodic memory - time-based and thematic retrieval
        episodic_memories = self._retrieve_episodic_memories(prompt, context, query_features)
        retrieved.extend(episodic_memories)
        
        # 3. Semantic memory - similarity-based retrieval
        semantic_memories = self._retrieve_semantic_memories(prompt, context, query_features)
        retrieved.extend(semantic_memories)
        
        # 4. Long-term memory - importance and stability-based retrieval
        lt_memories = self._retrieve_long_term_memories(prompt, context, query_features)
        retrieved.extend(lt_memories)
        
        # Rank all retrieved memories by composite score
        ranked_memories = self._rank_memories_by_composite_score(retrieved, prompt, context)
        
        # Update access statistics
        for mem in ranked_memories:
            mem_id = mem.get('_memory_id')
            if mem_id:
                self.access_counts[mem_id] += 1
                self.last_accessed[mem_id] = self._get_timestamp()
                # Boost score based on recency and frequency of access
                self._update_memory_score(mem_id)
        
        self.total_memories_retrieved += len(ranked_memories)
        
        # Return top memories (limit to prevent overload)
        max_to_return = min(10, len(ranked_memories))
        return ranked_memories[:max_to_return]
    
    def _extract_query_features(self, prompt: str) -> Dict[str, Any]:
        """Extract features from the prompt to guide memory retrieval"""
        words = re.findall(r'\b\w+\b', prompt.lower())
        return {
            'length': len(prompt),
            'word_count': len(words),
            'unique_words': len(set(words)),
            'keywords': set(words),  # Simplified
            'entities': self._extract_simple_entities(prompt),
            'temporal_indicators': any(word in prompt.lower() for word in 
                                     ['today', 'yesterday', 'tomorrow', 'recently', 'before', 'after']),
            'question_type': self._classify_question_type(prompt)
        }
    
    def _extract_simple_entities(self, text: str) -> List[str]:
        """Simple entity extraction (placeholder for real NER)"""
        # Look for capitalized words or known patterns
        entities = []
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(words[:5])  # Limit to avoid noise
        return entities
    
    def _classify_question_type(self, prompt: str) -> str:
        """Classify the type of question for better memory retrieval"""
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ['what', 'who', 'where', 'when']):
            return 'factual'
        elif any(word in prompt_lower for word in ['how', 'why']):
            return 'explanatory'
        elif any(word in prompt_lower for word in ['should', 'would', 'could']):
            return 'advice'
        elif any(word in prompt_lower for word in ['compare', 'difference']):
            return 'comparative'
        else:
            return 'general'
    
    def _calculate_advanced_relevance(self, prompt: str, context: Dict[str, Any], 
                                    memory: Dict[str, Any], query_features: Dict[str, Any]) -> float:
        """Calculate advanced relevance score using multiple factors"""
        memory_data = memory.get('data', {})
        if not isinstance(memory_data, dict):
            memory_data = {'content': str(memory_data)}
        
        memory_prompt = str(memory_data.get('prompt', ''))
        memory_context = str(memory_data.get('context', ''))
        
        # Factor 1: Textual similarity (simplified)
        textual_sim = self._calculate_text_similarity(prompt, memory_prompt + ' ' + memory_context)
        
        # Factor 2: Temporal relevance (how recent is the memory)
        memory_age = self._get_timestamp() - memory.get('timestamp', self._get_timestamp())
        temporal_relevance = max(0, 1 - (memory_age / 86400))  # Decay over a day
        
        # Factor 3: Importance score of the memory
        importance = memory.get('importance_score', 0.5)
        
        # Factor 4: Access frequency boost
        mem_id = memory.get('id', '')
        access_boost = min(self.access_counts.get(mem_id, 0) / 10.0, 0.5)  # Max 0.5 boost
        
        # Factor 5: Contextual relevance (matching context elements)
        contextual_sim = self._calculate_contextual_relevance(context, memory_data)
        
        # Factor 6: Query-feature matching
        feature_match = self._calculate_feature_match(query_features, memory_data)
        
        # Weighted combination
        weights = {
            'textual': 0.25,
            'temporal': 0.15,
            'importance': 0.2,
            'access': 0.1,
            'contextual': 0.2,
            'features': 0.1
        }
        
        score = (
            weights['textual'] * textual_sim +
            weights['temporal'] * temporal_relevance +
            weights['importance'] * importance +
            weights['access'] * access_boost +
            weights['contextual'] * contextual_sim +
            weights['features'] * feature_match
        )
        
        return min(max(score, 0.0), 1.0)  # Clamp to 0-1
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using cached word sets"""
        if not text1 or not text2:
            return 0.0
        return self._embedding_cache.similarity(text1, text2)
    
    def _calculate_contextual_relevance(self, context1: Dict[str, Any], 
                                      context2: Dict[str, Any]) -> float:
        """Calculate relevance based on context matching"""
        # Simple implementation: compare keys and values
        if not isinstance(context1, dict) or not isinstance(context2, dict):
            return 0.5 if context1 == context2 else 0.0
        
        # Compare keys
        keys1 = set(context1.keys())
        keys2 = set(context2.keys())
        key_similarity = len(keys1 & keys2) / max(len(keys1 | keys2), 1) if (keys1 | keys2) else 0.0
        
        # Compare some values (simplified)
        value_matches = 0
        total_comparisons = 0
        for key in keys1 & keys2:
            val1, val2 = context1[key], context2[key]
            if isinstance(val1, str) and isinstance(val2, str):
                total_comparisons += 1
                if val1.lower() == val2.lower():
                    value_matches += 1
            elif val1 == val2:
                total_comparisons += 1
                value_matches += 1
        
        value_similarity = value_matches / max(total_comparisons, 1) if total_comparisons > 0 else 0.5
        
        return (key_similarity * 0.6) + (value_similarity * 0.4)
    
    def _calculate_feature_match(self, query_features: Dict[str, Any], 
                               memory_data: Dict[str, Any]) -> float:
        """Match query features against memory data"""
        # Simplified feature matching
        score = 0.5  # Base score
        
        # Check for entity overlap
        query_entities = set(query_features.get('entities', []))
        memory_prompt = str(memory_data.get('prompt', ''))
        memory_entities = set(self._extract_simple_entities(memory_prompt))
        
        if query_entities or memory_entities:
            entity_overlap = len(query_entities & memory_entities) / max(len(query_entities | memory_entities), 1)
            score += 0.3 * entity_overlap
        
        # Check for temporal indicator matching
        query_temporal = query_features.get('temporal_indicators', False)
        memory_prompt_lower = memory_prompt.lower()
        memory_temporal = any(word in memory_prompt_lower for word in 
                            ['today', 'yesterday', 'tomorrow', 'recently', 'before', 'after'])
        
        if query_temporal == memory_temporal:
            score += 0.2
        
        return min(score, 1.0)
    
    def _retrieve_episodic_memories(self, prompt: str, context: Dict[str, Any], 
                                  query_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve memories from episodic storage"""
        retrieved = []
        episodic_memories = self.memories['episodic']
        
        # Sample from episodic memory (in reality, we'd use more sophisticated indexing)
        # For now, take recent items and score them
        recent_episodic = episodic_memories[-50:] if len(episodic_memories) > 50 else episodic_memories
        
        for mem in recent_episodic:
            relevance = self._calculate_advanced_relevance(prompt, context, mem, query_features)
            if relevance > self.memory_threshold:
                mem_copy = mem['data'].copy()
                mem_copy['_memory_id'] = mem['id']
                mem_copy['_relevance_score'] = relevance
                mem_copy['_memory_type'] = mem['type']
                retrieved.append(mem_copy)
        
        return retrieved
    
    def _retrieve_semantic_memories(self, prompt: str, context: Dict[str, Any], 
                                  query_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve memories from semantic storage using indexed search"""
        retrieved = []
        semantic_memories = self.memories['semantic']

        if not semantic_memories:
            return retrieved

        sorted_candidates = sorted(semantic_memories, key=lambda m: m.get('importance_score', 0), reverse=True)
        candidates = sorted_candidates[:100]
        scored = []
        for mem in candidates:
            relevance = self._calculate_advanced_relevance(prompt, context, mem, query_features)
            if relevance > self.memory_threshold:
                mem_copy = mem['data'].copy()
                mem_copy['_memory_id'] = mem['id']
                mem_copy['_relevance_score'] = relevance
                mem_copy['_memory_type'] = mem['type']
                scored.append((relevance, mem_copy))

        scored.sort(key=lambda x: x[0], reverse=True)
        retrieved = [mc for _, mc in scored[:10]]
        return retrieved
    
    def _retrieve_long_term_memories(self, prompt: str, context: Dict[str, Any], 
                                   query_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve memories from long-term storage"""
        retrieved = []
        lt_memories = self.memories['long_term']
        
        # Long-term retrieval focuses on high-importance, stable memories
        # Sort by importance score and take top candidates
        sorted_memories = sorted(lt_memories, 
                               key=lambda m: m.get('importance_score', 0), 
                               reverse=True)
        
        # Take top 20% or 50 memories, whichever is smaller
        top_count = max(1, min(len(sorted_memories) // 5, 50))
        candidates = sorted_memories[:top_count]
        
        for mem in candidates:
            relevance = self._calculate_advanced_relevance(prompt, context, mem, query_features)
            # Lower threshold for long-term since we pre-filtered by importance
            if relevance > (self.memory_threshold * 0.7):
                mem_copy = mem['data'].copy()
                mem_copy['_memory_id'] = mem['id']
                mem_copy['_relevance_score'] = relevance
                mem_copy['_memory_type'] = mem['type']
                retrieved.append(mem_copy)
        
        return retrieved
    
    def _rank_memories_by_composite_score(self, memories: List[Dict[str, Any]], 
                                        prompt: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank memories by a composite score including recency, frequency, and relevance"""
        def composite_score(mem):
            # Base relevance score
            relevance = mem.get('_relevance_score', 0.0)
            
            # Recency boost
            last_accessed = mem.get('_last_accessed', mem.get('timestamp', 0))
            age = self._get_timestamp() - last_accessed
            recency_boost = max(0, 1 - (age / (7 * 86400)))  # Boost for access within a week
            
            # Frequency boost
            mem_id = mem.get('_memory_id', '')
            freq = self.access_counts.get(mem_id, 0)
            freq_boost = min(freq / 20.0, 0.3)  # Max 0.3 boost for frequency
            
            # Importance from original storage
            importance = mem.get('_importance_score', mem.get('importance_score', 0.5))
            
            # Combine scores
            return (relevance * 0.5) + (recency_boost * 0.2) + (freq_boost * 0.2) + (importance * 0.1)
        
        return sorted(memories, key=composite_score, reverse=True)
    
    def _consolidate_memories_intelligent(self):
        """Intelligently consolidate memories between tiers incrementally"""
        consolidation_batch_size = 20
        with self._lock:
            # Working -> Episodic (incremental: only process batch_size items)
            if len(self.memories['working']) > self.memory_hierarchy['working']['capacity'] * 0.8:
                working_list = list(self.memories['working'])
                working_list.sort(key=lambda m: (
                    m.get('importance_score', 0.5) * 0.6 +
                    (1 - ((self._get_timestamp() - m.get('timestamp', 0)) / (24 * 3600))) * 0.4
                ), reverse=True)
                
                # Only move bottom batch_size items
                to_move = working_list[-consolidation_batch_size:]
                to_move_ids = {mem.get('id') for mem in to_move}
                
                remaining = [mem for mem in self.memories['working'] if mem.get('id') not in to_move_ids]
                self.memories['working'] = deque(remaining, maxlen=self.memory_hierarchy['working']['capacity'])
                
                for mem in to_move:
                    mem['type'] = 'episodic'
                    self.memories['episodic'].append(mem)
            
            episodic_capacity = self.memory_hierarchy['episodic']['capacity']
            if len(self.memories['episodic']) > episodic_capacity:
                self._prune_memory_by_score('episodic', episodic_capacity)

            # Episodic -> Semantic (incremental: only process batch_size items)
            if len(self.memories['episodic']) > 100:
                candidates = []
                for mem in self.memories['episodic']:
                    mem_id = mem.get('id', '')
                    access_freq = self.access_counts.get(mem_id, 0)
                    age = self._get_timestamp() - mem.get('timestamp', self._get_timestamp())
                    stability = max(0, 1 - (age / (30 * 86400)))
                    
                    score = (min(access_freq / 10.0, 1.0) * 0.6) + (stability * 0.4)
                    if score > 0.5:
                        candidates.append((score, mem))
                
                candidates.sort(reverse=True)
                to_move = [mem for score, mem in candidates[:consolidation_batch_size]]
                to_move_ids = {mem.get('id') for mem in to_move}
                
                self.memories['episodic'] = [mem for mem in self.memories['episodic'] if mem.get('id') not in to_move_ids]
                
                for mem in to_move:
                    mem['type'] = 'semantic'
                    self.memories['semantic'].append(mem)
    
    def _cleanup_loop(self):
        while not self._shutdown_event.is_set():
            if self._shutdown_event.wait(3600):
                break
            try:
                self._apply_forgetting_curve()
            except Exception:
                logger.exception("Forgetting curve cleanup failed")
    
    def shutdown(self):
        """Shut down the memory manager and cleanup thread."""
        self._shutdown_event.set()
        self._cleanup_thread.join(timeout=5)

    def __del__(self):
        """Ensure cleanup thread is stopped on garbage collection."""
        try:
            self.shutdown()
        except Exception:
            pass

    def _apply_forgetting_curve(self):
        """Apply forgetting curve to reduce memory strength over time"""
        with self._lock:
            current_time = self._get_timestamp()
            
            for memory_type in ['working', 'episodic', 'semantic', 'long_term']:
                if memory_type not in self.memories:
                    continue
                
                decay_rate = self.memory_hierarchy.get(memory_type, {}).get('decay_rate', 0.95)
                source_memories = self.memories[memory_type]
                
                surviving = []
                for mem in source_memories:
                    mem_id = mem.get('id', '')
                    if not mem_id:
                        surviving.append(mem)
                        continue
                    
                    age_hours = (current_time - mem.get('timestamp', current_time)) / 3600
                    decay_factor = decay_rate ** (age_hours / 24)
                    original_score = mem.get('importance_score', 0.5)
                    new_score = original_score * decay_factor
                    
                    mem['importance_score'] = new_score
                    self.memory_scores[mem_id] = new_score
                    
                    if new_score >= 0.1:
                        surviving.append(mem)
                    elif mem_id in self.memory_index:
                        del self.memory_index[mem_id]
                        self.logger.debug("Removed memory %s due to forgetting curve", mem_id)
                
                if memory_type == 'working':
                    self.memories[memory_type] = deque(surviving, maxlen=self.memory_hierarchy['working']['capacity'])
                else:
                    self.memories[memory_type] = surviving
    
    def _update_memory_rankings(self):
        """Update memory rankings based on recent access patterns"""
        # Boost scores for recently and frequently accessed memories
        current_time = self._get_timestamp()
        
        for mem_id, last_access in self.last_accessed.items():
            age_hours = (current_time - last_access) / 3600
            access_freq = self.access_counts.get(mem_id, 0)
            
            # Recency boost: accessed recently gets boost
            recency_boost = max(0, 1 - (age_hours / 168))  # Boost decays over a week
            
            # Frequency boost: logarithmic scaling
            if access_freq > 0:
                if np is not None:
                    freq_boost = min(0.5 * (1 - np.exp(-access_freq / 5.0)), 0.5)
                else:
                    freq_boost = min(0.5 * (1 - math.exp(-access_freq / 5.0)), 0.5)
            else:
                freq_boost = 0.0
            
            # Combined boost
            boost = (recency_boost * 0.6) + (freq_boost * 0.4)
            
            # Apply boost (but don't exceed 1.0)
            current_score = self.memory_scores.get(mem_id, 0.5)
            new_score = min(current_score + boost, 1.0)
            self.memory_scores[mem_id] = new_score
            
            # Update the memory object if it exists
            if mem_id in self.memory_index:
                self.memory_index[mem_id]['importance_score'] = new_score
    
    def _get_enhanced_memory_stats(self) -> Dict[str, Any]:
        """Get enhanced memory statistics"""
        stats = {}
        total_memories = 0
        
        for mem_type, memories in self.memories.items():
            count = len(memories)
            stats[f'{mem_type}_count'] = count
            total_memories += count
            
            # Calculate average importance for this type
            if memories:
                importance_sum = sum(mem.get('importance_score', 0) for mem in memories)
                avg_importance = importance_sum / len(memories)
                stats[f'{mem_type}_avg_importance'] = avg_importance
            else:
                stats[f'{mem_type}_avg_importance'] = 0.0
        
        stats['total_memories'] = total_memories
        stats['total_stored'] = self.total_memories_stored
        stats['total_retrieved'] = self.total_memories_retrieved
        total_accesses = self.hits + self.misses
        stats['hit_rate'] = self.hits / max(total_accesses, 1)
        
        # Memory score distribution
        if self.memory_scores:
            scores = list(self.memory_scores.values())
            stats['memory_score_mean'] = sum(scores) / len(scores)
            stats['memory_score_std'] = (sum((x - stats['memory_score_mean'])**2 for x in scores) / len(scores))**0.5 if len(scores) > 1 else 0.0
            stats['memory_score_min'] = min(scores)
            stats['memory_score_max'] = max(scores)
        
        return stats
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get memory metrics"""
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'memory_types': self.memory_types,
            'max_memories': self.max_memories,
            'memory_threshold': self.memory_threshold,
            'consolidation_threshold': self.consolidation_threshold,
            'stats': self._get_enhanced_memory_stats()
        })
        return base_metrics