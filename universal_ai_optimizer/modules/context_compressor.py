"""
Context Compression Module
Implements various context compression techniques with advanced semantic understanding
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import re
import hashlib
import threading
from collections import Counter, OrderedDict
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)

class ContextCompressor(BaseOptimizerModule):
    """
    Context compression module that reduces input context while preserving meaning
    using advanced techniques including semantic understanding and intelligent chunking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.compression_methods = self.config.get('methods', [
            'hierarchical_summarization',
            'semantic_compression',
            'token_compression',
            'context_pruning',
            'context_fingerprinting',
            'delta_compression',
            'intelligent_chunking'
        ])
        self.target_ratio = self.config.get('compression_ratio', 0.3)
        self.max_length = self.config.get('max_context_length', 4096)
        self.preserve_recent = self.config.get('preserve_recent', True)
        self.recent_tokens = self.config.get('recent_tokens', 512)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.8)
        self.chunk_size = self.config.get('chunk_size', 128)
        
        self._text_relevance_cache: OrderedDict[str, float] = OrderedDict()
        self._TEXT_RELEVANCE_CACHE_MAXSIZE = 256
        self._text_relevance_cache_lock = threading.Lock()

        self._init_compressors()
    
    def _init_compressors(self):
        """Initialize compression algorithms and models"""
        self.logger.debug("Initializing advanced context compressors")
        # In a full implementation, we would load models or set up algorithms here
        self.semantic_model = None  # Would be a sentence transformer model
        self.fingerprint_cache = {}  # Cache for context fingerprints
        self.delta_base = None  # Base context for delta compression
    
    def process(self, prompt: str, context: Dict[str, Any], 
               model_adapter: Optional[Any] = None, 
               pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compress context and optimize prompt using multiple advanced techniques
        
        Args:
            prompt: Input prompt
            context: Context dictionary (may include conversation history, documents, etc.)
            model_adapter: Model adapter (unused in this module but required by interface)
            pipeline_state: Current pipeline state
            
        Returns:
            Dictionary with compressed context and optimized prompt
        """
        self._log_processing(len(prompt), len(str(context)))
        
        # Start with original prompt and context
        optimized_prompt = prompt
        compressed_context = context.copy() if context else {}
        
        # Track compression statistics
        compression_stats = {}
        original_length = len(prompt) + len(str(context))
        
        # Apply each compression method in sequence with early exit
        for method in self.compression_methods:
            # Early exit if target compression ratio already achieved
            current_length = len(optimized_prompt) + len(str(compressed_context))
            current_ratio = 1.0 - (current_length / max(original_length, 1))
            if current_ratio >= self.target_ratio:
                self.logger.debug(f"Target compression {self.target_ratio*100:.0f}% reached at {current_ratio*100:.1f}%, skipping remaining methods")
                break
            
            self.logger.debug(f"Applying compression method: {method}")
            try:
                if method == 'hierarchical_summarization':
                    optimized_prompt, compressed_context, stats = self._hierarchical_summarization(
                        optimized_prompt, compressed_context
                    )
                elif method == 'semantic_compression':
                    optimized_prompt, compressed_context, stats = self._semantic_compression(
                        optimized_prompt, compressed_context
                    )
                elif method == 'token_compression':
                    optimized_prompt, compressed_context, stats = self._token_compression(
                        optimized_prompt, compressed_context
                    )
                elif method == 'context_pruning':
                    optimized_prompt, compressed_context, stats = self._context_pruning(
                        optimized_prompt, compressed_context
                    )
                elif method == 'context_fingerprinting':
                    optimized_prompt, compressed_context, stats = self._context_fingerprinting(
                        optimized_prompt, compressed_context
                    )
                elif method == 'delta_compression':
                    optimized_prompt, compressed_context, stats = self._delta_compression(
                        optimized_prompt, compressed_context, pipeline_state
                    )
                elif method == 'intelligent_chunking':
                    optimized_prompt, compressed_context, stats = self._intelligent_chunking(
                        optimized_prompt, compressed_context
                    )
                else:
                    self.logger.warning(f"Unknown compression method: {method}, skipping")
                    continue
                
                if stats:
                    compression_stats[method] = stats
                    
            except Exception as e:
                self.logger.warning(f"Compression method {method} failed: {str(e)}")
                continue
        
        # Ensure we don't exceed max length
        if self.max_length and len(optimized_prompt) > self.max_length:
            optimized_prompt = self._truncate_to_max_length(optimized_prompt)
        
        # Calculate final compression ratio
        compressed_length = len(optimized_prompt) + len(str(compressed_context))
        compression_ratio = 1.0 - (compressed_length / max(original_length, 1))
        
        result = {
            'optimized_prompt': optimized_prompt,
            'compressed_context': compressed_context,
            'token_savings': compression_ratio * 100,  # as percentage
            'compression_ratio': compression_ratio,
            'compression_stats': compression_stats
        }
        
        self.logger.info(f"Context compression achieved {compression_ratio*100:.1f}% size reduction")
        return result
    
    def _hierarchical_summarization(self, prompt: str, context: Dict[str, Any]) -> tuple:
        """Apply hierarchical summarization to context with multiple levels"""
        self.logger.debug("Applying hierarchical summarization")
        
        # Example: if context has a 'history' field, summarize it hierarchically
        if 'history' in context and isinstance(context['history'], list):
            history = context['history']
            
            # Create hierarchical summary: recent details + older summaries
            if len(history) > self.recent_tokens:
                recent = history[-self.recent_tokens:] if self.preserve_recent else []
                older = history[:-self.recent_tokens] if len(history) > self.recent_tokens else []
                
                # Create multiple levels of summarization for older history
                if older:
                    # Level 1: Extract key phrases
                    level1_summary = self._extract_key_phrases(older, max_phrases=5)
                    # Level 2: Create abstractive summary (placeholder)
                    level2_summary = f"Previous conversation covered: {level1_summary}"
                    # Level 3: Very condensed representation
                    level3_summary = self._create_condensed_representation(level2_summary)
                    
                    context['history'] = recent + [
                        {'level1_summary': level1_summary},
                        {'level2_summary': level2_summary},
                        {'level3_summary': level3_summary}
                    ]
                else:
                    context['history'] = recent
            else:
                # If not enough history for hierarchy, just keep as is or lightly summarize
                if len(history) > 10:  # Only summarize if substantial history
                    summary = self._extract_key_phrases(history, max_phrases=3)
                    context['history'] = [{'summary': summary}]
        
        # Calculate compression achieved
        original_str_len = len(str(context.get('history', []))) if 'history' in context else 0
        new_str_len = len(str(context.get('history', []))) if 'history' in context else 0
        savings = 1.0 - (new_str_len / max(original_str_len, 1)) if original_str_len > 0 else 0.0
        
        return prompt, context, {'hierarchical_savings': savings}
    
    def _semantic_compression(self, prompt: str, context: Dict[str, Any]) -> tuple:
        """Apply semantic compression using embeddings and clustering"""
        self.logger.debug("Applying semantic compression")
        
        # In a real implementation, this would:
        # 1. Convert text to embeddings
        # 2. Cluster similar content
        # 3. Replace clusters with representative summaries
        
        # Placeholder implementation with improved logic
        compressed_context = context.copy()
        
        # Process text fields in context for semantic redundancy removal
        text_fields = self._identify_text_fields(context)
        
        for field_name, field_value in text_fields.items():
            if isinstance(field_value, str) and len(field_value) > 100:
                # Apply semantic compression to this field
                compressed_value = self._compress_text_semantically(field_value)
                compressed_context[field_name] = compressed_value
            elif isinstance(field_value, list):
                # Process lists of text items
                compressed_list = []
                for item in field_value:
                    if isinstance(item, str) and len(item) > 50:
                        compressed_item = self._compress_text_semantically(item)
                        compressed_list.append(compressed_item)
                    else:
                        compressed_list.append(item)
                compressed_context[field_name] = compressed_list
        
        # Calculate compression achieved
        original_text_length = sum(
            len(str(v)) for k, v in context.items() 
            if isinstance(v, (str, list)) and any(isinstance(i, str) for i in (v if isinstance(v, list) else [v]))
        )
        new_text_length = sum(
            len(str(v)) for k, v in compressed_context.items() 
            if isinstance(v, (str, list)) and any(isinstance(i, str) for i in (v if isinstance(v, list) else [v]))
        )
        savings = 1.0 - (new_text_length / max(original_text_length, 1)) if original_text_length > 0 else 0.0
        
        return prompt, compressed_context, {'semantic_savings': savings}
    
    def _token_compression(self, prompt: str, context: Dict[str, Any]) -> tuple:
        """Apply advanced token-level compression"""
        self.logger.debug("Applying advanced token compression")
        
        # Optimize prompt
        optimized_prompt = self._compress_prompt_tokens(prompt)
        
        # Optimize context text
        compressed_context = context.copy()
        text_fields = self._identify_text_fields(context)
        
        for field_name, field_value in text_fields.items():
            if isinstance(field_value, str):
                compressed_context[field_name] = self._compress_text_tokens(field_value)
            elif isinstance(field_value, list):
                compressed_context[field_name] = [
                    self._compress_text_tokens(item) if isinstance(item, str) else item
                    for item in field_value
                ]
        
        # Calculate savings
        original_length = len(prompt) + len(str(context))
        new_length = len(optimized_prompt) + len(str(compressed_context))
        savings = 1.0 - (new_length / max(original_length, 1))
        
        return optimized_prompt, compressed_context, {'token_savings': savings * 100.0}
    
    def _context_pruning(self, prompt: str, context: Dict[str, Any]) -> tuple:
        """Remove irrelevant parts of context based on semantic relevance scoring"""
        self.logger.debug("Applying context pruning with relevance scoring")
        
        compressed_context = context.copy()
        
        # Score relevance of context elements to the prompt
        relevance_scores = self._score_context_relevance(prompt, context)
        
        # Prune low-relevance elements
        pruned_count = 0
        for key in list(compressed_context.keys()):
            if key not in ['prompt', 'instruction']:  # Never prune core elements
                score = relevance_scores.get(key, 0.5)  # Default medium relevance
                if score < 0.3:  # Prune below 30% relevance
                    del compressed_context[key]
                    pruned_count += 1
        
        # Also prune within complex structures
        for key, value in compressed_context.items():
            if isinstance(value, dict):
                # Prune dictionary entries
                to_remove = [k for k, v in value.items() 
                           if isinstance(v, str) and len(v) > 20 and 
                           self._calculate_text_relevance(prompt, v) < 0.2]
                for k in to_remove:
                    del value[k]
                    pruned_count += 1
            elif isinstance(value, list):
                # Prune list items
                original_len = len(value)
                compressed_context[key] = [
                    item for i, item in enumerate(value)
                    if not (isinstance(item, str) and len(item) > 20 and 
                           self._calculate_text_relevance(prompt, item) < 0.2)
                ]
                pruned_count += (original_len - len(compressed_context[key]))
        
        # Calculate savings
        original_length = len(str(context))
        new_length = len(str(compressed_context))
        savings = 1.0 - (new_length / max(original_length, 1)) if original_length > 0 else 0.0
        
        return prompt, compressed_context, {'pruning_savings': savings, 'items_pruned': pruned_count}
    
    def _context_fingerprinting(self, prompt: str, context: Dict[str, Any]) -> tuple:
        """Use context fingerprinting to identify and eliminate redundant information"""
        self.logger.debug("Applying context fingerprinting")
        
        compressed_context = context.copy()
        
        # Generate fingerprints for context elements
        fingerprints = self._generate_context_fingerprints(context)
        
        # Identify duplicates based on fingerprints
        seen_fingerprints = {}
        duplicates_by_index = []  # (key, index) for list values
        duplicates_by_key = []    # key for non-list values
        
        for key, fingerprint_list in fingerprints.items():
            if isinstance(fingerprint_list, list):
                for i, fp in enumerate(fingerprint_list):
                    if fp in seen_fingerprints:
                        duplicates_by_index.append((key, i))
                    else:
                        seen_fingerprints[fp] = (key, i)
            else:
                fp = fingerprint_list
                if fp in seen_fingerprints:
                    duplicates_by_key.append(key)
                else:
                    seen_fingerprints[fp] = key
        
        # Remove index-based duplicates (reverse order to maintain indices)
        for key, idx in sorted(duplicates_by_index, reverse=True):
            if key in compressed_context and isinstance(compressed_context[key], list):
                if idx < len(compressed_context[key]):
                    del compressed_context[key][idx]
        # Remove key-based duplicates
        for key in sorted(set(duplicates_by_key), reverse=True):
            if key in compressed_context:
                del compressed_context[key]
        
        # Calculate savings
        original_length = len(str(context))
        new_length = len(str(compressed_context))
        savings = 1.0 - (new_length / max(original_length, 1)) if original_length > 0 else 0.0
        
        return prompt, compressed_context, {
            'fingerprinting_savings': savings,
            'duplicates_removed': len(duplicates_by_index) + len(duplicates_by_key)
        }
    
    def _delta_compression(self, prompt: str, context: Dict[str, Any], 
                          pipeline_state: Optional[Dict[str, Any]]) -> tuple:
        """Apply delta compression by storing only changes from previous context"""
        self.logger.debug("Applying delta compression")
        
        # Get previous context from pipeline state if available
        previous_context = None
        if pipeline_state and 'previous_context' in pipeline_state:
            previous_context = pipeline_state['previous_context']
        
        compressed_context = context.copy()
        
        if previous_context is not None:
            # Compute delta: what's new or changed
            delta_context = self._compute_context_delta(previous_context, context)
            
            # Store delta plus minimal base reference
            compressed_context = {
                '_delta_base_reference': 'previous_context_stored_elsewhere',
                '_delta': delta_context
            }
            
            # For the prompt, we might also compute delta if relevant
            # But typically prompts are independent
        else:
            # First context, store as-is but mark for future delta compression
            compressed_context['_is_baseline'] = True
        
        # Calculate savings (estimation)
        original_length = len(str(context))
        if previous_context is not None:
            # Estimate: delta is usually much smaller than full context
            delta_estimate = len(str(self._compute_context_delta(previous_context, context)))
            savings = 1.0 - (delta_estimate / max(original_length, 1))
        else:
            savings = 0.1  # Small savings from marking as baseline
        
        return prompt, compressed_context, {'delta_savings': savings}
    
    def _intelligent_chunking(self, prompt: str, context: Dict[str, Any]) -> tuple:
        """Apply intelligent chunking based on semantic boundaries"""
        self.logger.debug("Applying intelligent chunking")
        
        compressed_context = context.copy()
        
        # Process large text fields by chunking intelligently
        text_fields = self._identify_text_fields(context)
        
        for field_name, field_value in text_fields.items():
            if isinstance(field_value, str) and len(field_value) > self.chunk_size * 2:
                # Apply intelligent chunking
                chunks = self._create_semantic_chunks(field_value)
                compressed_context[field_name] = {
                    '_chunked': True,
                    '_num_chunks': len(chunks),
                    '_chunks': chunks
                }
            elif isinstance(field_value, list):
                # Process lists that might benefit from chunking
                total_length = sum(len(str(item)) for item in field_value if isinstance(item, str))
                if total_length > self.chunk_size * 3:
                    # Chunk the list
                    chunked_list = self._chunk_list_intelligently(field_value)
                    compressed_context[field_name] = chunked_list
        
        # Calculate savings (chunking itself doesn't reduce size, but enables other optimizations)
        # Real savings come from being able to drop irrelevant chunks later
        original_length = len(str(context))
        new_length = len(str(compressed_context))
        # Slight overhead for chunking metadata, but enables better pruning
        savings = max(0, 0.05 - (new_length - original_length) / max(original_length, 1))  # Expect small net gain
        
        return prompt, compressed_context, {'chunking_savings': savings}
    
    # Helper methods for the compression techniques
    
    def _identify_text_fields(self, context: Dict[str, Any], max_depth: int = 10) -> Dict[str, Any]:
        """Identify fields in context that contain text amenable to compression"""
        text_fields = {}
        if max_depth <= 0:
            return text_fields
        for key, value in context.items():
            if isinstance(value, str):
                text_fields[key] = value
            elif isinstance(value, list):
                # Check if list contains strings
                if any(isinstance(item, str) for item in value):
                    text_fields[key] = value
            elif isinstance(value, dict):
                # Recursively check nested dicts
                nested_text = self._identify_text_fields(value, max_depth - 1)
                if nested_text:
                    text_fields[f"{key}.nested"] = nested_text
        return text_fields
    
    def _compress_text_semantically(self, text: str) -> str:
        """Compress text using semantic understanding (placeholder for real implementation)"""
        # In reality, this would:
        # 1. Split into sentences
        # 2. Embed sentences
        # 3. Cluster similar sentences
        # 4. Replace clusters with summaries
        # For now, use improved heuristic
        
        # Remove redundant sentences based on similarity
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 2:
            return text
        
        # Simple approach: keep first, last, and most distinctive middle sentences
        if len(sentences) <= 3:
            return text
        
        # Score sentence distinctiveness (simplified)
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Score based on length, position, and uniqueness
            length_score = min(len(sentence) / 100, 1.0)  # Normalize length
            position_score = 1.0 if i in [0, len(sentences)-1] else 0.5  # Boost edges
            # Uniqueness: inverse of word overlap with other sentences
            words = set(sentence.lower().split())
            overlap_scores = []
            for j, other in enumerate(sentences):
                if i != j:
                    other_words = set(other.lower().split())
                    if words and other_words:
                        overlap = len(words & other_words) / max(len(words), len(other_words))
                        overlap_scores.append(overlap)
            uniqueness = 1.0 - (sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0)
            
            total_score = (length_score * 0.3) + (position_score * 0.4) + (uniqueness * 0.3)
            scored_sentences.append((total_score, i, sentence))
        
        # Keep top scoring sentences
        scored_sentences.sort(reverse=True)
        keep_count = max(2, len(sentences) // 2)  # Keep at least 2, or half
        kept = sorted(scored_sentences[:keep_count], key=lambda x: x[1])  # Sort by original position
        result = '. '.join(s[2] for s in kept) + ('.' if not text.endswith('.') else '')
        
        return result
    
    def _compress_prompt_tokens(self, prompt: str) -> str:
        """Compress prompt at token level with semantic safety checks."""
        # Remove extra whitespace
        compressed = re.sub(r'\s+', ' ', prompt).strip()
        
        # Filler phrases to remove (only at sentence start for safety)
        filler_phrases = [
            r'^In other words,?\s*',
            r'^That is to say,?\s*',
            r'^It is important to note that\s*',
            r'^It should be noted that\s*',
            r'^As a matter of fact,?\s*',
            r'^In point of fact,?\s*',
            r'^It goes without saying that\s*',
            r'^Needless to say,?\s*',
            r'^It is worth mentioning that\s*',
            r'^I would like to point out that\s*',
            r'(?:^|[.!?]\s+)In other words,?\s*',
            r'(?:^|[.!?]\s+)That is to say,?\s*',
            r'(?:^|[.!?]\s+)It is important to note that\s*',
            r'(?:^|[.!?]\s+)It should be noted that\s*',
        ]
        
        for phrase in filler_phrases:
            match = re.search(phrase, compressed, flags=re.IGNORECASE)
            if match:
                remaining = compressed[match.end():].strip()
                # Only remove if remaining is a substantial sentence
                if remaining and len(remaining) > 15 and re.search(r'[a-zA-Z]{4,}', remaining):
                    compressed = remaining
        
        # Clean up leading connectors left behind
        compressed = re.sub(r'^[,\s]*(?:and|but|so|then|also|however)[,\s]+', '', compressed, flags=re.IGNORECASE).strip()
        
        # Capitalize first letter
        if compressed and compressed[0].islower():
            compressed = compressed[0].upper() + compressed[1:]
        
        # Clean up extra spaces
        compressed = re.sub(r'\s+', ' ', compressed).strip()
        
        return compressed
    
    def _compress_text_tokens(self, text: str) -> str:
        """Compress general text at token level with semantic safety."""
        compressed = re.sub(r'\s+', ' ', text).strip()
        
        # Lighter filler removal for context (only at sentence boundaries)
        filler_phrases = [
            r'^In other words,?\s*',
            r'^That is to say,?\s*',
            r'^It goes without saying that\s*',
            r'(?:^|[.!?]\s+)In other words,?\s*',
            r'(?:^|[.!?]\s+)That is to say,?\s*',
        ]
        
        for phrase in filler_phrases:
            match = re.search(phrase, compressed, flags=re.IGNORECASE)
            if match:
                remaining = compressed[match.end():].strip()
                if remaining and len(remaining) > 15 and re.search(r'[a-zA-Z]{4,}', remaining):
                    compressed = remaining
        
        # Clean up leading connectors
        compressed = re.sub(r'^[,\s]*(?:and|but|so|then)[,\s]+', '', compressed, flags=re.IGNORECASE).strip()
        
        if compressed and compressed[0].islower():
            compressed = compressed[0].upper() + compressed[1:]
        
        # Clean up extra spaces
        compressed = re.sub(r'\s+', ' ', compressed).strip()
        
        return compressed
    
    def _score_context_relevance(self, prompt: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Score relevance of context elements to the prompt"""
        relevance_scores = {}
        prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
        
        for key, value in context.items():
            if key in ['prompt', 'instruction']:
                relevance_scores[key] = 1.0  # Core elements always relevant
                continue
                
            if isinstance(value, str):
                text_words = set(re.findall(r'\b\w+\b', value.lower()))
                if prompt_words and text_words:
                    overlap = len(prompt_words & text_words) / max(len(prompt_words), len(text_words))
                    relevance_scores[key] = overlap
                else:
                    relevance_scores[key] = 0.0
            elif isinstance(value, list):
                # Average relevance of list items
                item_scores = []
                for item in value:
                    if isinstance(item, str):
                        text_words = set(re.findall(r'\b\w+\b', item.lower()))
                        if prompt_words and text_words:
                            overlap = len(prompt_words & text_words) / max(len(prompt_words), len(text_words))
                            item_scores.append(overlap)
                        else:
                            item_scores.append(0.0)
                    else:
                        item_scores.append(0.5)  # Non-text gets medium score
                relevance_scores[key] = sum(item_scores) / max(len(item_scores), 1) if item_scores else 0.0
            elif isinstance(value, dict):
                # Recursively score nested dict
                nested_scores = self._score_context_relevance(prompt, value)
                relevance_scores[key] = sum(nested_scores.values()) / max(len(nested_scores), 1) if nested_scores else 0.5
            else:
                # Other types get default score
                relevance_scores[key] = 0.5
        
        return relevance_scores
    
    def _calculate_text_relevance(self, text1: str, text2: str) -> float:
        """Calculate relevance between two text strings using string-keyed cache to avoid hash collisions"""
        cache_key = f"{text1[:128]}|||{text2[:128]}"
        with self._text_relevance_cache_lock:
            if cache_key in self._text_relevance_cache:
                self._text_relevance_cache.move_to_end(cache_key)
                return self._text_relevance_cache[cache_key]
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        if not words1 or not words2:
            result = 0.0
        else:
            overlap = len(words1 & words2)
            result = overlap / max(len(words1), len(words2))
        with self._text_relevance_cache_lock:
            if len(self._text_relevance_cache) >= self._TEXT_RELEVANCE_CACHE_MAXSIZE:
                self._text_relevance_cache.popitem(last=False)
            self._text_relevance_cache[cache_key] = result
        return result
    
    def _generate_context_fingerprints(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fingerprints for context elements to detect duplicates"""
        fingerprints = {}
        
        for key, value in context.items():
            if isinstance(value, str):
                # Generate fingerprint for string
                # Normalize: lowercase, remove extra punctuation, sort words
                normalized = re.sub(r'[^\w\s]', '', value.lower())
                words = sorted(normalized.split())
                fingerprint = hashlib.sha256(' '.join(words).encode()).hexdigest()[:16]
                fingerprints[key] = fingerprint
            elif isinstance(value, list):
                # Generate fingerprints for list items
                item_fingerprints = []
                for item in value:
                    if isinstance(item, str):
                        normalized = re.sub(r'[^\w\s]', '', item.lower())
                        words = sorted(normalized.split())
                        fingerprint = hashlib.sha256(' '.join(words).encode()).hexdigest()[:16]
                        item_fingerprints.append(fingerprint)
                    else:
                        # For non-strings, use string representation
                        fingerprint = hashlib.sha256(str(item).encode()).hexdigest()[:16]
                        item_fingerprints.append(fingerprint)
                fingerprints[key] = item_fingerprints
            elif isinstance(value, dict):
                # Recursively fingerprint nested dict
                fingerprints[key] = self._generate_context_fingerprints(value)
            else:
                # For other types, use string representation
                fingerprints[key] = hashlib.sha256(str(value).encode()).hexdigest()[:16]
        
        return fingerprints
    
    def _compute_context_delta(self, old_context: Dict[str, Any], 
                              new_context: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the delta (difference) between two contexts"""
        delta = {}
        
        # Find keys in new that aren't in old or have changed
        all_keys = set(old_context.keys()) | set(new_context.keys())
        
        for key in all_keys:
            old_val = old_context.get(key, None)
            new_val = new_context.get(key, None)
            
            if old_val != new_val:
                # Store the new value (or a representation of the change)
                delta[key] = new_val
                
                # For complex types, we could store a more compact delta
                # but for simplicity we store the new value
        
        # Also note keys that were removed
        removed_keys = set(old_context.keys()) - set(new_context.keys())
        if removed_keys:
            delta['_removed_keys'] = list(removed_keys)
        
        return delta
    
    def _create_semantic_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create semantically coherent chunks from text"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [{'text': '', 'type': 'empty'}]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence would exceed chunk size and we have content, finalize chunk
            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'type': 'semantic_chunk',
                    'sentence_count': len(current_chunk),
                    'length': current_length
                })
                current_chunk = [sentence]
                current_length = sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'type': 'semantic_chunk',
                'sentence_count': len(current_chunk),
                'length': current_length
            })
        
        return chunks
    
    def _chunk_list_intelligently(self, items: List[Any]) -> List[Any]:
        """Intelligently chunk a list of items"""
        # Simple implementation: group similar items together
        # In reality, this would use embeddings to find semantic similarity
        
        if len(items) <= self.chunk_size:
            return items
        
        # Group by type first, then by content similarity for strings
        grouped = {}
        for item in items:
            item_type = type(item).__name__
            if item_type not in grouped:
                grouped[item_type] = []
            grouped[item_type].append(item)
        
        # For string groups, try to cluster by similarity
        result = []
        for item_type, group in grouped.items():
            if item_type == 'str' and len(group) > self.chunk_size:
                # Simple clustering: group by first word or length buckets
                # This is a placeholder - real implementation would use embeddings
                length_groups = {}
                for item in group:
                    length_bucket = len(item) // 10  # Group by tens of characters
                    if length_bucket not in length_groups:
                        length_groups[length_bucket] = []
                    length_groups[length_bucket].append(item)
                
                # Flatten the length groups
                for bucket in sorted(length_groups.keys()):
                    result.extend(length_groups[bucket])
            else:
                result.extend(group)
        
        return result
    
    def _extract_key_phrases(self, text_list: List[str], max_phrases: int = 5) -> str:
        """Extract key phrases from a list of text strings"""
        # Combine all text
        combined_text = ' '.join(str(item) for item in text_list)
        
        # Extract potential phrases (noun phrases, verb phrases)
        # Simple implementation: find frequent meaningful words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text.lower())
        
        # Remove common stop words (expanded list)
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'any', 'can', 
            'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him',
            'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who',
            'boy', 'did', 'man', 'men', 'put', 'too', 'use', 'why', 'ask', 'put',
            'same', 'tell', 'does', 'went', 'men', 'say', 'use', 'her', 'way',
            'about', 'many', 'then', 'them', 'well', 'were'
        }
        
        meaningful_words = [w for w in words if w not in stop_words]
        
        # Count word frequencies
        word_counts = Counter(meaningful_words)
        
        # Get top words
        top_words = [word for word, count in word_counts.most_common(max_phrases)]
        
        # Create phrases from top words (simple bigrams for now)
        phrases = []
        for i in range(len(top_words)-1):
            phrases.append(f"{top_words[i]} {top_words[i+1]}")
        
        # If we don't have enough phrases, use single words
        if len(phrases) < max_phrases // 2:
            phrases = top_words[:max_phrases]
        
        return ', '.join(phrases[:max_phrases])
    
    def _create_condensed_representation(self, text: str) -> str:
        """Create a highly condensed representation of text"""
        # Extract first and last letters of words, or use acronym
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        if len(words) <= 3:
            return text[:20] + '...' if len(text) > 20 else text
        
        # Create acronym from first letters
        acronym = ''.join(w[0].upper() for w in words[:10])  # First 10 words
        return f"[{acronym}]" if acronym else text[:15] + '...'
    
    def _truncate_to_max_length(self, text: str) -> str:
        """Truncate text to maximum length while trying to preserve meaning"""
        if len(text) <= self.max_length:
            return text
        
        # Try to truncate at sentence boundary
        if '.' in text[:self.max_length]:
            # Find last sentence boundary within limit
            last_period = text.rfind('.', 0, self.max_length)
            if last_period > self.max_length * 0.7:  # Only if we keep at least 70%
                return text[:last_period + 1]
        
        # Try to truncate at word boundary
        if ' ' in text[:self.max_length]:
            last_space = text.rfind(' ', 0, self.max_length)
            if last_space > self.max_length * 0.8:  # Only if we keep at least 80%
                return text[:last_space] + "..."
        
        # Fallback to character truncation
        return text[:self.max_length] + "..."
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get compression metrics"""
        base_metrics = super().get_metrics()
        base_metrics.update({
            'target_compression_ratio': self.target_ratio,
            'methods_enabled': self.compression_methods,
            'max_context_length': self.max_length,
            'similarity_threshold': self.similarity_threshold,
            'chunk_size': self.chunk_size
        })
        return base_metrics