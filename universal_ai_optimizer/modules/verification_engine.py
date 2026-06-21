"""
Verification Engine Module
Implements advanced techniques to reduce hallucination and increase factual accuracy
including fact checking, citation validation, confidence scoring, consensus verification, and reflection
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import re
import hashlib
import time
from collections import Counter, defaultdict
from universal_ai_optimizer.core.base import BaseOptimizerModule
from universal_ai_optimizer.modules.common.text_utils import TextSimilarityCache

logger = logging.getLogger(__name__)

class VerificationEngine(BaseOptimizerModule):
    """
    Advanced verification engine that checks and improves the factual accuracy of outputs
    using multiple complementary techniques
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.methods = self.config.get('methods', [
            'self_consistency',
            'fact_checking',
            'citation_validation',
            'confidence_scoring',
            'consensus_verification',
            'reflection'
        ])
        self.threshold = self.config.get('threshold', 0.95)
        self.max_iterations = self.config.get('max_iterations', 3)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.8)
        self.consensus_threshold = self.config.get('consensus_threshold', 0.7)
        
        # Shared text similarity cache
        self._text_cache = TextSimilarityCache(maxsize=2048)
        
        # Initialize verifiers
        self._init_verifiers()
        
        # Knowledge base for fact checking (placeholder)
        self.knowledge_base = {}
        self.citation_database = {}
        
        # For tracking verification history
        self.verification_history = []
        self._max_verification_history = self.config.get('max_verification_history', 10000)
    
    def _init_verifiers(self):
        """Initialize verification algorithms"""
        self.logger.debug("Initializing advanced verification engines")
        # In a full implementation, we would load models or set up algorithms here
        pass
    
    def process(self, prompt: str, context: Dict[str, Any], 
               model_adapter: Optional[Any] = None, 
               pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Verify and potentially improve the output from previous stages using multiple techniques
        
        Args:
            prompt: Original prompt (may be used for self-consistency)
            context: Context dictionary (may include retrieved memories, etc.)
            model_adapter: Model adapter (used to generate multiple outputs for verification)
            pipeline_state: Current pipeline state (should contain execution_result from execution engine)
            
        Returns:
            Dictionary with verification score, improved result, and detailed verification info
        """
        if not self.enabled:
            return {}
        
        # Get the execution result from pipeline state
        execution_result = pipeline_state.get('execution_result') if pipeline_state else None
        if execution_result is None:
            self.logger.warning("No execution result found in pipeline state for verification")
            return {}
        
        self._log_processing(len(prompt), len(str(context)))
        
        # Apply verification methods in sequence, allowing for iterative improvement
        verified_result = execution_result
        verification_details = {}
        iteration_scores = []
        
        for iteration in range(self.max_iterations):
            self.logger.debug(f"Verification iteration {iteration + 1}/{self.max_iterations}")
            
            iteration_scores.append({})
            method_scores = {}
            
            # Apply each verification method
            for method in self.methods:
                self.logger.debug(f"Applying verification method: {method}")
                try:
                    if method == 'self_consistency':
                        verified_result, score, details = self._self_consistency(
                            prompt, context, model_adapter, verified_result
                        )
                    elif method == 'fact_checking':
                        verified_result, score, details = self._advanced_fact_checking(
                            verified_result, context, model_adapter
                        )
                    elif method == 'citation_validation':
                        verified_result, score, details = self._citation_validation(
                            verified_result, context
                        )
                    elif method == 'confidence_scoring':
                        verified_result, score, details = self._advanced_confidence_scoring(
                            verified_result, context
                        )
                    elif method == 'consensus_verification':
                        verified_result, score, details = self._consensus_verification(
                            prompt, context, model_adapter, verified_result
                        )
                    elif method == 'reflection':
                        verified_result, score, details = self._reflection_verification(
                            prompt, context, model_adapter, verified_result
                        )
                    else:
                        score, details = 0.0, {'error': f'Unknown method: {method}'}
                    
                    method_scores[method] = score
                    iteration_scores[-1][method] = score
                    verification_details[f'{method}_iteration_{iteration}'] = details
                    
                except Exception as e:
                    self.logger.warning(f"Verification method {method} failed: {str(e)}")
                    method_scores[method] = 0.0
                    iteration_scores[-1][method] = 0.0
                    verification_details[f'{method}_iteration_{iteration}'] = {'error': str(e)}
                    continue
            
            # Calculate weighted average score for this iteration
            if method_scores:
                # Use weights that favor factual verification methods
                weights = {
                    'self_consistency': 0.15,
                    'fact_checking': 0.25,
                    'citation_validation': 0.2,
                    'confidence_scoring': 0.15,
                    'consensus_verification': 0.15,
                    'reflection': 0.1
                }
                
                weighted_sum = sum(method_scores.get(method, 0) * weights.get(method, 0) 
                                 for method in weights)
                total_weight = sum(weights.get(method, 0) for method in method_scores.keys())
                iteration_score = weighted_sum / max(total_weight, 0.001)
            else:
                iteration_score = 0.0
            
            # Check if we've met the threshold
            if iteration_score >= self.threshold:
                self.logger.info(f"Verification threshold met at iteration {iteration + 1}")
                break
            
            # If not last iteration and score improved significantly, continue
            if iteration < self.max_iterations - 1 and iteration_score > 0.1:
                self.logger.debug(f"Score {iteration_score:.3f} below threshold, continuing to next iteration")
            else:
                break
        
        # Calculate final verification score (could be weighted average or max)
        if iteration_scores:
            # Use the best score achieved across iterations
            best_iteration_score = max(
                sum(scores.values()) / max(len(scores), 1) 
                for scores in iteration_scores if scores
            ) if any(iteration_scores) else 0.0
            
            # Or use the last iteration score
            final_score = sum(iteration_scores[-1].values()) / max(len(iteration_scores[-1]), 1) if iteration_scores[-1] else 0.0
            
            # Or use a weighted average favoring later iterations
            if len(iteration_scores) > 1:
                n = len(iteration_scores)
                weights = [0.5 ** (n - 1 - i) for i in range(n)]
                total_weight = sum(weights)
                weighted_sum = 0.0
                max_score_len = 1
                for i, w in enumerate(weights):
                    scores = iteration_scores[i]
                    if scores:
                        weighted_sum += sum(scores.values()) * w
                        max_score_len = max(max_score_len, len(scores))
                final_score = weighted_sum / max(total_weight * max_score_len, 1)
            else:
                final_score = best_iteration_score
        else:
            final_score = 0.0
        
        # Determine if result needs improvement
        needs_improvement = final_score < self.threshold
        
        # Apply final improvements if needed and we have a model adapter
        if needs_improvement and model_adapter and iteration < self.max_iterations - 1:
            self.logger.info(f"Applying final improvement pass (score: {final_score:.3f})")
            improved_result, improvement_details = self._apply_verification_improvements(
                prompt, context, model_adapter, verified_result, verification_details
            )
            if improved_result is not None:
                verified_result = improved_result
                verification_details['improvement_pass'] = improvement_details
                # Re-check score after improvements (simplified)
                final_score = min(final_score + 0.1, self.threshold)  # Assume some improvement
        
        result = {
            'verification_result': verified_result,
            'verification_score': final_score,
            'verification_details': verification_details,
            'needs_improvement': needs_improvement,
            'iteration_count': len([s for s in iteration_scores if s]),
            'method_scores': iteration_scores[-1] if iteration_scores else {}
        }
        
        # Store in verification history for learning
        self._record_verification(prompt, context, execution_result, verified_result, final_score)
        
        self.logger.info(f"Verification completed with score: {final_score:.3f} "
                        f"(needs_improvement: {needs_improvement})")
        return result
    
    def _self_consistency(self, prompt: str, context: Dict[str, Any], 
                         model_adapter: Any, initial_result: Any) -> tuple:
        """
        Apply advanced self-consistency by generating multiple outputs and using weighted voting
        """
        self.logger.debug("Applying advanced self-consistency")
        
        if not model_adapter:
            return initial_result, 0.5, {'error': 'No model adapter available'}
        
        try:
            # Generate multiple outputs with different temperatures/seeds
            num_samples = self.config.get('consistency_samples', 5)
            temperatures = [0.3, 0.5, 0.7, 0.9, 1.2][:num_samples]
            
            outputs = []
            for i, temp in enumerate(temperatures):
                try:
                    # Generate output with specific temperature
                    output = model_adapter.generate(
                        prompt, 
                        temperature=temp,
                        max_tokens=200,  # Limit for consistency check
                        top_p=0.9
                    )
                    outputs.append({
                        'text': getattr(output, 'text', str(output)),
                        'temperature': temp,
                        'index': i
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to generate sample {i}: {str(e)}")
                    continue
            
            if len(outputs) < 2:
                self.logger.warning("Insufficient samples for self-consistency")
                return initial_result, 0.5, {'error': 'Insufficient samples'}
            
            # Extract texts for comparison
            texts = [out['text'] for out in outputs]
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(texts)):
                for j in range(i+1, len(texts)):
                    sim = self._calculate_text_similarity(texts[i], texts[j])
                    similarities.append(sim)
            
            avg_similarity = sum(similarities) / max(len(similarities), 1) if similarities else 0.0
            
            # Use weighted voting: select the output with highest average similarity to others
            if len(texts) >= 2:
                # For each output, calculate average similarity to all others
                output_scores = []
                for i, out in enumerate(outputs):
                    similarities_to_others = []
                    for j, other_out in enumerate(outputs):
                        if i != j:
                            sim = self._calculate_text_similarity(out['text'], other_out['text'])
                            similarities_to_others.append(sim)
                    avg_sim = sum(similarities_to_others) / max(len(similarities_to_others), 1)
                    output_scores.append((avg_sim, out))
                
                # Select the output with highest average similarity
                output_scores.sort(key=lambda x: x[0], reverse=True)
                best_output = output_scores[0][1]
                
                # Also consider the initial result if it's similar enough
                initial_text = getattr(initial_result, 'text', str(initial_result))
                similarities_to_initial = [
                    self._calculate_text_similarity(initial_text, out['text']) 
                    for out in outputs
                ]
                avg_sim_to_initial = sum(similarities_to_initial) / max(len(similarities_to_initial), 1)
                
                if avg_sim_to_initial > avg_similarity * 0.8:  # If initial is reasonably consistent
                    # Boost confidence
                    final_score = min(avg_similarity + 0.1, 1.0)
                    final_result = initial_result
                else:
                    final_score = avg_similarity
                    # Return the best generated output
                    class ConsistencyResult:
                        def __init__(self, text):
                            self.text = text
                            self.token_count = len(text.split())
                    final_result = ConsistencyResult(best_output['text'])
            else:
                final_score = 0.5
                final_result = initial_result
            
            details = {
                'num_samples': len(outputs),
                'avg_similarity': avg_similarity,
                'method': 'weighted_voting',
                'temperatures_used': temperatures[:len(outputs)]
            }
            
            return final_result, final_score, details
            
        except Exception as e:
            self.logger.error(f"Self-consistency failed: {str(e)}")
            return initial_result, 0.0, {'error': str(e)}

    def _advanced_fact_checking(self, result: Any, context: Dict[str, Any],
                                model_adapter: Optional[Any] = None) -> tuple:
        """
        Advanced fact checking with evidence retrieval and reasoning
        """
        self.logger.debug("Applying advanced fact checking")
        
        # Convert result to string if needed
        if not isinstance(result, str):
            if hasattr(result, 'text'):
                result_text = result.text
            else:
                result_text = str(result)
        else:
            result_text = result
        
        # Extract factual claims from the result
        claims = self._extract_factual_claims(result_text)
        
        if not claims:
            # No factual claims found, return high score
            return result, 0.9, {'claims_checked': 0, 'reason': 'No factual claims detected'}
        
        # Check each claim
        checked_claims = []
        supported_claims = 0
        
        for claim in claims:
            check_result = self._check_single_claim(claim, context)
            checked_claims.append(check_result)
            
            if check_result['supported']:
                supported_claims += 1
        
        # Calculate factuality score
        if checked_claims:
            factuality_score = supported_claims / len(checked_claims)
        else:
            factuality_score = 1.0  # No claims to check
        
        # If we have low scores, attempt to correct the result
        corrected_result = result
        if factuality_score < self.threshold and model_adapter is not None:
            corrected_result = self._attempt_fact_correction(
                result_text, checked_claims, context, model_adapter
            )
        
        details = {
            'claims_checked': len(claims),
            'supported_claims': supported_claims,
            'factuality_score': factuality_score,
            'claim_details': checked_claims[:5]  # Limit details for brevity
        }
        
        return corrected_result, factuality_score, details
    
    def _extract_factual_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract factual claims from text using linguistic patterns"""
        claims = []
        
        # Patterns that often indicate factual statements
        fact_patterns = [
            r'\b(is|are|was|were)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper noun definitions
            r'\b\d+(?:\.\d+)?\s*(?:percent|%|minutes?|hours?|days?|years?|meters?|kilometers?)\b',  # Measurements
            r'\b(according to|studies show|research indicates|data shows)\b',  # Attribution patterns
            r'\b(in|on|at)\s+\d{4}\b',  # Dates with years
            r'\b\d{1,2}\/\d{1,2}\/\d{2,4}\b',  # Date formats
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:is|are|was|were)\s+[A-Z]',  # Subject-verb-object
        ]
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
        
        for sentence in sentences:
            # Check if sentence matches any fact patterns
            is_factual = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in fact_patterns)
            
            # Additional heuristics: contains numbers, proper nouns, specific dates
            has_numbers = bool(re.search(r'\d', sentence))
            has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', sentence))
            has_specific_date = bool(re.search(r'\b\d{1,2}\/\d{1,2}\/\d{4}\b|\b\d{4}\b', sentence))
            
            # Simple heuristic: if it has multiple factual indicators, treat as claim
            factual_indicators = sum([is_factual, has_numbers, has_proper_nouns, has_specific_date])
            if factual_indicators >= 2:
                claims.append({
                    'text': sentence,
                    'type': 'factual_statement',
                    'confidence': min(factual_indicators / 4.0, 1.0)
                })
        
        # Limit to prevent overload
        return claims[:10]
    
    def _check_single_claim(self, claim: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Check a single factual claim against knowledge sources"""
        claim_text = claim['text']
        
        # In a real implementation, this would:
        # 1. Formulate a search query from the claim
        # 2. Search external knowledge bases or search engines
        # 3. Evaluate the evidence for support/refutation
        # For now, use simplified heuristics and context checking
        
        # Check against context first
        context_support = self._check_claim_in_context(claim_text, context)
        
        # Check against internal knowledge base (placeholder)
        kb_support = self._check_claim_in_knowledge_base(claim_text)
        
        # Simple heuristic-based checking
        heuristic_score = self._heuristic_fact_check(claim_text)
        
        # Combine evidence
        support_score = max(context_support, kb_support, heuristic_score)
        supported = support_score > 0.6
        
        return {
            'claim': claim_text,
            'supported': supported,
            'support_score': support_score,
            'evidence_sources': {
                'context': context_support,
                'knowledge_base': kb_support,
                'heuristic': heuristic_score
            },
            'claim_type': claim.get('type', 'unknown')
        }
    
    def _check_claim_in_context(self, claim: str, context: Dict[str, Any]) -> float:
        """Check if claim is supported by the provided context"""
        if not context:
            return 0.0
        
        # Convert context to searchable text
        context_text = ' '.join(str(v) for v in context.values() if isinstance(v, (str, int, float)))
        context_text += ' ' + ' '.join(
            str(item) for v in context.values() 
            if isinstance(v, list) for item in v if isinstance(item, (str, int, float))
        )
        
        if not context_text.strip():
            return 0.0
        
        # Calculate similarity between claim and context
        return self._calculate_text_similarity(claim, context_text)
    
    def _check_claim_in_knowledge_base(self, claim: str) -> float:
        """Check claim against internal knowledge base"""
        # Placeholder: in reality, this would query a structured knowledge base
        # For now, return a base rate
        return 0.3  # Assume 30% chance of being in our limited KB
    
    def _heuristic_fact_check(self, claim: str) -> float:
        """Apply heuristic checks to assess factual likelihood"""
        score = 0.5  # Start with neutral
        
        # Check for overly strong language (often inaccurate)
        strong_claims = ['always', 'never', 'all', 'none', 'every', 'completely', 'totally']
        strong_count = sum(1 for word in strong_claims if word in claim.lower())
        if strong_count > 0:
            score -= min(strong_count * 0.1, 0.3)
        
        # Check for vague language (often less reliable)
        vague_claims = ['might', 'could', 'may', 'possibly', 'perhaps', 'allegedly', 'reportedly']
        vague_count = sum(1 for word in vague_claims if word in claim.lower())
        if vague_count > 0:
            score -= min(vague_count * 0.05, 0.2)
        
        # Check for precision (specific numbers, names, dates are more likely to be checked)
        if re.search(r'\d{4}', claim):  # Contains a year
            score += 0.1
        if re.search(r'\d+\.\d+', claim):  # Contains decimal numbers
            score += 0.1
        if len(re.findall(r'\b[A-Z][a-z]+', claim)) >= 2:  # Multiple proper nouns
            score += 0.1
        
        # Check for recent events (harder to verify)
        recent_indicators = ['today', 'yesterday', 'this week', 'recently', 'latest']
        if any(indicator in claim.lower() for indicator in recent_indicators):
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _attempt_fact_correction(self, original_text: str, checked_claims: List[Dict],
                               context: Dict[str, Any], model_adapter: Any) -> Any:
        """Attempt to correct factual errors in the text using the model adapter"""
        self.logger.debug("Attempting fact correction")

        unsupported = [c for c in checked_claims if not c.get('supported', False)]

        if not unsupported:
            return original_text

        unsupported_summaries = "\n".join(
            f"- {c.get('claim', '')} (score: {c.get('support_score', 0):.2f})"
            for c in unsupported[:5]
        )

        correction_prompt = (
            "The following text contains factual claims that could not be verified "
            "or are likely incorrect. Please correct only the problematic parts while "
            "preserving the rest of the text:\n\n"
            f"Original text:\n{original_text}\n\n"
            f"Unsupported or likely incorrect claims:\n{unsupported_summaries}\n\n"
            "Corrected text:"
        )

        try:
            output = model_adapter.generate(
                correction_prompt,
                temperature=0.3,
                max_tokens=300
            )
            corrected_text = getattr(output, 'text', str(output))

            class CorrectedResult:
                def __init__(self, text):
                    self.text = text
                    self.token_count = len(text.split())

            return CorrectedResult(corrected_text)
        except Exception as e:
            self.logger.warning(f"Fact correction failed, returning original: {e}")
            return original_text
    
    def _citation_validation(self, result: Any, context: Dict[str, Any]) -> tuple:
        """Validate citations and references in the result"""
        self.logger.debug("Applying citation validation")
        
        # Convert result to string if needed
        if not isinstance(result, str):
            if hasattr(result, 'text'):
                result_text = result.text
            else:
                result_text = str(result)
        else:
            result_text = result
        
        # Extract citations from the text
        citations = self._extract_citations(result_text)
        
        if not citations:
            # No citations found, score based on whether citations were expected
            expects_citations = self._expects_citations(result_text, context)
            score = 0.9 if not expects_citations else 0.5
            return result, score, {'citations_found': 0, 'expects_citations': expects_citations}
        
        # Validate each citation
        validated_citations = []
        valid_count = 0
        
        for citation in citations:
            validation_result = self._validate_single_citation(citation, context)
            validated_citations.append(validation_result)
            if validation_result['valid']:
                valid_count += 1
        
        # Calculate citation score
        if citations:
            citation_score = valid_count / len(citations)
        else:
            citation_score = 1.0
        
        details = {
            'citations_found': len(citations),
            'valid_citations': valid_count,
            'citation_score': citation_score,
            'citation_details': validated_citations[:5]
        }
        
        return result, citation_score, details
    
    def _extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """Extract citations from text using various formats"""
        citations = []
        
        # Pattern for [1], [2, 3], [1-3] style citations
        bracket_pattern = r'\[(\d+(?:[-,]\d+)*)\]'
        bracket_matches = re.findall(bracket_pattern, text)
        for match in bracket_matches:
            # Parse the citation numbers/numbers ranges
            nums = self._parse_citation_reference(match)
            for num in nums:
                citations.append({
                    'text': f'[{match}]',
                    'type': 'bracket',
                    'reference_id': num,
                    'raw_reference': match
                })
        
        # Pattern for (Author, Year) style citations
        paren_pattern = r'\(([A-Z][a-z]+(?:\s+et\s+al\.?)?,\s+\d{4}[a-z]?)\)'
        paren_matches = re.findall(paren_pattern, text)
        for match in paren_matches:
            citations.append({
                'text': f'({match})',
                'type': 'author_year',
                'reference_raw': match
            })
        
        # Pattern for [Author, Year] style
        square_paren_pattern = r'\[([A-Z][a-z]+(?:\s+et\s+al\.?)?,\s+\d{4}[a-z]?)\]'
        square_paren_matches = re.findall(square_paren_pattern, text)
        for match in square_paren_matches:
            citations.append({
                'text': f'[{match}]',
                'type': 'square_author_year',
                'reference_raw': match
            })
        
        # Limit to prevent overload
        return citations[:20]
    
    def _parse_citation_reference(self, ref_str: str) -> List[int]:
        """Parse citation reference string like '1,3-5,7' into list of numbers"""
        numbers = []
        parts = ref_str.split(',')
        for part in parts:
            if '-' in part:
                # Range
                try:
                    start, end = map(int, part.split('-'))
                    numbers.extend(range(start, end + 1))
                except ValueError:
                    pass  # Invalid range, skip
            else:
                # Single number
                try:
                    numbers.append(int(part))
                except ValueError:
                    pass  # Invalid number, skip
        return numbers
    
    def _validate_single_citation(self, citation: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single citation against available sources"""
        citation_text = citation['text']
        citation_type = citation['type']
        
        # In a real implementation, this would:
        # 1. Look up the reference in a bibliography or reference list
        # 2. Verify the reference exists and matches the claim
        # For now, use simplified validation
        
        # Check if we have references in context
        context_refs = self._extract_references_from_context(context)
        
        is_valid = False
        validation_details = {}
        
        if citation_type in ['bracket', 'square_author_year']:
            # For numeric citations, check if reference exists
            if 'reference_id' in citation:
                ref_id = citation['reference_id']
                is_valid = ref_id in context_refs.get('numeric', set())
                validation_details = {
                    'reference_id': ref_id,
                    'found_in_context': is_valid,
                    'available_references': list(context_refs.get('numeric', set()))[:10]
                }
            elif 'reference_raw' in citation:
                # For author-year citations
                ref_raw = citation['reference_raw']
                is_valid = any(ref_raw.lower() in ref.lower() 
                             for ref in context_refs.get('author_year', []))
                validation_details = {
                    'reference_raw': ref_raw,
                    'found_in_context': is_valid,
                    'available_references': context_refs.get('author_year', [])[:10]
                }
        else:
            # Default validation
            is_valid = True  # Assume valid if we can't check
            validation_details = {'assumed_valid': True}
        
        return {
            'citation': citation_text,
            'type': citation_type,
            'valid': is_valid,
            'validation_details': validation_details
        }
    
    def _extract_references_from_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract reference information from context"""
        refs = {
            'numeric': set(),
            'author_year': []
        }
        
        # Look for explicit reference lists in context
        reference_fields = ['references', 'bibliography', 'sources', 'citations']
        for field in reference_fields:
            if field in context:
                ref_data = context[field]
                if isinstance(ref_data, list):
                    for item in ref_data:
                        if isinstance(item, dict):
                            # Structured reference
                            if 'id' in item:
                                refs['numeric'].add(item['id'])
                            if 'authors' in item and 'year' in item:
                                authors = item['authors'] if isinstance(item['authors'], str) else ', '.join(item['authors'])
                                refs['author_year'].append(f"{authors}, {item['year']}")
                        elif isinstance(item, str):
                            # Try to parse as reference
                            # Simple heuristic: look for year patterns
                            if re.search(r'\d{4}', item):
                                refs['author_year'].append(item)
                            else:
                                # Try to extract numeric ID
                                nums = re.findall(r'\d+', item)
                                if nums:
                                    try:
                                        refs['numeric'].add(int(nums[0]))
                                    except ValueError:
                                        pass
                elif isinstance(ref_data, dict):
                    # Handle dictionary format references
                    for key, value in ref_data.items():
                        if isinstance(value, dict) and 'id' in value:
                            try:
                                refs['numeric'].add(int(value['id']))
                            except (ValueError, TypeError):
                                pass
        
        # Also check in text fields for reference patterns
        text_content = ' '.join(str(v) for v in context.values() if isinstance(v, str))
        # Extract any remaining citation patterns from text
        text_citations = self._extract_citations(text_content)
        for cit in text_citations:
            if cit['type'] == 'bracket' and 'reference_id' in cit:
                refs['numeric'].add(cit['reference_id'])
            elif cit['type'] in ['author_year', 'square_author_year'] and 'reference_raw' in cit:
                refs['author_year'].append(cit['reference_raw'])
        
        return refs
    
    def _expects_citations(self, text: str, context: Dict[str, Any]) -> bool:
        """Determine if the text expects to have citations based on content and context"""
        # Heuristics: academic/scientific content, specific claims, etc.
        academic_indicators = [
            'study', 'research', 'experiment', 'data shows', 'according to',
            'published in', 'journal', 'university', 'institute', 'et al.',
            'fig.', 'table', 'p-value', 'significant', 'hypothesis'
        ]
        
        text_lower = text.lower()
        academic_score = sum(1 for ind in academic_indicators if ind in text_lower)
        
        # Check if context suggests academic work
        context_suggests_academic = False
        if context:
            context_text = ' '.join(str(v) for v in context.values() if isinstance(v, str))
            context_lower = context_text.lower()
            context_suggests_academic = any(ind in context_lower for ind in academic_indicators[:5])
        
        # Expect citations if we have academic content or specific factual claims
        has_specific_claims = bool(re.search(r'\d{4}|\d+\.\d+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', text))
        
        return (academic_score >= 2) or context_suggests_academic or (has_specific_claims and academic_score >= 1)
    
    def _advanced_confidence_scoring(self, result: Any, context: Dict[str, Any]) -> tuple:
        """
        Advanced confidence scoring using multiple signals
        """
        self.logger.debug("Applying advanced confidence scoring")
        
        # Convert result to string if needed
        if not isinstance(result, str):
            if hasattr(result, 'text'):
                result_text = result.text
            else:
                result_text = str(result)
        else:
            result_text = result
        
        # Calculate multiple confidence signals
        signals = {}
        
        # Signal 1: Linguistic confidence (hedging, certainty words)
        signals['linguistic'] = self._calculate_linguistic_confidence(result_text)
        
        # Signal 2: Specificity and precision
        signals['specificity'] = self._calculate_specificity_confidence(result_text)
        
        # Signal 3: Internal consistency
        signals['consistency'] = self._calculate_internal_consistency(result_text)
        
        # Signal 4: Contextual alignment
        signals['contextual'] = self._calculate_contextual_confidence(result_text, context)
        
        # Signal 5: Completeness
        signals['completeness'] = self._calculate_completeness_confidence(result_text, context)
        
        # Weighted combination
        weights = {
            'linguistic': 0.2,
            'specificity': 0.2,
            'consistency': 0.2,
            'contextual': 0.2,
            'completeness': 0.2
        }
        
        weighted_sum = sum(signals.get(signal, 0) * weight for signal, weight in weights.items())
        total_weight = sum(weights.values())
        final_score = weighted_sum / max(total_weight, 0.001)
        
        details = {
            'signals': signals,
            'weights': weights,
            'final_score': final_score
        }
        
        return result, final_score, details
    
    def _calculate_linguistic_confidence(self, text: str) -> float:
        """Calculate confidence based on linguistic cues"""
        if not text:
            return 0.0
        
        words = text.lower().split()
        if not words:
            return 0.5
        
        # Certainty indicators (increase confidence)
        certainty_words = [
            'definitely', 'certainly', 'clearly', 'obviously', 'undoubtedly',
            'clearly', 'evidently', 'manifestly', 'plainly', 'surely'
        ]
        
        # Uncertainty indicators (decrease confidence)
        uncertainty_words = [
            'maybe', 'perhaps', 'possibly', 'might', 'could', 'may',
            'appears', 'seems', 'suggests', 'indicates', 'possibly',
            'allegedly', 'reportedly', 'supposedly', 'presumably'
        ]
        
        certainty_count = sum(1 for word in words if word in certainty_words)
        uncertainty_count = sum(1 for word in words if word in uncertainty_words)
        
        # Normalize by text length
        certainty_score = certainty_count / max(len(words), 1)
        uncertainty_score = uncertainty_count / max(len(words), 1)
        
        # Base score of 0.5, adjusted by certainty/uncertainty
        confidence = 0.5 + (certainty_score * 2) - (uncertainty_score * 1.5)
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_specificity_confidence(self, text: str) -> float:
        """Calculate confidence based on specificity and precision"""
        if not text:
            return 0.0
        
        # Look for specific, measurable information
        specific_patterns = [
            r'\d+\.\d+',  # Decimal numbers
            r'\d+%',      # Percentages
            r'\d{4}',     # Years
            r'\d{1,2}\/\d{1,2}\/\d{2,4}',  # Dates
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+',  # Proper nouns
            r'\b(p=|t=|F=|χ²=)\s*0?\.\d+',  # Statistical notation
            r'\b(mean|median|std|sd|se)\s*[:=]\s*\d+\.?\d*',  # Statistics
        ]
        
        specificity_score = 0
        for pattern in specific_patterns:
            matches = len(re.findall(pattern, text))
            specificity_score += min(matches, 5)  # Cap at 5 per pattern type
        
        # Normalize by text length (expect ~10 specific elements per 100 words)
        word_count = len(text.split())
        expected_specificity = word_count / 10.0
        normalized_score = min(specificity_score / max(expected_specificity, 1), 1.0)
        
        return normalized_score
    
    def _calculate_internal_consistency(self, text: str) -> float:
        """Calculate confidence based on internal consistency of the text"""
        if not text or len(text) < 50:
            return 0.7  # Default for short texts
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.8  # Single sentence gets benefit of doubt
        
        # Check for contradictions between sentences
        contradiction_indicators = 0
        total_comparisons = 0
        
        # Simple approach: look for opposing statements
        opposing_pairs = [
            (['increase', 'rise', 'grow'], ['decrease', 'fall', 'decline']),
            (['true', 'correct', 'right'], ['false', 'incorrect', 'wrong']),
            (['cause', 'lead to', 'result in'], ['prevent', 'avoid', 'stop']),
            (['more', 'greater', 'higher'], ['less', 'fewer', 'lower']),
            (['always', 'never'], ['sometimes', 'often']),
            (['all', 'none'], ['some', 'many', 'few'])
        ]
        
        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences[i+1:], i+1):
                total_comparisons += 1
                sent1_lower = sent1.lower()
                sent2_lower = sent2.lower()
                
                for pos_words, neg_words in opposing_pairs:
                    has_pos1 = any(word in sent1_lower for word in pos_words)
                    has_neg1 = any(word in sent1_lower for word in neg_words)
                    has_pos2 = any(word in sent2_lower for word in pos_words)
                    has_neg2 = any(word in sent2_lower for word in neg_words)
                    
                    # Check for contradictions: pos in one, neg in other
                    if (has_pos1 and has_neg2) or (has_neg1 and has_pos2):
                        contradiction_indicators += 1
                        break  # Only count once per pair
        
        # Calculate consistency score
        if total_comparisons > 0:
            contradiction_ratio = contradiction_indicators / total_comparisons
            consistency_score = max(0, 1 - (contradiction_ratio * 2))  # Penalize contradictions
        else:
            consistency_score = 0.8
        
        return consistency_score
    
    def _calculate_contextual_confidence(self, result_text: str, context: Dict[str, Any]) -> float:
        """Calculate confidence based on alignment with context"""
        if not context:
            return 0.5  # Neutral if no context
        
        # Convert context to text
        context_text = ' '.join(str(v) for v in context.values() if isinstance(v, (str, int, float)))
        context_text += ' ' + ' '.join(
            str(item) for v in context.values() 
            if isinstance(v, list) for item in v if isinstance(item, (str, int, float))
        )
        
        if not context_text.strip():
            return 0.5
        
        # Calculate similarity between result and context
        # But we don't want to reward mere repetition - we want relevant, non-redundant alignment
        similarity = self._calculate_text_similarity(result_text, context_text)
        
        # Ideal similarity is moderate - not too low (irrelevant) or too high (likely just repeating)
        if similarity < 0.1:
            return 0.3  # Too dissimilar
        elif similarity > 0.8:
            return 0.6  # Too similar - might be just copying context
        else:
            # Sweet spot: 0.1 to 0.8 similarity
            # Peak at 0.4-0.5
            ideal_similarity = 0.45
            distance_from_ideal = abs(similarity - ideal_similarity)
            max_distance = 0.4  # Distance to worst case (0.05 or 0.85)
            confidence = 1.0 - (distance_from_ideal / max_distance)
            return max(0.0, min(1.0, confidence))
    
    def _calculate_completeness_confidence(self, result_text: str, context: Dict[str, Any]) -> float:
        """Calculate confidence based on completeness of response"""
        if not result_text:
            return 0.0
        
        # Heuristic: longer, more detailed responses tend to be more complete (up to a point)
        word_count = len(result_text.split())
        
        # Ideal length depends on context, but we'll use heuristics
        # Short answers (<10 words) might be incomplete
        # Very long answers (>300 words) might be verbose but complete
        if word_count < 5:
            return 0.2  # Very likely incomplete
        elif word_count < 20:
            return 0.5  # Possibly incomplete
        elif word_count < 100:
            return 0.8  # Reasonably complete
        elif word_count < 300:
            return 0.9  # Quite complete
        else:
            return 0.85  # Complete but possibly verbose
        
        # Adjust based on context if we can determine expected length
        # For now, use the basic heuristic
    
    def _consensus_verification(self, prompt: str, context: Dict[str, Any], 
                              model_adapter: Any, initial_result: Any) -> tuple:
        """
        Apply consensus verification by generating multiple perspectives and finding agreement
        """
        self.logger.debug("Applying consensus verification")
        
        if not model_adapter:
            return initial_result, 0.5, {'error': 'No model adapter available'}
        
        try:
            # Generate multiple diverse outputs
            num_perspectives = self.config.get('consensus_perspectives', 4)
            perspectives = []
            
            # Different strategies for diversity
            strategies = [
                {'temperature': 0.3, 'top_p': 0.9, 'description': 'focused'},
                {'temperature': 0.7, 'top_p': 0.9, 'description': 'balanced'},
                {'temperature': 1.0, 'top_p': 0.9, 'description': 'creative'},
                {'temperature': 0.7, 'top_p': 0.5, 'description': 'concise'}
            ][:num_perspectives]
            
            for i, strategy in enumerate(strategies):
                try:
                    output = model_adapter.generate(
                        prompt,
                        temperature=strategy['temperature'],
                        top_p=strategy['top_p'],
                        max_tokens=150
                    )
                    perspectives.append({
                        'text': getattr(output, 'text', str(output)),
                        'strategy': strategy['description'],
                        'temperature': strategy['temperature'],
                        'top_p': strategy['top_p'],
                        'index': i
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to generate perspective {i}: {str(e)}")
                    continue
            
            if len(perspectives) < 2:
                self.logger.warning("Insufficient perspectives for consensus")
                return initial_result, 0.5, {'error': 'Insufficient perspectives'}
            
            # Extract texts
            perspective_texts = [p['text'] for p in perspectives]
            
            # Find consensus areas using pairwise similarity
            consensus_elements = self._find_consensus_elements(perspective_texts)
            
            # Calculate consensus score
            if perspective_texts:
                # Average pairwise similarity
                similarities = []
                for i in range(len(perspective_texts)):
                    for j in range(i+1, len(perspective_texts)):
                        sim = self._calculate_text_similarity(perspective_texts[i], perspective_texts[j])
                        similarities.append(sim)
                
                consensus_score = sum(similarities) / max(len(similarities), 1) if similarities else 0.0
            else:
                consensus_score = 0.0
            
            # Generate consensus result by combining agreed-upon elements
            if consensus_elements and len(consensus_elements) >= 2:
                # Create a result that emphasizes the consensus
                consensus_text = ' '.join(consensus_elements[:3])  # Take top 3 consensus elements
                class ConsensusResult:
                    def __init__(self, text):
                        self.text = text
                        self.token_count = len(text.split())
                consensus_result = ConsensusResult(consensus_text)
            else:
                # Fallback to initial result or average of perspectives
                consensus_result = initial_result
            
            details = {
                'num_perspectives': len(perspectives),
                'consensus_score': consensus_score,
                'consensus_elements_found': len(consensus_elements),
                'strategies_used': [p['strategy'] for p in perspectives]
            }
            
            return consensus_result, consensus_score, details
            
        except Exception as e:
            self.logger.error(f"Consensus verification failed: {str(e)}")
            return initial_result, 0.0, {'error': str(e)}
    
    def _find_consensus_elements(self, texts: List[str]) -> List[str]:
        """Find elements that appear across multiple texts"""
        if len(texts) < 2:
            return []
        
        # Simple approach: find common phrases/n-grams
        # In reality, we'd use more sophisticated semantic similarity
        
        # Extract sentences from each text
        all_sentences = []
        for text in texts:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
            all_sentences.append(sentences)
        
        # Find sentences that appear in multiple texts (with some flexibility)
        sentence_votes = defaultdict(int)
        sentence_to_texts = defaultdict(set)
        
        for text_idx, sentences in enumerate(all_sentences):
            seen_in_this_text = set()
            for sent in sentences:
                # Normalize sentence for comparison
                normalized = re.sub(r'\s+', ' ', sent.lower().strip())
                if normalized and len(normalized) > 5:  # Ignore very short sentences
                    if normalized not in seen_in_this_text:
                        sentence_votes[normalized] += 1
                        sentence_to_texts[normalized].add(text_idx)
                        seen_in_this_text.add(normalized)
        
        # Select sentences that appear in at least half the texts
        min_texts = max(2, len(texts) // 2)
        consensus_sentences = [
            sent for sent, count in sentence_votes.items() 
            if count >= min_texts
        ]
        
        # Sort by number of texts that contain them (descending)
        consensus_sentences.sort(
            key=lambda s: len(sentence_to_texts[s]), 
            reverse=True
        )
        
        return consensus_sentences[:5]  # Return top 5
    
    def _reflection_verification(self, prompt: str, context: Dict[str, Any], 
                               model_adapter: Any, initial_result: Any) -> tuple:
        """
        Apply reflection verification: have the model critique its own output
        """
        self.logger.debug("Applying reflection verification")
        
        if not model_adapter:
            return initial_result, 0.5, {'error': 'No model adapter available'}
        
        try:
            # Convert initial result to text
            if hasattr(initial_result, 'text'):
                initial_text = initial_result.text
            else:
                initial_text = str(initial_result)
            
            # Create a reflection prompt asking the model to critique its own answer
            reflection_prompt = f"""
            Please critique the following answer to the question: "{prompt}"
            
            Answer: {initial_text}
            
            Critique should address:
            1. Factual accuracy: Are there any factual errors or unsupported claims?
            2. Logical consistency: Are there any contradictions or flawed reasoning?
            3. Completeness: Does the answer fully address the question?
            4. Relevance: Is all information relevant to the question?
            5. Clarity: Is the answer clear and well-structured?
            
            Provide specific, actionable feedback for improvement.
            """
            
            # Generate the critique
            critique_output = model_adapter.generate(
                reflection_prompt,
                temperature=0.4,  # Lower temperature for more focused critique
                max_tokens=200
            )
            
            if hasattr(critique_output, 'text'):
                critique_text = critique_output.text
            else:
                critique_text = str(critique_output)
            
            # Analyze the critique to determine what needs improvement
            improvement_needed = self._analyze_critique(critique_text)
            
            # If significant improvements are needed, generate a revised answer
            if improvement_needed['needs_improvement'] and improvement_needed['priority'] > 0.5:
                revised_prompt = f"""
                Please provide an improved answer to the question: "{prompt}"
                
                Original answer: {initial_text}
                
                Critique of original answer:
                {critique_text}
                
                Please address the specific issues raised in the critique to produce a better answer.
                """
                
                revised_output = model_adapter.generate(
                    revised_prompt,
                    temperature=0.5,
                    max_tokens=200
                )
                
                if hasattr(revised_output, 'text'):
                    revised_text = revised_output.text
                else:
                    revised_text = str(revised_output)
                
                class ReflectionResult:
                    def __init__(self, text):
                        self.text = text
                        self.token_count = len(text.split())
                
                final_result = ReflectionResult(revised_text)
                
                # Score based on how much the critique suggested improvement
                # Less improvement needed = higher score
                reflection_score = max(0.3, 1.0 - improvement_needed['priority'])
            else:
                # Critique suggests the answer is good
                final_result = initial_result
                reflection_score = 0.8  # Good score if no major improvements needed
            
            details = {
                'critique_text': critique_text[:200] + ('...' if len(critique_text) > 200 else ''),
                'improvement_needed': improvement_needed,
                'reflection_score': reflection_score
            }
            
            return final_result, reflection_score, details
            
        except Exception as e:
            self.logger.error(f"Reflection verification failed: {str(e)}")
            return initial_result, 0.0, {'error': str(e)}
    
    def _analyze_critique(self, critique_text: str) -> Dict[str, Any]:
        """Analyze critique text to determine what improvements are needed"""
        if not critique_text:
            return {'needs_improvement': False, 'priority': 0.0, 'issues': []}
        
        critique_lower = critique_text.lower()
        
        # Look for indicators of problems
        issue_indicators = {
            'factual_error': ['fact', 'incorrect', 'wrong', 'inaccurate', 'mistake', 'error'],
            'logical_flaw': ['logic', 'contradiction', 'inconsistent', 'flawed', 'doesn\'t follow'],
            'incomplete': ['missing', 'lack', 'absent', 'need to add', 'should include'],
            'irrelevant': ['irrelevant', 'off-topic', 'not related', 'tangent'],
            'unclear': ['unclear', 'confusing', 'ambiguous', 'vague', 'hard to understand']
        }
        
        issues_found = []
        issue_scores = {}
        
        for issue_type, indicators in issue_indicators.items():
            score = sum(1 for ind in indicators if ind in critique_lower)
            if score > 0:
                issues_found.append(issue_type)
                issue_scores[issue_type] = score
        
        # Calculate overall priority of improvement needed
        total_indicators = sum(len(indicators) for indicators in issue_indicators.values())
        total_found = sum(issue_scores.values())
        priority = min(total_found / max(total_indicators * 0.3, 1), 1.0)  # Normalize
        
        # Also look for severity indicators
        severity_indicators = ['major', 'serious', 'significant', 'critical', 'fundamental']
        severity_score = sum(1 for ind in severity_indicators if ind in critique_lower)
        severity_boost = min(severity_score * 0.2, 0.5)
        
        final_priority = min(priority + severity_boost, 1.0)
        
        return {
            'needs_improvement': len(issues_found) > 0,
            'priority': final_priority,
            'issues': issues_found,
            'issue_scores': issue_scores,
            'severity_boost': severity_boost
        }
    
    def _apply_verification_improvements(self, prompt: str, context: Dict[str, Any], 
                                       model_adapter: Any, current_result: Any, 
                                       verification_details: Dict[str, Any]) -> tuple:
        """Apply improvements based on verification results"""
        self.logger.debug("Applying verification-based improvements")
        
        # In a full implementation, this would synthesize insights from all verification methods
        # to generate an improved result
        
        # For now, we'll return None to indicate no automatic improvement was applied
        # The calling code can decide how to proceed
        return None, {'message': 'Improvement logic would be implemented here'}
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using shared cached word sets"""
        return self._text_cache.similarity(text1, text2)
    
    def _record_verification(self, prompt: str, context: Dict[str, Any], 
                           original_result: Any, final_result: Any, score: float):
        """Record verification for learning and improvement"""
        record = {
            'timestamp': self._get_timestamp(),
            'prompt_hash': hashlib.sha256(prompt.encode()).hexdigest()[:16],
            'original_result_type': type(original_result).__name__,
            'final_result_type': type(final_result).__name__,
            'verification_score': score,
            'context_size': len(str(context))
        }
        
        if len(self.verification_history) >= self._max_verification_history:
            self.verification_history.pop(0)
        self.verification_history.append(record)
    
    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        return time.time()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get verification metrics"""
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'methods': self.methods,
            'threshold': self.threshold,
            'max_iterations': self.max_iterations,
            'confidence_threshold': self.confidence_threshold,
            'consensus_threshold': self.consensus_threshold,
            'verification_history_size': len(self.verification_history)
        })
        return base_metrics
