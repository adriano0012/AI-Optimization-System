"""PII Filter - Regex-based sensitive data masking"""
import re
import logging
from typing import Dict, List, Tuple, Optional, Any

from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)


class PIIFilter(BaseOptimizerModule):
    """Filters and masks personally identifiable information"""

    CPF_PATTERN = re.compile(
        r'\b(?:\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11})\b'
    )

    CNPJ_PATTERN = re.compile(
        r'\b(?:\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}|\d{14})\b'
    )

    EMAIL_PATTERN = re.compile(
        r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    )

    CREDIT_CARD_PATTERN = re.compile(
        r'\b(?:'
        r'4[0-9]{12}(?:[0-9]{3})?|'                    # Visa
        r'5[1-5][0-9]{14}|'                            # Mastercard
        r'3[47][0-9]{13}|'                             # Amex
        r'3(?:0[0-5]|[68][0-9])[0-9]{11}|'             # Diners Club
        r'6(?:011|5[0-9]{2})[0-9]{12}|'                # Discover
        r'(?:2131|1800|35\d{3})\d{11}'                 # JCB
        r')\b'
    )

    CREDIT_CARD_WITH_SEPARATORS = re.compile(
        r'\b(?:'
        r'(?:4[0-9]{3}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4})|'
        r'(?:5[1-5][0-9]{2}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4})|'
        r'(?:3[47][0-9]{2}[-\s]?[0-9]{6}[-\s]?[0-9]{5})|'
        r'(?:6(?:011|5[0-9]{2})[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4})'
        r')\b'
    )

    PASSPORT_PATTERNS = {
        'BR': re.compile(r'\b[A-Z]{2}\d{6}\b'),           # Brazil
        'US': re.compile(r'\b\d{9}\b'),                    # USA
        'EU': re.compile(r'\b[A-Z]{1,2}\d{6,8}\b'),       # EU generic
        'generic': re.compile(r'\b[A-Z]{1,3}\d{6,9}\b'),   # Generic
    }

    PHONE_PATTERN = re.compile(
        r'(?:\+?55\s*)?'                                  # Brazil country code
        r'(?:\(?\d{2}\)?\s*)?'                            # Area code
        r'(?:9\s*)?'                                      # Mobile prefix
        r'\d{4}[-\s]?\d{4}'                               # Number
    )

    RG_PATTERN = re.compile(
        r'\b\d{1,2}\.\d{3}\.\d{3}[-\s]?\d{1,2}\b'        # Brazilian RG
    )

    CNH_PATTERN = re.compile(
        r'\b\d{11}\b'                                     # Brazilian CNH (11 digits)
    )

    IPV4_PATTERN = re.compile(
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    )

    IPV6_PATTERN = re.compile(
        r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'
    )

    REDACTION = "[REDACTED_PII]"

    def __init__(self, config: Optional[Dict[str, Any]] = None, custom_patterns: Dict[str, re.Pattern] = None):
        super().__init__(config)
        self.custom_patterns = custom_patterns or {}
        self._compiled_patterns = self._build_patterns()

    def _build_patterns(self) -> List[Tuple[str, re.Pattern]]:
        """Build ordered list of (name, pattern) tuples"""
        return [
            ('cpf', self.CPF_PATTERN),
            ('cnpj', self.CNPJ_PATTERN),
            ('email', self.EMAIL_PATTERN),
            ('credit_card', self.CREDIT_CARD_PATTERN),
            ('credit_card_sep', self.CREDIT_CARD_WITH_SEPARATORS),
            ('passport_br', self.PASSPORT_PATTERNS['BR']),
            ('passport_us', self.PASSPORT_PATTERNS['US']),
            ('passport_eu', self.PASSPORT_PATTERNS['EU']),
            ('passport_generic', self.PASSPORT_PATTERNS['generic']),
            ('phone', self.PHONE_PATTERN),
            ('rg', self.RG_PATTERN),
            ('cnh', self.CNH_PATTERN),
            ('ipv4', self.IPV4_PATTERN),
            ('ipv6', self.IPV6_PATTERN),
        ] + [(name, pattern) for name, pattern in self.custom_patterns.items()]

    def process(self, prompt: str, context: Dict[str, Any],
                model_adapter: Optional[Any] = None,
                pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input by filtering PII from the prompt and context.
        Returns filtered prompt, filtered context, and metadata.
        """
        self._log_processing(len(prompt), len(context) if context else 0)
        filtered_prompt, prompt_matches = self.filter_with_details(prompt)
        filtered_context = self.filter_dict(context) if context else {}
        context_matches = []
        if context:
            for key, value in context.items():
                if isinstance(value, str):
                    _, matches = self.filter_with_details(value)
                    for m in matches:
                        m['context_key'] = key
                    context_matches.extend(matches)
        all_matches = prompt_matches + context_matches
        return {
            'prompt': filtered_prompt,
            'context': filtered_context,
            'pii_matches': all_matches,
            'pii_count': len(all_matches),
            'module': self.__class__.__name__
        }

    def filter(self, text: str) -> str:
        """
        Filter PII from text, replacing with [REDACTED_PII].
        Validates CPF/CNPJ with check digits before redacting.
        Uses priority-based match selection to resolve overlapping patterns.
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ""

        # Collect all matches with validation and priority scores
        candidates = []
        priority_map = {'cpf': 4, 'cnpj': 4, 'credit_card': 3, 'credit_card_sep': 3,
                        'cnh': 2, 'phone': 1, 'rg': 1}
        for name, pattern in self._compiled_patterns:
            for match in pattern.finditer(text):
                start, end = match.span()
                matched = match.group()
                if name == 'cpf' and not self.validate_cpf(matched):
                    continue
                if name == 'cnpj' and not self.validate_cnpj(matched):
                    continue
                if name == 'cnh' and not self.validate_cnh(matched):
                    continue
                if name == 'credit_card' and not self.validate_credit_card(matched):
                    continue
                if name == 'credit_card_sep' and not self.validate_credit_card(matched):
                    continue
                priority = priority_map.get(name, 0)
                candidates.append((start, end, priority, name))

        # Sort by start position, then by priority (higher wins), then by length
        candidates.sort(key=lambda x: (x[0], -x[2], -(x[1] - x[0])))

        # Select non-overlapping matches (higher priority wins conflicts)
        selected = []
        for start, end, priority, name in candidates:
            overlap = False
            for s, e, _, _ in selected:
                if start < e and end > s:
                    overlap = True
                    break
            if not overlap:
                selected.append((start, end, priority, name))

        # Apply redactions in reverse position order to preserve indices
        selected.sort(key=lambda x: -x[0])
        result = text
        for start, end, priority, name in selected:
            result = result[:start] + self.REDACTION + result[end:]

        return result

    def filter_with_details(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Filter PII and return details about what was redacted.
        Returns (filtered_text, list_of_matches)
        """
        if not isinstance(text, str):
            return str(text) if text is not None else "", []

        matches = []
        result = text

        for name, pattern in self._compiled_patterns:
            for match in pattern.finditer(text):
                start, end = match.span()
                matched_text = match.group()
                # Apply same validation as filter()
                if name == 'cpf' and not self.validate_cpf(matched_text):
                    continue
                if name == 'cnpj' and not self.validate_cnpj(matched_text):
                    continue
                if name == 'cnh' and not self.validate_cnh(matched_text):
                    continue
                if name in ('credit_card', 'credit_card_sep') and not self.validate_credit_card(matched_text):
                    continue
                matches.append({
                    'type': name,
                    'original': matched_text,
                    'start': start,
                    'end': end,
                    'redacted': self.REDACTION
                })

        matches.sort(key=lambda x: x['start'], reverse=True)

        for match in matches:
            start = match['start']
            end = match['end']
            result = result[:start] + self.REDACTION + result[end:]

        matches.sort(key=lambda x: x['start'])
        return result, matches

    def filter_dict(self, data: Dict, max_depth: int = 10, current_depth: int = 0) -> Dict:
        """Recursively filter PII from dictionary values"""
        if current_depth >= max_depth:
            return data

        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.filter(value)
            elif isinstance(value, dict):
                result[key] = self.filter_dict(value, max_depth, current_depth + 1)
            elif isinstance(value, list):
                result[key] = [
                    self.filter(item) if isinstance(item, str) else
                    self.filter_dict(item, max_depth, current_depth + 1) if isinstance(item, dict) else
                    item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def validate_cpf(self, cpf: str) -> bool:
        """Validate Brazilian CPF with check digits"""
        cpf = re.sub(r'\D', '', cpf)
        if len(cpf) != 11 or len(set(cpf)) == 1:
            return False

        for i in range(9, 11):
            total = sum(int(cpf[j]) * ((i + 1) - j) for j in range(i))
            digit = ((total * 10) % 11) % 10
            if digit != int(cpf[i]):
                return False
        return True

    def validate_cnpj(self, cnpj: str) -> bool:
        """Validate Brazilian CNPJ with check digits"""
        cnpj = re.sub(r'\D', '', cnpj)
        if len(cnpj) != 14 or len(set(cnpj)) == 1:
            return False

        weights1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        weights2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]

        for i, weights in enumerate([weights1, weights2], 1):
            total = sum(int(cnpj[j]) * weights[j] for j in range(len(weights)))
            digit = (total % 11)
            digit = 0 if digit < 2 else 11 - digit
            if digit != int(cnpj[11 + i]):
                return False
        return True

    def validate_cnh(self, cnh: str) -> bool:
        """Validate Brazilian CNH (driver's license) with check digits"""
        cnh = re.sub(r'\D', '', cnh)
        if len(cnh) != 11 or len(set(cnh)) == 1:
            return False

        weights1 = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        weights2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        for i, weights in enumerate([weights1, weights2]):
            total = sum(int(cnh[j]) * weights[j] for j in range(9))
            remainder = total % 11
            digit = 0 if remainder >= 10 else remainder
            if digit != int(cnh[9 + i]):
                return False
        return True

    def validate_credit_card(self, number: str) -> bool:
        """Validate credit card using Luhn algorithm"""
        digits = re.sub(r'\D', '', number)
        if len(digits) < 13 or len(digits) > 19:
            return False

        total = 0
        for i, d in enumerate(reversed(digits)):
            n = int(d)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return total % 10 == 0