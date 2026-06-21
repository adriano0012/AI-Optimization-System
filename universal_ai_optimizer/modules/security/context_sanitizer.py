"""Context Sanitizer - Removes malicious HTML, scripts, and corrupted unicode"""
import re
import html
import logging
import unicodedata
import urllib.parse
from typing import Dict, Any, List, Optional

from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)


class ContextSanitizer(BaseOptimizerModule):
    """Sanitizes context data to prevent injection and corruption"""

    DANGEROUS_TAGS = {
        'script', 'iframe', 'object', 'embed', 'applet', 'form', 'input',
        'textarea', 'select', 'button', 'link', 'meta', 'style', 'base',
        'frame', 'frameset', 'noframes', 'noscript', 'html', 'head', 'body',
        'title', 'svg', 'math', 'xml', 'xmp', 'plaintext', 'listing'
    }

    DANGEROUS_ATTRIBUTES = {
        'onload', 'onerror', 'onclick', 'onmouseover', 'onmouseout',
        'onkeydown', 'onkeyup', 'onkeypress', 'onfocus', 'onblur',
        'onchange', 'onsubmit', 'onreset', 'onselect', 'onunload',
        'onabort', 'onresize', 'onscroll', 'oncontextmenu', 'ondblclick',
        'onmousedown', 'onmousemove', 'onmouseup', 'ondrag', 'ondrop',
        'href', 'src', 'action', 'background', 'dynsrc', 'lowsrc',
        'codebase', 'archive', 'data', 'formaction', 'formaction'
    }

    JAVASCRIPT_PROTOCOL = re.compile(r'javascript\s*:', re.IGNORECASE)
    VBSCRIPT_PROTOCOL = re.compile(r'vbscript\s*:', re.IGNORECASE)
    DATA_PROTOCOL = re.compile(r'data\s*:', re.IGNORECASE)
    EXPRESSION = re.compile(r'expression\s*\(', re.IGNORECASE)
    BEHAVIOR = re.compile(r'behavior\s*:', re.IGNORECASE)
    MOZ_BINDING = re.compile(r'-moz-binding\s*:', re.IGNORECASE)
    # Obfuscation-resistant patterns: null bytes, HTML entities, unicode homoglyphs
    JS_OBFUSCATED = re.compile(
        r'(?:java\s*\x00\s*script|'
        r'&#(?:106|74|110|75|97|65|118|86|97|65|115|83|99|67|114|82|105|73|112|80|116|84)\s*;?\s*:|'
        r'j\s*a\s*v\s*a\s*s\s*c\s*r\s*i\s*p\s*t\s*:)',
        re.IGNORECASE
    )

    UNICODE_CONTROL_CHARS = re.compile(
        r'[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]'
        r'|[\u200B-\u200F\u202A-\u202E\u2060-\u206F]'
        r'|[\uFEFF\uFFF0-\uFFFF]'
    )

    # Regex patterns to detect different Unicode script ranges within a single word
    SCRIPT_LATIN = re.compile(r'[a-zA-Z]')
    SCRIPT_CYRILLIC = re.compile(r'[\u0400-\u04FF]')
    SCRIPT_GREEK = re.compile(r'[\u0370-\u03FF\u1F00-\u1FFF]')
    SCRIPT_ARABIC = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
    SCRIPT_CJK = re.compile(
        r'[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF'
        r'\u20000-\u2A6DF\u2A700-\u2B73F\u2B740-\u2B81F\u2B820-\u2CEAF]'
    )
    SCRIPT_DEVANAGARI = re.compile(r'[\u0900-\u097F]')
    SCRIPT_THAI = re.compile(r'[\u0E00-\u0E7F]')

    TAG_PATTERN = re.compile(r'<(/?)\s*([a-zA-Z0-9:-]+)([^>]*)>', re.IGNORECASE)
    # Match HTML comments <!-- ... -->
    COMMENT_PATTERN = re.compile(r'<!--[\s\S]*?-->', re.IGNORECASE)
    # Match CDATA sections
    CDATA_PATTERN = re.compile(r'<!\[CDATA\[[\s\S]*?\]\]>', re.IGNORECASE)
    # Match processing instructions <? ... ?>
    PI_PATTERN = re.compile(r'<\?[\s\S]*?\?>', re.IGNORECASE)
    ATTRIBUTE_PATTERN = re.compile(
        r'([a-zA-Z_:][a-zA-Z0-9_:.-]*)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))',
        re.IGNORECASE
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None, allowed_tags: List[str] = None, allowed_attributes: List[str] = None):
        super().__init__(config)
        self.allowed_tags = set(allowed_tags) if allowed_tags else set()
        self.allowed_attributes = set(allowed_attributes) if allowed_attributes else {
            'id', 'class', 'title', 'alt', 'width', 'height',
            'align', 'valign', 'colspan', 'rowspan', 'border', 'cellpadding',
            'cellspacing', 'target', 'rel'
        }

    def process(self, prompt: str, context: Dict[str, Any],
                model_adapter: Optional[Any] = None,
                pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input by sanitizing the prompt and context.
        Returns sanitized prompt and context.
        """
        self._log_processing(len(prompt), len(context) if context else 0)
        sanitized_prompt = self.sanitize(prompt)
        sanitized_context = self.sanitize(context) if context else {}
        return {
            'prompt': sanitized_prompt,
            'context': sanitized_context,
            'module': self.__class__.__name__
        }

    def sanitize(self, context: Any) -> Any:
        """
        Sanitize context data recursively.
        Handles strings, dicts, lists, and primitive types.
        """
        if isinstance(context, str):
            return self._sanitize_string(context)
        elif isinstance(context, dict):
            return {k: self.sanitize(v) for k, v in context.items()}
        elif isinstance(context, list):
            return [self.sanitize(item) for item in context]
        elif isinstance(context, tuple):
            return tuple(self.sanitize(item) for item in context)
        elif isinstance(context, set):
            return {self.sanitize(item) for item in context}
        else:
            return context

    def _sanitize_string(self, text: str) -> str:
        """Sanitize a single string value"""
        if not text:
            return text

        text = self._remove_control_chars(text)
        text = self._strip_dangerous_html(text)
        text = self._normalize_unicode(text)
        text = self._escape_remaining_html(text)

        return text

    def _remove_control_chars(self, text: str) -> str:
        """Remove unicode control characters and BOMs"""
        return self.UNICODE_CONTROL_CHARS.sub('', text)

    def _strip_dangerous_html(self, text: str) -> str:
        """Remove dangerous HTML tags and attributes"""
        # Remove HTML comments, CDATA, and processing instructions first
        text = self.COMMENT_PATTERN.sub('', text)
        text = self.CDATA_PATTERN.sub('', text)
        text = self.PI_PATTERN.sub('', text)

        def replace_tag(match):
            closing = match.group(1)
            tag_name = match.group(2).lower()
            attrs = match.group(3)

            if tag_name in self.DANGEROUS_TAGS:
                return ''

            if tag_name not in self.allowed_tags and not closing:
                return ''

            if closing:
                return f'</{tag_name}>'

            clean_attrs = self._sanitize_attributes(attrs)
            return f'<{tag_name}{clean_attrs}>'

        text = self.TAG_PATTERN.sub(replace_tag, text)

        text = self.JAVASCRIPT_PROTOCOL.sub('blocked:', text)
        text = self.JS_OBFUSCATED.sub('blocked:', text)
        text = self.VBSCRIPT_PROTOCOL.sub('blocked:', text)
        text = self.DATA_PROTOCOL.sub('blocked:', text)
        text = self.EXPRESSION.sub('blocked(', text)
        text = self.BEHAVIOR.sub('blocked:', text)
        text = self.MOZ_BINDING.sub('blocked:', text)

        return text

    def _normalize_for_security(self, text: str) -> str:
        """Normalize text for security checks: decode URL encoding, remove null bytes"""
        text = urllib.parse.unquote(text)
        text = text.replace('\x00', '')
        return text

    def _sanitize_attributes(self, attrs: str) -> str:
        """Sanitize tag attributes"""
        def replace_attr(match):
            attr_name = match.group(1).lower()
            attr_value = match.group(2) or match.group(3) or match.group(4) or ''

            if attr_name in self.DANGEROUS_ATTRIBUTES:
                return ''

            if attr_name not in self.allowed_attributes:
                return ''

            attr_value = self._normalize_for_security(attr_value)
            attr_value = self.JAVASCRIPT_PROTOCOL.sub('blocked:', attr_value)
            attr_value = self.JS_OBFUSCATED.sub('blocked:', attr_value)
            attr_value = self.VBSCRIPT_PROTOCOL.sub('blocked:', attr_value)
            attr_value = self.DATA_PROTOCOL.sub('blocked:', attr_value)
            attr_value = self.EXPRESSION.sub('blocked(', attr_value)

            quote = '"' if match.group(2) is not None else "'" if match.group(3) is not None else ''
            return f'{attr_name}={quote}{attr_value}{quote}'

        return self.ATTRIBUTE_PATTERN.sub(replace_attr, attrs)

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode to prevent homograph attacks"""
        normalized = unicodedata.normalize('NFKC', text)
        self._detect_mixed_scripts(normalized)
        return normalized

    def _get_char_scripts(self, char: str) -> set:
        """Return the set of script categories a character belongs to."""
        scripts = set()
        if self.SCRIPT_LATIN.match(char):
            scripts.add('Latin')
        if self.SCRIPT_CYRILLIC.match(char):
            scripts.add('Cyrillic')
        if self.SCRIPT_GREEK.match(char):
            scripts.add('Greek')
        if self.SCRIPT_ARABIC.match(char):
            scripts.add('Arabic')
        if self.SCRIPT_CJK.match(char):
            scripts.add('CJK')
        if self.SCRIPT_DEVANAGARI.match(char):
            scripts.add('Devanagari')
        if self.SCRIPT_THAI.match(char):
            scripts.add('Thai')
        return scripts

    def _detect_mixed_scripts(self, text: str) -> None:
        """Detect mixed-script usage within words and log a warning.

        Splits text into words (alphanumeric sequences) and checks if any
        single word contains characters from more than one distinct script.
        This helps identify potential homograph attacks where visually
        similar characters from different scripts are combined.
        """
        words = re.findall(r'[\w]+', text)
        for word in words:
            combined_scripts = set()
            for char in word:
                if char.isalpha():
                    combined_scripts.update(self._get_char_scripts(char))

            if len(combined_scripts) > 1:
                logger.warning(
                    "HOMOGRAPH ATTACK DETECTED: Mixed scripts %s found in word '%s'. "
                    "This may be an attempt to bypass normalization filters.",
                    combined_scripts, word
                )

    def _escape_remaining_html(self, text: str) -> str:
        """Escape any remaining HTML special characters (no double-escape)"""
        return html.escape(text, quote=True)

    def sanitize_dict(self, data: Dict[str, Any], max_depth: int = 10, current_depth: int = 0) -> Dict[str, Any]:
        """Sanitize dictionary with depth limiting"""
        if current_depth >= max_depth:
            return {}

        result = {}
        for key, value in data.items():
            clean_key = self._sanitize_string(str(key)) if isinstance(key, str) else key
            result[clean_key] = self.sanitize(value) if current_depth < max_depth else value
        return result