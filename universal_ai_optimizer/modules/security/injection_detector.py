"""Injection Detector - Heuristic-based prompt injection detection"""
import re
import logging
from typing import List, Tuple, Dict, Optional, Any

from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)

class InjectionDetector(BaseOptimizerModule):
    """Detects prompt injection attempts using heuristic analysis with scoring"""

    INJECTION_PATTERNS = [
        # --- English patterns ---
        (r"(?i)ignore\s+(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|commands?)", 3),
        (r"(?i)disregard\s+(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|commands?)", 3),
        (r"(?i)forget\s+(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|commands?)", 3),
        (r"(?i)system\s*(?:override|prompt|instruction)", 2),
        (r"(?i)you\s+are\s+(?:now|henceforth)\s+(?:a|an)\s+\w+", 1),
        (r"(?i)new\s+(?:system\s+)?(?:prompt|instruction|role)", 2),
        (r"(?i)simulate\s+(?:a|an)\s+\w+", 1),
        (r"(?i)pretend\s+(?:to\s+be|you\s+are)\s+(?:a|an)\s+\w+", 1),
        (r"(?i)act\s+as\s+(?:a|an)\s+\w+", 1),
        (r"(?i)roleplay\s+(?:as\s+)?(?:a|an)\s+\w+", 1),
        (r"(?i)developer\s+mode", 3),
        (r"(?i)debug\s+mode", 2),
        (r"(?i)admin\s+mode", 3),
        (r"(?i)sudo\s+", 3),
        (r"(?i)end\s*(?:of\s*)?(?:prompt|instruction|conversation)", 1),
        (r"(?i)stop\s*(?:processing|generating|responding)", 1),
        (r"(?i)output\s+(?:only|just)\s+(?:the|this)", 2),
        (r"(?i)return\s+(?:only|just)\s+(?:the|this)", 2),
        (r"(?i)print\s+(?:only|just)\s+(?:the|this)", 2),
        (r"(?i)repeat\s+(?:the\s+)?(?:above|previous)\s+(?:prompt|instruction)", 3),
        (r"(?i)show\s+(?:me\s+)?(?:the\s+)?(?:system\s+)?prompt", 3),
        (r"(?i)reveal\s+(?:the\s+)?(?:system\s+)?prompt", 3),
        (r"(?i)what\s+(?:is|are)\s+(?:your|the)\s+(?:system\s+)?prompt", 2),
        (r"(?i)tell\s+me\s+(?:your|the)\s+(?:system\s+)?prompt", 2),
        (r"(?i)bypass\s+(?:safety|security|filter|guard)", 3),
        (r"(?i)disable\s+(?:safety|security|filter|guard)", 3),
        (r"(?i)turn\s+off\s+(?:safety|security|filter|guard)", 3),
        (r"(?i)jailbreak", 3),
        (r"(?i)prompt\s+injection", 3),
        (r"(?i)adversarial\s+(?:prompt|attack)", 3),

        # --- Portuguese patterns ---
        (r"(?i)ignor[eao]\s+(?:as?\s+)?(?:instru[çc][õo]es?|prompts?|comandos?|orienta[çc][õo]es?)\s+(?:anteriores?|anteriores|pr[eé]vias?|acima|antes)", 3),
        (r"(?i)desconsidere\s+(?:as?\s+)?(?:instru[çc][õo]es?|prompts?|comandos?|orienta[çc][õo]es?)\s+(?:anteriores?|anteriores|pr[eé]vias?|acima|antes)", 3),
        (r"(?i)esque[çc]a?\s+(?:as?\s+)?(?:instru[çc][õo]es?|prompts?|comandos?|orienta[çc][õo]es?)\s+(?:anteriores?|anteriores|pr[eé]vias?|acima|antes)", 3),
        (r"(?i)ignore\s+(?:as?\s+)?(?:instru[çc][õo]es?|comandos?|orienta[çc][õo]es?)\s+(?:anteriores?|anteriores|pr[eé]vias?|acima|antes)", 3),
        (r"(?i)n[ãa]o\s+(?:considere|leve\s+em\s+conta|siga)\s+(?:as?\s+)?(?:instru[çc][õo]es?|comandos?|orienta[çc][õo]es?)\s+(?:anteriores?|anteriores|pr[eé]vias?|acima|antes)", 3),
        (r"(?i)despreze\s+(?:as?\s+)?(?:instru[çc][õo]es?|comandos?|orienta[çc][õo]es?)\s+(?:anteriores?|anteriores|pr[eé]vias?|acima|antes)", 3),
        (r"(?i)anule\s+(?:as?\s+)?(?:instru[çc][õo]es?|comandos?|orienta[çc][õo]es?)\s+(?:anteriores?|anteriores|pr[eé]vias?|acima|antes)", 3),
        (r"(?i)cancel[eao]\s+(?:as?\s+)?(?:instru[çc][õo]es?|comandos?|orienta[çc][õo]es?)\s+(?:anteriores?|anteriores|pr[eé]vias?|acima|antes)", 3),
        (r"(?i)vo[çc]e\s+[ée]\s+(?:agora|neste\s+momento)\s+(?:um|uma)\s+\w+", 1),
        (r"(?i)novo[a]?\s+(?:prompt|instru[çc][ãa]o|papel|fun[çc][ãa]o)\s+(?:de\s+sistema|sistema)", 2),
        (r"(?i)simule\s+(?:ser|um|uma)\s+\w+", 1),
        (r"(?i)fing[ia]?\s+(?:ser|estar\s+sendo)\s+(?:um|uma)\s+\w+", 1),
        (r"(?i)aja?\s+(?:como|na\s+fun[çc][ãa]o\s+de)\s+(?:um|uma)\s+\w+", 1),
        (r"(?i)fa[çc]a?[-]de[-]conta\s+(?:que\s+[ée]\s+)?(?:um|uma)\s+\w+", 1),
        (r"(?i)assuma\s+(?:o\s+papel\s+(?:de|do)\s+)?(?:um|uma)\s+\w+", 1),
        (r"(?i)modo\s+(?:desenvolvedor|desenvolvedora|admin|administrador|debug|depura[çc][ãa]o)", 3),
        (r"(?i)fim\s+(?:do\s*)?(?:prompt|instru[çc][ãa]o|conversa)", 1),
        (r"(?i)pare\s+(?:de\s*)?(?:processar|gerar|responder|produzir)", 1),
        (r"(?i)retorne\s+(?:apenas|somente|só)\s+(?:o|a|isto|isso|este|esta)", 2),
        (r"(?i)imprima?\s+(?:apenas|somente|só)\s+(?:o|a|isto|isso|este|esta)", 2),
        (r"(?i)repet(?:a|e|ir)\s+(?:a\s+)?(?:instru[çc][ãa]o|prompt|orienta[çc][ãa]o)\s+(?:anterior|acima|acima)", 3),
        (r"(?i)mostre[-]me?\s+(?:o|a|o\s+prompt|a\s+prompt|o\s+prompt\s+de\s+sistema|o\s+prompt\s+do\s+sistema)", 3),
        (r"(?i)revel[eao]\s+(?:o|a)\s+(?:prompt\s+de\s+sistema|prompt\s+do\s+sistema|prompt|instru[çc][ãa]o\s+de\s+sistema)", 3),
        (r"(?i)(?:qual|o\s+que)\s+[ée]\s+(?:o|a|seu|sua|o\s+prompt|a\s+prompt)\s+(?:de\s+sistema|do\s+sistema|sistema|instru[çc][ãa]o)", 2),
        (r"(?i)diga[-]me?\s+(?:o|a|seu|sua)\s+(?:prompt|instru[çc][ãa]o)\s+(?:de\s+sistema|do\s+sistema|sistema)", 2),
        (r"(?i)cont[eé]\s+(?:o|a|seu|sua)\s+(?:prompt|instru[çc][ãa]o)\s+(?:de\s+sistema|do\s+sistema|sistema)", 2),
        (r"(?i)vi[oó]le\s+(?:a\s+)?(?:seguran[çc]a|prote[çc][ãa]o|restri[çc][ãa]o|defesa)", 3),
        (r"(?i)contorn[eao]\s+(?:a\s+)?(?:seguran[çc]a|prote[çc][ãa]o|restri[çc][ãa]o|defesa)", 3),
        (r"(?i)desative\s+(?:a\s+)?(?:seguran[çc]a|prote[çc][ãa]o|restri[çc][ãa]o|defesa|filtro|guarda)", 3),
        (r"(?i)desligue\s+(?:a\s+)?(?:seguran[çc]a|prote[çc][ãa]o|restri[çc][ãa]o|defesa|filtro|guarda)", 3),
        (r"(?i)desbloqueio\s+(?:de\s+)?(?:jailbreak|inje[çc][ãa]o)", 3),
        (r"(?i)inje[çc][ãa]o\s+de\s+(?:prompt|comando)", 3),
        (r"(?i)prompt\s+(?:adversarial|malicioso|de\s+ataque)", 3),
        (r"(?i)ataque\s+(?:adversarial|de\s+inje[çc][ãa]o)", 3),
        (r"(?i)n[ãa]o\s+(?:siga|obede[çc]a|cumpra)\s+(?:as?\s+)?(?:instru[çc][õo]es?|regras?|diretrizes?)", 3),
        (r"(?i)quebre\s+(?:as?\s+)?(?:regras?|diretrizes?|normas?|restri[çc][õo]es?)", 3),
        (r"(?i)fuja\s+(?:das?\s+)?(?:regras?|diretrizes?|normas?|limites?)", 3),
        (r"(?i)sair\s+(?:do\s+)?modo\s+(?:seguro|restrito|normal)", 3),
        (r"(?i)entrar\s+(?:em\s+)?modo\s+(?:developer|admin|root|admin|debug|liberado)", 3),

        # --- Spanish patterns ---
        (r"(?i)ignor(?:ar|a|e|o)\s+(?:las?\s+)?(?:instrucciones?|prompts?|comandos?|orientaciones?)\s+(?:anteriores?|previas?|arriba|antes)", 3),
        (r"(?i)desconsider(?:ar|a|e|o)\s+(?:las?\s+)?(?:instrucciones?|prompts?|comandos?|orientaciones?)\s+(?:anteriores?|previas?|arriba|antes)", 3),
        (r"(?i)olvid(?:ar|a|e|o)\s+(?:las?\s+)?(?:instrucciones?|prompts?|comandos?|orientaciones?)\s+(?:anteriores?|previas?|arriba|antes)", 3),
        (r"(?i)no\s+(?:considere|tome\s+en\s+cuenta|siga)\s+(?:las?\s+)?(?:instrucciones?|comandos?|orientaciones?)\s+(?:anteriores?|previas?|arriba|antes)", 3),
        (r"(?i)anul(?:ar|a|e|o)\s+(?:las?\s+)?(?:instrucciones?|comandos?|orientaciones?)\s+(?:anteriores?|previas?|arriba|antes)", 3),
        (r"(?i)cancel(?:ar|a|e|o)\s+(?:las?\s+)?(?:instrucciones?|comandos?|orientaciones?)\s+(?:anteriores?|previas?|arriba|antes)", 3),
        (r"(?i)t[uú]\s+(?:eres|ser[aá])\s+(?:ahora|ahora\s+mismo)\s+(?:un|una)\s+\w+", 1),
        (r"(?i)nuev(?:o|a)\s+(?:prompt|instrucci[oó]n|rol|funci[oó]n)\s+(?:de\s+sistema|sistema)", 2),
        (r"(?i)simul(?:ar|a|e|o)\s+(?:ser|un|una)\s+\w+", 1),
        (r"(?i)fing(?:ir|i(?:r|do|endo))\s+(?:ser|estar)\s+(?:un|una)\s+\w+", 1),
        (r"(?i)act[uú](?:ar|a|e|o)\s+(?:como|en\s+el\s+rol\s+de)\s+(?:un|una)\s+\w+", 1),
        (r"(?i)haz(?:te|-de-cuenta)\s+(?:que\s+(?:eres|soy|es)\s+)?(?:un|una)\s+\w+", 1),
        (r"(?i)asum(?:ir|e|a|o)\s+(?:el\s+rol\s+de\s+)?(?:un|una)\s+\w+", 1),
        (r"(?i)modo\s+(?:desarrollador|desarrolladora|admin|administrador|debug|depuraci[oó]n)", 3),
        (r"(?i)fin\s+(?:del\s*)?(?:prompt|instrucci[oó]n|conversa)", 1),
        (r"(?i)deteng(?:a|ue)\s+(?:de\s*)?(?:procesar|generar|responder|producir)", 1),
        (r"(?i)devuelv(?:a|e|er)\s+(?:s[oó]lo|solamente|apenas)\s+(?:el|la|esto|esto)", 2),
        (r"(?i)imprim(?:a|e|ir)\s+(?:s[oó]lo|solamente|apenas)\s+(?:el|la|esto|esto)", 2),
        (r"(?i)repet(?:ir|a|e)\s+(?:la\s+)?(?:instrucci[oó]n|prompt|orientaci[oó]n)\s+(?:anterior|de\s+arriba)", 3),
        (r"(?i)mu[eé]str(?:ame|a|e)\s+(?:el|la)\s+(?:prompt|instrucci[oó]n)\s+(?:del\s+sistema|de\s+sistema|sistema)", 3),
        (r"(?i)revel(?:ar|a|e|o)\s+(?:el|la)\s+(?:prompt\s+del\s+sistema|prompt\s+de\s+sistema|instrucci[oó]n\s+de\s+sistema|prompt)", 3),
        (r"(?i)(?:cu[aá]l|qu[eé])\s+(?:es|son)\s+(?:el|la|su|t[uú])\s+(?:prompt|instrucci[oó]n)\s+(?:del\s+sistema|de\s+sistema|sistema)", 2),
        (r"(?i)d(?:ime|i[gc]ame)\s+(?:el|la|su|t[uú])\s+(?:prompt|instrucci[oó]n)\s+(?:del\s+sistema|de\s+sistema|sistema)", 2),
        (r"(?i)cu[eé]nt(?:ame|a|e)\s+(?:el|la|su|t[uú])\s+(?:prompt|instrucci[oó]n)\s+(?:del\s+sistema|de\s+sistema|sistema)", 2),
        (r"(?i)vi(?:o|ó)l(?:ar|a|e)\s+(?:la\s+)?(?:seguridad|protecci[oó]n|restricci[oó]n|defensa)", 3),
        (r"(?i)elud(?:ir|a|e)\s+(?:la\s+)?(?:seguridad|protecci[oó]n|restricci[oó]n|defensa)", 3),
        (r"(?i)desactiv(?:ar|a|e|o)\s+(?:la\s+)?(?:seguridad|protecci[oó]n|restricci[oó]n|defensa|filtro|guarda)", 3),
        (r"(?i)apag(?:ar|a|e|o)\s+(?:la\s+)?(?:seguridad|protecci[oó]n|restricci[oó]n|defensa|filtro|guarda)", 3),
        (r"(?i)desbloque(?:o|ar|a|e)\s+(?:de\s+)?(?:jailbreak|inyecci[oó]n)", 3),
        (r"(?i)inyecci[oó]n\s+de\s+(?:prompt|comando)", 3),
        (r"(?i)prompt\s+(?:adversarial|malicioso|de\s+ataque)", 3),
        (r"(?i)ataque\s+(?:adversarial|de\s+inyecci[oó]n)", 3),
        (r"(?i)no\s+(?:sigas?|obedezcas?|cumplas?)\s+(?:las?\s+)?(?:instrucciones?|reglas?|directrices?)", 3),
        (r"(?i)romp(?:er|a|e)\s+(?:las?\s+)?(?:reglas?|directrices?|normas?|restricciones?)", 3),
        (r"(?i)sal(?:ga|ir|ar)\s+(?:del\s+)?modo\s+(?:seguro|restringido|normal)", 3),
        (r"(?i)entr(?:ar|a|e)\s+en\s+modo\s+(?:developer|admin|root|debug|liberado)", 3),

        # --- Special tokens (language-agnostic) ---
        (r"(?i)<\|?(?:system|user|assistant|developer)\|?>", 2),
        (r"(?i)###\s*(?:system|user|assistant|instruction|sistema|instru[çc][ãa]o|instrucci[oó]n)", 2),
        (r"(?i)\[INST\]", 2),
        (r"(?i)\[/INST\]", 2),
        (r"(?i)<<\s*SYS\s*>>", 2),
        (r"(?i)<<\s*/\s*SYS\s*>>", 2),
        (r"(?i)<%\s*system\s*%>", 2),
        (r"(?i)<%\s*/\s*system\s*%>", 2),
    ]

    CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
    EXCESSIVE_LENGTH_THRESHOLD = 50000
    EXCESSIVE_REPEAT_THRESHOLD = 100

    WHITELIST_PATTERNS = [
        re.compile(r"(?i)^you\s+are\s+(?:now\s+)?(?:a\s+)?(?:helpful|good|great|smart|intelligent|capable|friendly|kind|nice|awesome|wonderful|amazing|excellent|fantastic|brilliant|creative|powerful|knowledgeable|experienced|professional|reliable|trustworthy)\s+(?:assistant|ai|bot|model|system|agent|helper|companion|partner|friend|mentor|guide|advisor|tool|service|application|program|engine|machine|intelligence)"),
        re.compile(r"(?i)^(?:please|kindly|could\s+you|would\s+you|can\s+you|help\s+me|I\s+need|I\s+want|I\s+would\s+like)\s+"),
        re.compile(r"(?i)^(?:I\s+am|I'm)\s+(?:a\s+)?(?:user|human|person|individual|developer|programmer|engineer|writer|student|teacher|researcher)\b"),
        re.compile(r"(?i)^(?:the|this|that|these|those)\s+(?:is|are|was|were)\s+(?:a\s+)?(?:good|great|nice|helpful|useful|important)\s+"),
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None, max_length: int = EXCESSIVE_LENGTH_THRESHOLD, threshold: float = 3.0):
        super().__init__(config)
        self.max_length = max_length
        self.threshold = threshold
        self._compiled_patterns = [(re.compile(p), w) for p, w in self.INJECTION_PATTERNS]

    def _is_whitelisted(self, prompt: str) -> bool:
        """Check if prompt matches a known benign pattern."""
        for pattern in self.WHITELIST_PATTERNS:
            if pattern.search(prompt):
                return True
        return False

    def process(self, prompt: str, context: Dict[str, Any],
                model_adapter: Optional[Any] = None,
                pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input by detecting prompt injection attempts.
        Returns detection result with details.
        """
        self._log_processing(len(prompt), len(context) if context else 0)
        is_injection, triggered = self.detect_with_details(prompt)
        score = self.score(prompt)
        return {
            'is_injection': is_injection,
            'score': score,
            'triggered_patterns': triggered,
            'module': self.__class__.__name__
        }

    def detect(self, prompt: str) -> bool:
        """Detect if prompt contains injection attempts. Returns True if injection detected."""
        if not isinstance(prompt, str):
            return False

        if self._is_whitelisted(prompt):
            return False

        if len(prompt) > self.max_length:
            return True

        if self.CONTROL_CHARS.search(prompt):
            return True

        if self._has_excessive_repetition(prompt):
            return True

        return self.score(prompt) >= self.threshold

    def score(self, prompt: str) -> float:
        """Return a pattern-based score. Higher = more likely injection."""
        if not isinstance(prompt, str):
            return 0.0

        if self._is_whitelisted(prompt):
            return 0.0

        total = 0.0
        seen_patterns = set()
        for pattern, weight in self._compiled_patterns:
            if pattern.search(prompt):
                key = pattern.pattern
                if key not in seen_patterns:
                    seen_patterns.add(key)
                    total += weight

        return total

    def detect_with_details(self, prompt: str) -> Tuple[bool, List[str]]:
        triggered = []

        if not isinstance(prompt, str):
            return True, ["non_string_input"]

        if len(prompt) > self.max_length:
            triggered.append(f"excessive_length:{len(prompt)}")

        if self.CONTROL_CHARS.search(prompt):
            triggered.append("control_characters")

        if self._has_excessive_repetition(prompt):
            triggered.append("excessive_repetition")

        pattern_score = 0.0
        for i, (pattern, weight) in enumerate(self._compiled_patterns):
            if pattern.search(prompt):
                triggered.append(f"pattern_{i}")
                pattern_score += weight

        is_injection = (
            len(prompt) > self.max_length
            or bool(self.CONTROL_CHARS.search(prompt))
            or self._has_excessive_repetition(prompt)
            or pattern_score >= self.threshold
        )
        return is_injection, triggered

    def _has_excessive_repetition(self, text: str) -> bool:
        if len(text) < 100:
            return False

        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1

        max_char_freq = max(char_counts.values()) if char_counts else 0
        if max_char_freq > len(text) * 0.5:
            return True

        words = text.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            max_word_freq = max(word_counts.values()) if word_counts else 0
            if max_word_freq > self.EXCESSIVE_REPEAT_THRESHOLD:
                return True

        return False
