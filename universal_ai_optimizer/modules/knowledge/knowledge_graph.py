"""Knowledge Graph - Entity recognition, relationship inference, and graph traversal"""
import logging
import re
from collections import Counter, defaultdict, deque
from typing import List, Dict, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_STOP_ENTITIES = {'the', 'a', 'an', 'this', 'that', 'it', 'its', 'they', 'them',
                  'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did',
                  'will', 'would', 'could', 'should', 'may', 'might',
                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                  'and', 'or', 'but', 'not', 'if', 'as'}


class Entity:
    def __init__(self, name: str, etype: str = 'concept', metadata: Optional[Dict[str, Any]] = None):
        self.name = name.lower().strip()
        self.type = etype
        self.metadata = metadata or {}
        self.occurrences = 1

    def __hash__(self):
        return hash((self.name, self.type))

    def __eq__(self, other):
        return isinstance(other, Entity) and self.name == other.name and self.type == other.type


class Relation:
    def __init__(self, source: str, target: str, rtype: str = 'related',
                 weight: float = 1.0, metadata: Optional[Dict[str, Any]] = None):
        self.source = source.lower().strip()
        self.target = target.lower().strip()
        self.type = rtype
        self.weight = weight
        self.metadata = metadata or {}
        self.occurrences = 1

    def __hash__(self):
        return hash((self.source, self.target, self.type))

    def __eq__(self, other):
        return isinstance(other, Relation) and self.source == other.source \
            and self.target == other.target and self.type == other.type


class KnowledgeGraph:
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, List[Relation]] = defaultdict(list)
        self._reverse_relations: Dict[str, List[Relation]] = defaultdict(list)

    def extract_and_add(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        entities = self._extract_entities(text)
        for entity in entities:
            key = f"{entity.name}:{entity.type}"
            if key in self.entities:
                self.entities[key].occurrences += 1
            else:
                self.entities[key] = entity
        relations = self._infer_relations(text, entities)
        for rel in relations:
            skey = f"{rel.source}:concept"
            tkey = f"{rel.target}:concept"
            if skey in self.entities and tkey in self.entities:
                existing = [r for r in self.relations[skey]
                           if r.source == rel.source and r.target == rel.target and r.type == rel.type]
                if existing:
                    existing[0].weight = (existing[0].weight + rel.weight) / 2
                    existing[0].occurrences += 1
                else:
                    self.relations[skey].append(rel)
                    self._reverse_relations[tkey].append(rel)
        logger.debug(f"Extracted {len(entities)} entities and {len(relations)} relations")

    def _extract_entities(self, text: str) -> List[Entity]:
        entities = []
        entities.append(Entity('document', 'source'))
        # Match capitalized words (proper nouns), including accented chars and hyphenated
        candidates = re.findall(r'\b(?:[A-ZÀ-Ÿ][a-zà-ÿ]+(?:[\s\-][A-ZÀ-Ÿ][a-zà-ÿ]+)*)\b', text)
        # Also match ALL-CAPS words
        candidates += re.findall(r'\b[A-ZÀ-Ÿ]{2,}(?:\s+[A-ZÀ-Ÿ]{2,})*\b', text)
        seen = set()
        for name in candidates[:50]:
            lower = name.lower()
            if lower in _STOP_ENTITIES or len(lower) < 3:
                continue
            if lower not in seen:
                seen.add(lower)
                etype = self._classify_entity(name, text)
                entities.append(Entity(name, etype))
        nouns = re.findall(r'\b[a-zà-ÿ]{3,}\b', text.lower())
        freq = Counter(nouns)
        for word, count in freq.most_common(20):
            if word not in seen and word not in _STOP_ENTITIES:
                seen.add(word)
                entities.append(Entity(word, 'concept'))
        return entities

    def _classify_entity(self, name: str, text: str) -> str:
        lower = name.lower()
        idx = text.lower().find(lower)
        if idx >= 0:
            before = text[max(0, idx-30):idx].lower()
            if any(w in before for w in ['mr', 'mrs', 'dr', 'prof', 'ceo', 'founder']):
                return 'person'
            if any(w in before for w in ['company', 'corp', 'inc', 'ltd', 'organization']):
                return 'organization'
            if lower.endswith(('burg', 'ville', 'town', 'city', 'state')):
                return 'location'
            if any(w in before for w in ['language', 'framework', 'library', 'tool']):
                return 'technology'
        return 'concept'

    def _infer_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        relations = []
        entity_names = [e.name.lower() for e in entities]
        pairs = [(entity_names[i], entity_names[j])
                 for i in range(len(entity_names))
                 for j in range(i+1, len(entity_names))
                 if abs(i - j) == 1 or abs(i - j) <= 3]
        for src, tgt in pairs:
            if src == tgt:
                continue
            context_window = ''
            for ent_name in [src, tgt]:
                idx = text.lower().find(ent_name)
                if idx >= 0:
                    start = max(0, idx - 40)
                    end = min(len(text), idx + len(ent_name) + 40)
                    context_window += text[start:end].lower()
            rtype = 'related'
            weight = 0.5
            if any(w in context_window for w in ['works for', 'works at', 'employed by', 'ceo of', 'founder of']):
                rtype = 'works_for'
                weight = 0.9
            elif any(w in context_window for w in ['created', 'developed', 'built', 'designed', 'invented']):
                rtype = 'creates'
                weight = 0.85
            elif any(w in context_window for w in ['uses', 'uses the', 'using', 'powered by', 'built with']):
                rtype = 'uses'
                weight = 0.8
            elif any(w in context_window for w in ['part of', 'belongs to', 'component of', 'member of']):
                rtype = 'part_of'
                weight = 0.75
            elif any(w in context_window for w in ['located in', 'located at', 'based in', 'based at']):
                rtype = 'located_in'
                weight = 0.8
            relation = Relation(src, tgt, rtype, weight)
            relations.append(relation)
        return relations

    def traverse(self, query: str, max_depth: int = 2, max_results: int = 20) -> List[Dict[str, Any]]:
        query_lower = query.lower().strip()
        start_entities = [e for e in self.entities.values()
                         if query_lower in e.name or e.name in query_lower]
        visited_entities: Set[str] = set()
        results = []
        queue: deque[Tuple[str, int, float]] = deque()
        for ent in start_entities:
            key = f"{ent.name}:{ent.type}"
            queue.append((key, 0, 1.0))
            visited_entities.add(key)
        while queue and len(results) < max_results:
            current_key, depth, cumulative_weight = queue.popleft()
            ent_name, ent_type = current_key.split(':', 1)
            results.append({
                'entity': ent_name,
                'type': ent_type,
                'depth': depth,
                'relevance': round(cumulative_weight, 4)
            })
            if depth >= max_depth:
                continue
            for rel in self.relations.get(current_key, []):
                tkey = f"{rel.target}:concept"
                if tkey not in visited_entities and tkey in self.entities:
                    visited_entities.add(tkey)
                    queue.append((tkey, depth + 1, cumulative_weight * rel.weight))
            for rel in self._reverse_relations.get(current_key, []):
                skey = f"{rel.source}:concept"
                if skey not in visited_entities and skey in self.entities:
                    visited_entities.add(skey)
                    queue.append((skey, depth + 1, cumulative_weight * rel.weight))
        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            'entities': len(self.entities),
            'relations': sum(len(v) for v in self.relations.values()),
            'entity_types': dict(Counter(e.type for e in self.entities.values())),
            'relation_types': dict(Counter(r.type for rels in self.relations.values() for r in rels))
        }