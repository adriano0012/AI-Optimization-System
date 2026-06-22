"""
A/B Testing Framework
Create and manage experiments with statistical analysis.
"""

import time
import secrets
import hashlib
import threading
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class VariantStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    WINNER = "winner"
    LOSER = "loser"


@dataclass
class Variant:
    variant_id: str
    name: str
    weight: float
    config: Dict[str, Any] = field(default_factory=dict)
    status: VariantStatus = VariantStatus.ACTIVE
    impressions: int = 0
    conversions: int = 0
    revenue: float = 0.0

    @property
    def conversion_rate(self) -> float:
        return self.conversions / self.impressions if self.impressions > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'variant_id': self.variant_id,
            'name': self.name,
            'weight': self.weight,
            'config': self.config,
            'status': self.status.value,
            'impressions': self.impressions,
            'conversions': self.conversions,
            'conversion_rate': round(self.conversion_rate, 4),
            'revenue': self.revenue,
        }


@dataclass
class Experiment:
    experiment_id: str
    name: str
    description: str
    status: ExperimentStatus = ExperimentStatus.DRAFT
    variants: List[Variant] = field(default_factory=list)
    target_metric: str = "conversion_rate"
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    organization_id: Optional[str] = None
    created_by: Optional[str] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    winner_variant_id: Optional[str] = None

    @property
    def total_impressions(self) -> int:
        return sum(v.impressions for v in self.variants)

    @property
    def total_conversions(self) -> int:
        return sum(v.conversions for v in self.variants)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'variants': [v.to_dict() for v in self.variants],
            'target_metric': self.target_metric,
            'total_impressions': self.total_impressions,
            'total_conversions': self.total_conversions,
            'started_at': self.started_at,
            'ended_at': self.ended_at,
            'winner_variant_id': self.winner_variant_id,
        }


class ABTestManager:
    """
    A/B testing framework with statistical significance testing.
    """

    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}
        self._lock = threading.Lock()

    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        target_metric: str = "conversion_rate",
        min_sample_size: int = 1000,
        confidence_level: float = 0.95,
        organization_id: Optional[str] = None,
    ) -> Experiment:
        exp_id = f"exp-{secrets.token_hex(8)}"
        now = time.time()

        variant_objs = []
        for i, v in enumerate(variants):
            variant_objs.append(Variant(
                variant_id=f"var-{secrets.token_hex(6)}",
                name=v.get('name', f'variant_{i}'),
                weight=v.get('weight', 1.0 / len(variants)),
                config=v.get('config', {}),
            ))

        experiment = Experiment(
            experiment_id=exp_id, name=name, description=description,
            variants=variant_objs, target_metric=target_metric,
            min_sample_size=min_sample_size, confidence_level=confidence_level,
            organization_id=organization_id, created_at=now,
        )
        with self._lock:
            self._experiments[exp_id] = experiment
        return experiment

    def start_experiment(self, experiment_id: str) -> Optional[Experiment]:
        exp = self._experiments.get(experiment_id)
        if exp and exp.status == ExperimentStatus.DRAFT:
            exp.status = ExperimentStatus.RUNNING
            exp.started_at = time.time()
        return exp

    def stop_experiment(self, experiment_id: str) -> Optional[Experiment]:
        exp = self._experiments.get(experiment_id)
        if exp and exp.status == ExperimentStatus.RUNNING:
            exp.status = ExperimentStatus.COMPLETED
            exp.ended_at = time.time()
        return exp

    def assign_variant(self, experiment_id: str, user_id: str) -> Optional[Variant]:
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return None

        hash_val = int(hashlib.md5(
            f"{experiment_id}:{user_id}".encode()
        ).hexdigest(), 16)
        normalized = (hash_val % 10000) / 10000.0

        cumulative = 0.0
        for variant in exp.variants:
            cumulative += variant.weight
            if normalized < cumulative:
                return variant
        return exp.variants[-1] if exp.variants else None

    def record_impression(self, experiment_id: str, variant_id: str):
        exp = self._experiments.get(experiment_id)
        if exp:
            for v in exp.variants:
                if v.variant_id == variant_id:
                    v.impressions += 1
                    break

    def record_conversion(
        self, experiment_id: str, variant_id: str, revenue: float = 0.0
    ):
        exp = self._experiments.get(experiment_id)
        if exp:
            for v in exp.variants:
                if v.variant_id == variant_id:
                    v.conversions += 1
                    v.revenue += revenue
                    break

    def calculate_significance(
        self, experiment_id: str
    ) -> Optional[Dict[str, Any]]:
        exp = self._experiments.get(experiment_id)
        if not exp or len(exp.variants) < 2:
            return None

        control = exp.variants[0]
        treatment = exp.variants[1]

        if control.impressions == 0 or treatment.impressions == 0:
            return None

        p1 = control.conversion_rate
        p2 = treatment.conversion_rate
        n1 = control.impressions
        n2 = treatment.impressions

        p_pool = (control.conversions + treatment.conversions) / (n1 + n2)
        if p_pool == 0 or p_pool == 1:
            return {'significant': False, 'p_value': 1.0}

        se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        if se == 0:
            return {'significant': False, 'p_value': 1.0}

        z_score = (p2 - p1) / se
        p_value = 2 * (1 - self._norm_cdf(abs(z_score)))
        significant = p_value < (1 - exp.confidence_level)

        lift = ((p2 - p1) / p1 * 100) if p1 > 0 else 0.0

        return {
            'significant': significant,
            'p_value': round(p_value, 6),
            'z_score': round(z_score, 4),
            'lift_percent': round(lift, 2),
            'control_rate': round(p1, 4),
            'treatment_rate': round(p2, 4),
            'control_n': n1,
            'treatment_n': n2,
        }

    def declare_winner(self, experiment_id: str, variant_id: str):
        exp = self._experiments.get(experiment_id)
        if exp:
            exp.winner_variant_id = variant_id
            for v in exp.variants:
                if v.variant_id == variant_id:
                    v.status = VariantStatus.WINNER
                else:
                    v.status = VariantStatus.LOSER
            exp.status = ExperimentStatus.COMPLETED
            exp.ended_at = time.time()

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        return self._experiments.get(experiment_id)

    def list_experiments(
        self, organization_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        exps = list(self._experiments.values())
        if organization_id:
            exps = [e for e in exps if e.organization_id == organization_id]
        return [e.to_dict() for e in exps]

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_experiments': len(self._experiments),
            'running': sum(1 for e in self._experiments.values()
                         if e.status == ExperimentStatus.RUNNING),
            'completed': sum(1 for e in self._experiments.values()
                           if e.status == ExperimentStatus.COMPLETED),
        }
