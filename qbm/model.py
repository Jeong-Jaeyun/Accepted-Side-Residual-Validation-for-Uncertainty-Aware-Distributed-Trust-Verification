from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple
import json
import math

import numpy as np


FEATURE_COLUMNS: Tuple[str, ...] = (
    "F1_unique_mmsi_count",
    "F2_new_mmsi_rate",
    "F3_message_burstiness",
    "F4_position_jump_rate",
    "F5_speed_heading_inconsistency",
    "F6_spatial_density_entropy",
)


class DropType(str, Enum):
    NONE = "none"
    VERIFICATION = "verification"
    NETWORK = "network"
    OVERFLOW = "overflow"


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def stable_hash(payload: Mapping[str, Any]) -> str:
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(packed.encode("utf-8")).hexdigest()


def explanation_hash(explanation_topk: Sequence[str]) -> str:
    return stable_hash({"topk": list(explanation_topk)})


def jaccard_similarity(lhs: Sequence[str], rhs: Sequence[str]) -> float:
    a, b = set(lhs), set(rhs)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pairwise_mean(values: Sequence[float]) -> float:
    n = len(values)
    if n <= 1:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += values[i] * values[j]
            pairs += 1
    return total / max(pairs, 1)


def pairwise_trust_correlation(trust_scores: Sequence[float]) -> float:
    n = len(trust_scores)
    if n <= 1:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = abs(trust_scores[i] - trust_scores[j])
            total += 1.0 - min(1.0, dist)
            pairs += 1
    return total / max(pairs, 1)


def trust_consistency_varnorm(trust_scores: Sequence[float], sigma_max: float = 0.25) -> float:
    """
    Stable trust-consistency score in [0,1]:
      C = 1 - Var(T_i) / sigma_max
    For trust scores bounded in [0,1], the maximum variance is 0.25.
    """
    if len(trust_scores) <= 1:
        return 1.0
    sigma_cap = max(float(sigma_max), 1.0e-12)
    variance = float(np.var(np.asarray(trust_scores, dtype=float)))
    return clamp01(1.0 - (variance / sigma_cap))


def mean_explanation_similarity(explanations: Sequence[Sequence[str]]) -> float:
    n = len(explanations)
    if n <= 1:
        return 1.0
    sims: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(jaccard_similarity(explanations[i], explanations[j]))
    return float(np.mean(sims)) if sims else 1.0


def explanation_consistency_per_evidence(explanations: Sequence[Sequence[str]]) -> List[float]:
    n = len(explanations)
    if n <= 1:
        return [1.0] * n
    out: List[float] = []
    for i in range(n):
        sims: List[float] = []
        for j in range(n):
            if i == j:
                continue
            sims.append(jaccard_similarity(explanations[i], explanations[j]))
        out.append(float(np.mean(sims)) if sims else 1.0)
    return out


def compute_trust_scores(
    anomaly_scores: Sequence[float],
    confidences: Sequence[float],
    explanations: Sequence[Sequence[str]],
    weights: Mapping[str, float],
) -> List[float]:
    ecs = explanation_consistency_per_evidence(explanations)
    ws = float(weights.get("score", 0.45))
    wc = float(weights.get("confidence", 0.40))
    we = float(weights.get("explanation", 0.15))
    norm = ws + wc + we
    if norm <= 0:
        ws, wc, we = 0.45, 0.40, 0.15
        norm = 1.0
    ws, wc, we = ws / norm, wc / norm, we / norm
    out: List[float] = []
    for s, c, e in zip(anomaly_scores, confidences, ecs):
        out.append(clamp01((ws * s) + (wc * c) + (we * e)))
    return out


@dataclass
class Evidence:
    event_id: str
    window_id: int
    node_id: str
    decision: str
    anomaly_score: float
    confidence: float
    uncertainty: float
    explanation_topk: Tuple[str, ...]
    explanation_hash: str
    model_id: str
    policy_id: str
    params_hash: str
    context_hash: str
    timestamp_ms: int
    prev_evidence_hash: str
    malicious: bool = False
    attack_label: str = "A0"

    def payload_hash(self) -> str:
        return stable_hash(
            {
                "event_id": self.event_id,
                "window_id": self.window_id,
                "node_id": self.node_id,
                "decision": self.decision,
                "anomaly_score": round(float(self.anomaly_score), 6),
                "confidence": round(float(self.confidence), 6),
                "uncertainty": round(float(self.uncertainty), 6),
                "explanation_hash": self.explanation_hash,
                "model_id": self.model_id,
                "policy_id": self.policy_id,
                "params_hash": self.params_hash,
                "context_hash": self.context_hash,
                "timestamp_ms": int(self.timestamp_ms),
                "prev_evidence_hash": self.prev_evidence_hash,
                "malicious": bool(self.malicious),
                "attack_label": self.attack_label,
            }
        )


@dataclass
class EvidenceBatch:
    window_id: int
    evidences: List[Evidence]
    network_state: Dict[str, Any] = field(default_factory=dict)

    @property
    def malicious_count(self) -> int:
        return sum(1 for e in self.evidences if e.malicious)

    @property
    def attack_labels(self) -> Set[str]:
        return {e.attack_label for e in self.evidences if e.malicious}


@dataclass
class VerifyResult:
    accept: bool
    reason: str
    scores: Dict[str, float] = field(default_factory=dict)
    drop_type: DropType = DropType.NONE


@dataclass
class VerificationHistory:
    committed_event_ids: Set[str] = field(default_factory=set)
    committed_payload_hashes: Set[str] = field(default_factory=set)
    committed_context_hashes: Set[str] = field(default_factory=set)
    committed_evidences: List[Evidence] = field(default_factory=list)
    last_ts_by_node: Dict[str, int] = field(default_factory=dict)
    last_window_id: Optional[int] = None
    replay_buffer_max: int = 2048

    def record_commit(self, batch: EvidenceBatch) -> None:
        for e in batch.evidences:
            self.committed_event_ids.add(e.event_id)
            self.committed_payload_hashes.add(e.payload_hash())
            self.committed_context_hashes.add(e.context_hash)
            self.last_ts_by_node[e.node_id] = int(e.timestamp_ms)
            self.committed_evidences.append(e)
        self.last_window_id = int(batch.window_id)
        if len(self.committed_evidences) > self.replay_buffer_max:
            keep = self.committed_evidences[-self.replay_buffer_max :]
            self.committed_evidences = keep


@dataclass
class SimulationSummary:
    scenario: str
    attack_id: str
    verifier_name: str
    processed_tps_mean: float
    latency_ms_mean: float
    backlog_max: float
    dropped_sum: float
    dropped_by_verification_sum: float
    dropped_by_network_sum: float
    dropped_by_overflow_sum: float
    policy_fired_ratio: float
    asr: float
    ftr: float
    tcp: float
    ttd_windows: float
    auroc: float = float("nan")
    auprc: float = float("nan")
    mcc: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    f1_score: float = float("nan")
    fpr_at_tpr95: float = float("nan")
    detection_delay_windows: float = float("nan")
    cost_sensitive_score: float = float("nan")
    n_gray_zone: int = 0
    n_strongq_called: int = 0
    n_flip_mev_to_reject: int = 0
    n_flip_mev_to_accept: int = 0
    n_strongq_agree: int = 0
    n_strongq_disagree: int = 0
