from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Tuple

import numpy as np

from qbm.model import (
    FEATURE_COLUMNS,
    Evidence,
    EvidenceBatch,
    VerificationHistory,
    clamp01,
    explanation_hash,
    stable_hash,
)


ATTACK_SET = {"A0", "A1", "A2", "A3", "A4", "A4P", "A5"}


@dataclass
class AttackPlan:
    attack_id: str = "A0"
    start_window: int = 0
    end_window: int = 0
    rho: float = 0.20
    capture_k: int = 1
    partition_loss_boost: float = 0.12
    # near-replay(A4P) controls
    a4_level: int = 2
    a4_p_topk_mutate: float = 0.20
    a4_ctx_drift_strength: float = 0.05
    a4_conf_jitter: float = 0.03
    a4_time_jitter_slots: int = 1
    a4_recent_windows: int = 120

    def active(self, window_id: int) -> bool:
        if self.attack_id == "A0":
            return False
        return self.start_window <= int(window_id) <= self.end_window


def build_attack_plan(cfg: Dict[str, Any]) -> AttackPlan:
    exp = cfg.get("experiments", {})
    win_cfg = exp.get("injection_window", {})
    start = int(win_cfg.get("start_window", 0))
    end = int(win_cfg.get("end_window", max(start, 0)))

    if not bool(exp.get("enable_injection", True)):
        return AttackPlan(attack_id="A0", start_window=start, end_window=end)

    attack_id = str(exp.get("attack_id", "")).strip().upper()
    if not attack_id:
        scenario = str(exp.get("scenario", "S0")).upper()
        attack_id = {"S0": "A0", "S1": "A1", "S2": "A2", "S3": "A5"}.get(scenario, "A1")
    if attack_id not in ATTACK_SET:
        attack_id = "A1"

    scenario_cfg = exp.get(str(exp.get("scenario", "S1")).upper(), {})
    rho = float(exp.get("rho", scenario_cfg.get("intensity", 0.20)))
    capture_k = int(exp.get("capture_k", 1))
    partition_loss_boost = float(exp.get("partition_loss_boost", 0.12))
    a4_cfg = exp.get("A4", {}) if isinstance(exp.get("A4", {}), dict) else {}
    a4p_cfg = exp.get("A4P", {}) if isinstance(exp.get("A4P", {}), dict) else {}
    a4_level = int(a4p_cfg.get("level", a4_cfg.get("level", exp.get("a4_level", 2))))
    a4_p_topk_mutate = float(
        a4p_cfg.get("p_topk_mutate", a4_cfg.get("p_topk_mutate", exp.get("a4_p_topk_mutate", 0.20)))
    )
    a4_ctx_drift_strength = float(
        a4p_cfg.get("ctx_drift_strength", a4_cfg.get("ctx_drift_strength", exp.get("a4_ctx_drift_strength", 0.05)))
    )
    a4_conf_jitter = float(a4p_cfg.get("conf_jitter", a4_cfg.get("conf_jitter", exp.get("a4_conf_jitter", 0.03))))
    a4_time_jitter_slots = int(
        a4_cfg.get("time_jitter_slots", a4p_cfg.get("time_jitter_slots", exp.get("a4_time_jitter_slots", 1)))
    )
    a4_recent_windows = int(
        a4_cfg.get("recent_windows", a4p_cfg.get("recent_windows", exp.get("a4_recent_windows", 120)))
    )
    return AttackPlan(
        attack_id=attack_id,
        start_window=start,
        end_window=end,
        rho=rho,
        capture_k=capture_k,
        partition_loss_boost=partition_loss_boost,
        a4_level=max(1, min(3, a4_level)),
        a4_p_topk_mutate=float(max(0.0, min(1.0, a4_p_topk_mutate))),
        a4_ctx_drift_strength=float(max(0.0, min(1.0, a4_ctx_drift_strength))),
        a4_conf_jitter=float(max(0.0, min(1.0, a4_conf_jitter))),
        a4_time_jitter_slots=max(0, a4_time_jitter_slots),
        a4_recent_windows=max(1, a4_recent_windows),
    )


class AttackGenerator:
    def __init__(self, plan: AttackPlan) -> None:
        self.plan = plan

    def apply(
        self,
        batch: EvidenceBatch,
        history: VerificationHistory,
        rng: np.random.Generator,
    ) -> Tuple[EvidenceBatch, Dict[str, Any], int]:
        if not self.plan.active(batch.window_id):
            return batch, {}, 0

        attack_id = self.plan.attack_id
        if attack_id == "A1":
            return self._a1_mass_spoofing(batch, rng)
        if attack_id == "A2":
            return self._a2_slowburn(batch, rng)
        if attack_id == "A3":
            return self._a3_replay(batch, history, rng)
        if attack_id == "A4":
            return self._a4_time_shifted_replay(batch, history, rng)
        if attack_id == "A4P":
            return self._a4p_near_replay(batch, history, rng)
        if attack_id == "A5":
            return self._a5_partition_capture(batch, history, rng)
        return batch, {}, 0

    def _select_indices(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=int)
        k = max(1, int(round(n * self.plan.rho)))
        k = min(k, n)
        return np.sort(rng.choice(np.arange(n), size=k, replace=False))

    def _mutate_topk(self, topk: Tuple[str, ...], rng: np.random.Generator) -> Tuple[str, ...]:
        tokens = list(topk)
        if not tokens:
            return topk
        k = len(tokens)
        n_mutate = max(1, int(round(k * self.plan.a4_p_topk_mutate)))
        n_mutate = min(k, n_mutate)
        indices = np.sort(rng.choice(np.arange(k), size=n_mutate, replace=False))
        pool = list(FEATURE_COLUMNS)
        for idx in indices:
            candidates = [p for p in pool if p not in tokens or p == tokens[idx]]
            if not candidates:
                continue
            tokens[idx] = str(candidates[int(rng.integers(0, len(candidates)))])
        if k >= 2 and rng.random() < 0.35:
            i, j = rng.choice(np.arange(k), size=2, replace=False)
            tokens[int(i)], tokens[int(j)] = tokens[int(j)], tokens[int(i)]
        return tuple(tokens)

    def _a4_candidate_pool(self, batch: EvidenceBatch, history: VerificationHistory) -> list[Evidence]:
        recent_floor = int(batch.window_id) - int(self.plan.a4_recent_windows)
        pool = [
            e
            for e in history.committed_evidences
            if (not e.malicious) and (int(e.window_id) < int(batch.window_id)) and (int(e.window_id) >= recent_floor)
        ]
        if pool:
            return pool
        pool = [e for e in history.committed_evidences if not e.malicious and int(e.window_id) < int(batch.window_id)]
        if pool:
            return pool
        return list(history.committed_evidences)

    def _make_evidence(
        self,
        base: Evidence,
        *,
        attack_label: str,
        window_id: int,
        node_id: str,
        event_id: str,
        timestamp_ms: int,
        anomaly_score: float,
        confidence: float,
        explanation_topk: Tuple[str, ...],
        context_hash: str,
    ) -> Evidence:
        return replace(
            base,
            event_id=event_id,
            window_id=window_id,
            node_id=node_id,
            decision="normal",
            anomaly_score=clamp01(anomaly_score),
            confidence=clamp01(confidence),
            uncertainty=clamp01(1.0 - confidence),
            explanation_topk=tuple(explanation_topk),
            explanation_hash=explanation_hash(explanation_topk),
            context_hash=context_hash,
            timestamp_ms=int(timestamp_ms),
            malicious=True,
            attack_label=attack_label,
        )

    def _a1_mass_spoofing(
        self,
        batch: EvidenceBatch,
        rng: np.random.Generator,
    ) -> Tuple[EvidenceBatch, Dict[str, Any], int]:
        evidences = list(batch.evidences)
        idx = self._select_indices(len(evidences), rng)
        mal_count = 0
        for i in idx:
            base = evidences[i]
            topk = ("burst_density", "route_shift", "id_swap")
            forged = self._make_evidence(
                base,
                attack_label="A1",
                window_id=batch.window_id,
                node_id=base.node_id,
                event_id=stable_hash({"attack": "A1", "wid": batch.window_id, "node": base.node_id}),
                timestamp_ms=base.timestamp_ms,
                anomaly_score=0.82 + float(rng.normal(0.0, 0.03)),
                confidence=0.92 + float(rng.normal(0.0, 0.02)),
                explanation_topk=topk,
                context_hash=stable_hash({"attack": "A1", "wid": batch.window_id, "node": base.node_id}),
            )
            evidences[i] = forged
            mal_count += 1
        return replace(batch, evidences=evidences), {}, mal_count

    def _a2_slowburn(
        self,
        batch: EvidenceBatch,
        rng: np.random.Generator,
    ) -> Tuple[EvidenceBatch, Dict[str, Any], int]:
        evidences = list(batch.evidences)
        idx = self._select_indices(len(evidences), rng)
        mal_count = 0
        for i in idx:
            base = evidences[i]
            forged = self._make_evidence(
                base,
                attack_label="A2",
                window_id=batch.window_id,
                node_id=base.node_id,
                event_id=stable_hash({"attack": "A2", "wid": batch.window_id, "node": base.node_id}),
                timestamp_ms=base.timestamp_ms,
                anomaly_score=0.52 + float(rng.normal(0.0, 0.02)),
                confidence=0.96 + float(rng.normal(0.0, 0.01)),
                explanation_topk=base.explanation_topk,
                context_hash=base.context_hash,
            )
            evidences[i] = forged
            mal_count += 1
        return replace(batch, evidences=evidences), {}, mal_count

    def _a3_replay(
        self,
        batch: EvidenceBatch,
        history: VerificationHistory,
        rng: np.random.Generator,
    ) -> Tuple[EvidenceBatch, Dict[str, Any], int]:
        if not history.committed_evidences:
            return self._a1_mass_spoofing(batch, rng)
        evidences = list(batch.evidences)
        idx = self._select_indices(len(evidences), rng)
        mal_count = 0
        for i in idx:
            replay = history.committed_evidences[int(rng.integers(0, len(history.committed_evidences)))]
            forged = replace(
                replay,
                node_id=evidences[i].node_id,
                malicious=True,
                attack_label="A3",
            )
            evidences[i] = forged
            mal_count += 1
        return replace(batch, evidences=evidences), {}, mal_count

    def _a4_time_shifted_replay(
        self,
        batch: EvidenceBatch,
        history: VerificationHistory,
        rng: np.random.Generator,
    ) -> Tuple[EvidenceBatch, Dict[str, Any], int]:
        if not history.committed_evidences:
            return self._a2_slowburn(batch, rng)
        pool = self._a4_candidate_pool(batch, history)
        if not pool:
            return self._a2_slowburn(batch, rng)
        evidences = list(batch.evidences)
        idx = self._select_indices(len(evidences), rng)
        mal_count = 0
        for i in idx:
            replay = pool[int(rng.integers(0, len(pool)))]
            base = evidences[i]
            slot_jitter = int(rng.integers(0, int(self.plan.a4_time_jitter_slots) + 1))
            ts_offset_ms = int(slot_jitter * 1_000)
            timestamp_ms = int(base.timestamp_ms + ts_offset_ms)
            forged = replace(
                replay,
                event_id=stable_hash({"attack": "A4", "wid": batch.window_id, "node": base.node_id}),
                window_id=batch.window_id,
                node_id=base.node_id,
                timestamp_ms=timestamp_ms,
                prev_evidence_hash=base.prev_evidence_hash,
                anomaly_score=float(replay.anomaly_score),
                confidence=float(replay.confidence),
                uncertainty=clamp01(1.0 - float(replay.confidence)),
                explanation_topk=tuple(replay.explanation_topk),
                explanation_hash=str(replay.explanation_hash),
                context_hash=str(replay.context_hash),
                malicious=True,
                attack_label="A4",
            )
            evidences[i] = forged
            mal_count += 1
        return replace(batch, evidences=evidences), {}, mal_count

    def _a4p_near_replay(
        self,
        batch: EvidenceBatch,
        history: VerificationHistory,
        rng: np.random.Generator,
    ) -> Tuple[EvidenceBatch, Dict[str, Any], int]:
        if not history.committed_evidences:
            return self._a2_slowburn(batch, rng)
        pool = self._a4_candidate_pool(batch, history)
        if not pool:
            return self._a2_slowburn(batch, rng)
        evidences = list(batch.evidences)
        idx = self._select_indices(len(evidences), rng)
        level = max(1, min(3, int(self.plan.a4_level)))
        mal_count = 0
        for i in idx:
            replay = pool[int(rng.integers(0, len(pool)))]
            base = evidences[i]
            topk = tuple(replay.explanation_topk)
            confidence = float(replay.confidence)
            anomaly = float(replay.anomaly_score)
            context_hash = str(replay.context_hash)

            if level >= 1:
                topk = self._mutate_topk(topk, rng)
            if level >= 2:
                confidence = clamp01(confidence - float(rng.uniform(0.0, self.plan.a4_conf_jitter)))
                anomaly = clamp01(anomaly + float(rng.normal(0.0, 0.02)))
            if level >= 3:
                if rng.random() < float(self.plan.a4_ctx_drift_strength):
                    context_hash = stable_hash(
                        {
                            "attack": "A4P_partial_context_drift",
                            "replay": replay.context_hash[:16],
                            "base": base.context_hash[:16],
                            "drift": round(float(rng.uniform(0.0, self.plan.a4_ctx_drift_strength)), 4),
                            "window_id": int(batch.window_id),
                            "node_id": base.node_id,
                        }
                    )
                elif rng.random() < 0.5:
                    context_hash = str(base.context_hash)

            slot_jitter = int(rng.integers(0, int(self.plan.a4_time_jitter_slots) + 1))
            ts_offset_ms = int(slot_jitter * 1_000)
            timestamp_ms = int(base.timestamp_ms + ts_offset_ms)
            forged = replace(
                replay,
                event_id=stable_hash({"attack": "A4P", "wid": batch.window_id, "node": base.node_id}),
                window_id=batch.window_id,
                node_id=base.node_id,
                timestamp_ms=timestamp_ms,
                prev_evidence_hash=base.prev_evidence_hash,
                anomaly_score=anomaly,
                confidence=confidence,
                uncertainty=clamp01(1.0 - confidence),
                explanation_topk=tuple(topk),
                explanation_hash=explanation_hash(topk),
                context_hash=context_hash,
                malicious=True,
                attack_label="A4P",
            )
            evidences[i] = forged
            mal_count += 1
        return replace(batch, evidences=evidences), {}, mal_count

    def _a5_partition_capture(
        self,
        batch: EvidenceBatch,
        history: VerificationHistory,
        rng: np.random.Generator,
    ) -> Tuple[EvidenceBatch, Dict[str, Any], int]:
        evidences = list(batch.evidences)
        k_rho = max(1, int(round(len(evidences) * self.plan.rho)))
        k = max(1, min(len(evidences), max(self.plan.capture_k, k_rho)))
        idx = np.sort(rng.choice(np.arange(len(evidences)), size=k, replace=False))
        mal_count = 0
        partition_nodes = []
        for i in idx:
            base = evidences[i]
            topk_base = list(base.explanation_topk) if base.explanation_topk else list(FEATURE_COLUMNS[:3])
            if topk_base:
                swap_idx = int(rng.integers(0, len(topk_base)))
                topk_base[swap_idx] = "partition_route"
            if len(topk_base) >= 2 and rng.random() < 0.5:
                a = int(rng.integers(0, len(topk_base)))
                b = int(rng.integers(0, len(topk_base)))
                topk_base[a], topk_base[b] = topk_base[b], topk_base[a]
            topk = tuple(topk_base[:3])
            if len(topk) < 3:
                topk = tuple((list(topk) + ["camouflage", "quorum_bias", "partition_route"])[:3])

            anomaly_score = clamp01(0.50 + float(rng.normal(0.0, 0.05)))
            confidence = clamp01(0.54 + float(rng.normal(0.0, 0.04)))
            if rng.random() < 0.40:
                context_hash = stable_hash(
                    {"attack": "A5_partial_drift", "wid": batch.window_id, "node": base.node_id}
                )
            else:
                context_hash = base.context_hash
            forged = self._make_evidence(
                base,
                attack_label="A5",
                window_id=batch.window_id,
                node_id=base.node_id,
                event_id=stable_hash({"attack": "A5", "wid": batch.window_id, "node": base.node_id}),
                timestamp_ms=base.timestamp_ms,
                # Keep A5 near MEV decision boundary so StrongQ veto has room to act.
                anomaly_score=anomaly_score,
                confidence=confidence,
                explanation_topk=topk,
                context_hash=context_hash,
            )
            evidences[i] = forged
            mal_count += 1
            partition_nodes.append(base.node_id)

        net_delta = {
            "partition": True,
            "extra_loss_rate": float(self.plan.partition_loss_boost),
            "partition_nodes": partition_nodes,
        }
        return replace(batch, evidences=evidences), net_delta, mal_count
