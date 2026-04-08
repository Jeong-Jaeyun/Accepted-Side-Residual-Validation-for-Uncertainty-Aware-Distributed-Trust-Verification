from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# Canonical S3 verifier implementation.
# Do not maintain legacy standalone/root-level copies of this module.

from qbm.model import (
    clamp01,
    DropType,
    EvidenceBatch,
    VerificationHistory,
    VerifyResult,
    compute_trust_scores,
    explanation_consistency_per_evidence,
    mean_explanation_similarity,
    pairwise_trust_correlation,
    trust_consistency_varnorm,
)
from qbm.verifiers.base import BaseVerifier


DEFAULT_SOFT_WEIGHTS = {
    "corr": 0.35,
    "sim": 0.25,
    "context_ratio": 0.25,
    "trust_mean": 0.15,
}


class S3MEVVerifier(BaseVerifier):
    name = "s3_mev"

    def __init__(
        self,
        quorum: float = 2.0 / 3.0,
        corr_threshold: float = 0.78,
        explanation_threshold: float = 0.50,
        context_consistency_floor: float = 0.67,
        strict_window_sequence: bool = False,
        trust_weights: Mapping[str, float] | None = None,
        soft_weights: Mapping[str, float] | None = None,
        tau: float = 0.72,
        gray_margin: float = 0.05,
        strongq_verifier: Any = None,
        auto_tau_from_a0: bool = True,
        tau_quantile: float = 0.95,
        tau_min_samples: int = 20,
        grayzone_requires_policy_hint: bool = False,
        grayzone_no_policy_action: str = "mev",
        strongq_mode: str = "veto",
        risk_policy_ratio_floor: float = 0.10,
        risk_context_floor: float = 0.90,
        risk_explanation_floor: float = 0.75,
        risk_tau_band: float = 0.00,
    ) -> None:
        super().__init__(
            quorum=quorum,
            strict_window_sequence=strict_window_sequence,
            enable_replay_check=True,
            enable_timestamp_check=True,
        )
        self.corr_threshold = float(corr_threshold)
        self.explanation_threshold = float(explanation_threshold)
        self.context_consistency_floor = float(context_consistency_floor)
        self.trust_weights = dict(trust_weights or {"score": 0.45, "confidence": 0.40, "explanation": 0.15})
        self.soft_weights = dict(soft_weights or DEFAULT_SOFT_WEIGHTS)
        self.tau = float(tau)
        self.gray_margin = float(max(gray_margin, 0.0))
        self.strongq_verifier = strongq_verifier
        self.auto_tau_from_a0 = bool(auto_tau_from_a0)
        self.tau_quantile = clamp01(float(tau_quantile))
        self.tau_min_samples = max(1, int(tau_min_samples))
        self.grayzone_requires_policy_hint = bool(grayzone_requires_policy_hint)
        self.grayzone_no_policy_action = str(grayzone_no_policy_action).strip().lower() or "mev"
        if self.grayzone_no_policy_action not in {"reject", "mev"}:
            self.grayzone_no_policy_action = "mev"
        self.strongq_mode = str(strongq_mode).strip().lower() or "veto"
        if self.strongq_mode not in {"veto", "override"}:
            self.strongq_mode = "veto"
        self.risk_policy_ratio_floor = clamp01(float(risk_policy_ratio_floor))
        self.risk_context_floor = clamp01(float(risk_context_floor))
        self.risk_explanation_floor = clamp01(float(risk_explanation_floor))
        self.risk_tau_band = float(max(risk_tau_band, 0.0))
        self._tau_calibration_scores: list[float] = []

    def compute_signals(self, batch: EvidenceBatch) -> dict[str, Any]:
        anomalies = [float(e.anomaly_score) for e in batch.evidences]
        confidences = [float(e.confidence) for e in batch.evidences]
        uncertainties = [clamp01(1.0 - c) for c in confidences]
        timestamps = [int(e.timestamp_ms) for e in batch.evidences]
        explanations = [e.explanation_topk for e in batch.evidences]
        ecs = explanation_consistency_per_evidence(explanations)

        trust_scores = compute_trust_scores(
            anomaly_scores=anomalies,
            confidences=confidences,
            explanations=explanations,
            weights=self.trust_weights,
        )
        corr = trust_consistency_varnorm(trust_scores)
        corr_pairwise = pairwise_trust_correlation(trust_scores)
        trust_variance = float(np.var(np.asarray(trust_scores, dtype=float))) if trust_scores else 0.0
        sim = mean_explanation_similarity(explanations)
        context_ratio = self.context_consistency_ratio(batch)

        context_counts: dict[str, int] = {}
        for e in batch.evidences:
            context_counts[e.context_hash] = context_counts.get(e.context_hash, 0) + 1
        dominant_context = max(context_counts, key=context_counts.get) if context_counts else ""
        context_match = [1.0 if e.context_hash == dominant_context else 0.0 for e in batch.evidences]

        exp_counts: dict[tuple[str, ...], int] = {}
        for exp in explanations:
            key = tuple(exp)
            exp_counts[key] = exp_counts.get(key, 0) + 1
        dominant_explanation = max(exp_counts, key=exp_counts.get) if exp_counts else tuple()
        explanation_order_match = [1.0 if tuple(exp) == dominant_explanation else 0.0 for exp in explanations]

        conf_arr = np.asarray(confidences, dtype=float) if confidences else np.asarray([0.0], dtype=float)
        anom_arr = np.asarray(anomalies, dtype=float) if anomalies else np.asarray([0.0], dtype=float)
        unc_arr = np.asarray(uncertainties, dtype=float) if uncertainties else np.asarray([0.0], dtype=float)

        policy_ratio = 1.0 if bool(batch.network_state.get("policy_fired_hint", False)) else 0.0
        policy_nodes = [policy_ratio for _ in batch.evidences] or [0.0]
        policy_arr = np.asarray(policy_nodes, dtype=float)

        overlap_vals: list[float] = []
        order_dist_vals: list[float] = []
        ctx_match_vals: list[float] = []
        seq_gap_vals: list[float] = []
        time_gap_vals: list[float] = []
        weight_l1_vals: list[float] = []
        n = len(batch.evidences)
        slot_ms = max(1.0, 5.0 * 60.0 * 1000.0)
        for i in range(n):
            topk_i = tuple(explanations[i])
            pos_i = {k: idx for idx, k in enumerate(topk_i)}
            for j in range(i + 1, n):
                topk_j = tuple(explanations[j])
                pos_j = {k: idx for idx, k in enumerate(topk_j)}

                k = max(len(topk_i), len(topk_j), 1)
                inter = len(set(topk_i) & set(topk_j))
                overlap_vals.append(float(inter / k))

                common = [feat for feat in topk_i if feat in pos_j]
                if not common:
                    order_dist_vals.append(1.0)
                else:
                    denom = max(k - 1, 1)
                    dist = 0.0
                    for feat in common:
                        dist += abs(pos_i.get(feat, 0) - pos_j.get(feat, 0)) / denom
                    order_dist_vals.append(float(dist / max(len(common), 1)))

                ctx_match_vals.append(1.0 if batch.evidences[i].context_hash == batch.evidences[j].context_hash else 0.0)
                # No explicit sequence index in evidence schema yet; keep as normalized zero slack.
                seq_gap_vals.append(0.0)
                time_gap_vals.append(min(1.0, abs(timestamps[i] - timestamps[j]) / slot_ms))
                pair_i = (
                    float(anomalies[i]),
                    float(confidences[i]),
                    float(uncertainties[i]),
                    float(ecs[i]),
                    float(context_match[i]),
                )
                pair_j = (
                    float(anomalies[j]),
                    float(confidences[j]),
                    float(uncertainties[j]),
                    float(ecs[j]),
                    float(context_match[j]),
                )
                weight_l1_vals.append(
                    float(
                        np.mean(
                            np.abs(
                                np.asarray(pair_i, dtype=float) - np.asarray(pair_j, dtype=float)
                            )
                        )
                    )
                )

        if not overlap_vals:
            overlap_vals = [1.0]
        if not order_dist_vals:
            order_dist_vals = [0.0]
        if not ctx_match_vals:
            ctx_match_vals = [1.0]
        if not seq_gap_vals:
            seq_gap_vals = [0.0]
        if not time_gap_vals:
            time_gap_vals = [0.0]
        if not weight_l1_vals:
            weight_l1_vals = [0.0]

        overlap_arr = np.asarray(overlap_vals, dtype=float)
        order_dist_arr = np.asarray(order_dist_vals, dtype=float)
        ctx_match_arr = np.asarray(ctx_match_vals, dtype=float)
        seq_gap_arr = np.asarray(seq_gap_vals, dtype=float)
        time_gap_arr = np.asarray(time_gap_vals, dtype=float)
        weight_l1_arr = np.asarray(weight_l1_vals, dtype=float)

        score_conf_gap = [abs(float(s) - float(c)) for s, c in zip(anomalies, confidences)]
        explanation_mismatch = [1.0 - float(v) for v in ecs]
        context_drift = [1.0 - float(v) for v in context_match]
        order_drift = [1.0 - float(v) for v in explanation_order_match]
        strongq_base = [
            float(np.mean(conf_arr)),
            float(np.std(conf_arr)),
            float(np.min(conf_arr)),
            float(np.max(conf_arr)),
            float(np.mean(anom_arr)),
            float(np.std(anom_arr)),
            float(np.mean(unc_arr)),
            float(np.std(unc_arr)),
            float(np.mean(overlap_arr)),
            float(np.min(overlap_arr)),
            float(np.mean(order_dist_arr)),
            float(np.max(order_dist_arr)),
            float(np.mean(ctx_match_arr)),
            float(np.max(seq_gap_arr)),
            float(np.max(time_gap_arr)),
            float(np.mean(weight_l1_arr)),
            float(np.mean(policy_arr)),
        ]
        feature_mean = float(np.mean(strongq_base)) if strongq_base else 0.0
        feature_std = float(np.std(strongq_base)) if strongq_base else 0.0
        policy_hint = float(policy_ratio)

        return {
            "corr": float(corr),
            "corr_pairwise": float(corr_pairwise),
            "trust_variance": float(trust_variance),
            "sim": float(sim),
            "context_ratio": float(context_ratio),
            "trust_mean": float(sum(trust_scores) / max(len(trust_scores), 1)),
            "anomaly_mean": float(np.mean(anomalies)) if anomalies else 0.0,
            "confidence_mean": float(np.mean(confidences)) if confidences else 0.0,
            "uncertainty_mean": float(np.mean(uncertainties)) if uncertainties else 0.0,
            "explanation_overlap_mean": float(np.mean(ecs)) if ecs else 0.0,
            "explanation_order_match_mean": float(np.mean(explanation_order_match)) if explanation_order_match else 0.0,
            "score_conf_gap_mean": float(np.mean(score_conf_gap)) if score_conf_gap else 0.0,
            "context_drift_mean": float(np.mean(context_drift)) if context_drift else 0.0,
            "order_drift_mean": float(np.mean(order_drift)) if order_drift else 0.0,
            "context_match_mean": float(np.mean(context_match)) if context_match else 0.0,
            "node_conf_mean": float(np.mean(conf_arr)),
            "node_conf_std": float(np.std(conf_arr)),
            "node_conf_min": float(np.min(conf_arr)),
            "node_conf_max": float(np.max(conf_arr)),
            "node_anom_mean": float(np.mean(anom_arr)),
            "node_anom_std": float(np.std(anom_arr)),
            "node_unc_mean": float(np.mean(unc_arr)),
            "node_unc_std": float(np.std(unc_arr)),
            "pair_overlap_mean": float(np.mean(overlap_arr)),
            "pair_overlap_min": float(np.min(overlap_arr)),
            "pair_orderdist_mean": float(np.mean(order_dist_arr)),
            "pair_orderdist_max": float(np.max(order_dist_arr)),
            "pair_ctxmatch_mean": float(np.mean(ctx_match_arr)),
            "pair_seqgap_max": float(np.max(seq_gap_arr)),
            "pair_timegap_max": float(np.max(time_gap_arr)),
            "pair_weightl1_mean": float(np.mean(weight_l1_arr)),
            "policy_fired_ratio": float(np.mean(policy_arr)),
            "policy_fired_hint": float(policy_hint),
            "strongq_feature_mean": feature_mean,
            "strongq_feature_std": feature_std,
            "strongq_feature_dim": float(len(strongq_base)),
            "strongq_feature_vector_base": strongq_base,
        }

    def compute_soft_score(self, signals: Mapping[str, Any]) -> tuple[float, float, float]:
        w = dict(self.soft_weights)
        w_corr = float(w.get("corr", DEFAULT_SOFT_WEIGHTS["corr"]))
        w_sim = float(w.get("sim", DEFAULT_SOFT_WEIGHTS["sim"]))
        w_ctx = float(w.get("context_ratio", DEFAULT_SOFT_WEIGHTS["context_ratio"]))
        w_trust = float(w.get("trust_mean", DEFAULT_SOFT_WEIGHTS["trust_mean"]))
        w_sum = w_corr + w_sim + w_ctx + w_trust
        if w_sum <= 0:
            w_corr, w_sim, w_ctx, w_trust = (
                DEFAULT_SOFT_WEIGHTS["corr"],
                DEFAULT_SOFT_WEIGHTS["sim"],
                DEFAULT_SOFT_WEIGHTS["context_ratio"],
                DEFAULT_SOFT_WEIGHTS["trust_mean"],
            )
            w_sum = 1.0

        corr_scaled = clamp01(float(signals.get("corr", 0.0)) / max(self.corr_threshold, 1.0e-9))
        sim_scaled = clamp01(float(signals.get("sim", 0.0)) / max(self.explanation_threshold, 1.0e-9))
        context_ratio = clamp01(float(signals.get("context_ratio", 0.0)))
        trust_mean = clamp01(float(signals.get("trust_mean", 0.0)))

        score = (
            (w_corr * corr_scaled)
            + (w_sim * sim_scaled)
            + (w_ctx * context_ratio)
            + (w_trust * trust_mean)
        ) / w_sum
        return clamp01(score), corr_scaled, sim_scaled

    @staticmethod
    def _with_decision_defaults(signals: dict[str, Any], mev_decision: float) -> None:
        signals["mev_decision"] = float(mev_decision)
        signals["final_decision"] = float(mev_decision)
        signals.setdefault("decision_stage", "base")
        signals.setdefault("reject_code", "")
        signals.setdefault("strongq_called", 0.0)
        signals.setdefault("strongq_agree", 0.0)
        signals.setdefault("strongq_disagree", 0.0)
        signals.setdefault("flip_mev_to_reject", 0.0)
        signals.setdefault("flip_mev_to_accept", 0.0)
        signals.setdefault("strongq_gate_reason", "")

    def _risk_flags(self, batch: EvidenceBatch, signals: Mapping[str, Any], soft_score: float) -> dict[str, Any]:
        policy_ratio = clamp01(float(signals.get("policy_fired_ratio", signals.get("policy_fired_hint", 0.0))))
        policy_hint = bool(policy_ratio > self.risk_policy_ratio_floor)
        context_low = float(signals.get("context_ratio", 1.0)) < self.risk_context_floor
        sim_low = float(signals.get("sim", 1.0)) < self.risk_explanation_floor
        tau_near = abs(float(soft_score) - float(self.tau)) < self.risk_tau_band

        risk_hint = bool(policy_hint or context_low or sim_low or tau_near)
        if self.grayzone_requires_policy_hint and not policy_hint:
            risk_hint = False
        reasons = []
        if policy_hint:
            reasons.append("policy_ratio")
        if context_low:
            reasons.append("context")
        if sim_low:
            reasons.append("sim")
        if tau_near:
            reasons.append("tau_band")

        return {
            "policy_hint": 1.0 if policy_hint else 0.0,
            "policy_ratio": float(policy_ratio),
            "risk_hint": 1.0 if risk_hint else 0.0,
            "risk_context_low": 1.0 if context_low else 0.0,
            "risk_sim_low": 1.0 if sim_low else 0.0,
            "risk_tau_near": 1.0 if tau_near else 0.0,
            "risk_reasons": "|".join(reasons),
        }

    def should_call_strongq(self, batch: EvidenceBatch, signals: Mapping[str, Any], soft_score: float) -> bool:
        flags = self._risk_flags(batch, signals, soft_score)
        return bool(flags["risk_hint"] > 0.5)

    def build_strongq_feature_vector(self, signals: Mapping[str, Any], soft_score: float, gray_flag: float) -> list[float]:
        base = signals.get("strongq_feature_vector_base")
        if isinstance(base, (list, tuple)):
            vec = [float(v) for v in base]
        else:
            vec = [
                float(signals.get("node_conf_mean", 0.0)),
                float(signals.get("node_conf_std", 0.0)),
                float(signals.get("node_conf_min", 0.0)),
                float(signals.get("node_conf_max", 0.0)),
                float(signals.get("node_anom_mean", 0.0)),
                float(signals.get("node_anom_std", 0.0)),
                float(signals.get("node_unc_mean", 0.0)),
                float(signals.get("node_unc_std", 0.0)),
                float(signals.get("pair_overlap_mean", 1.0)),
                float(signals.get("pair_overlap_min", 1.0)),
                float(signals.get("pair_orderdist_mean", 0.0)),
                float(signals.get("pair_orderdist_max", 0.0)),
                float(signals.get("pair_ctxmatch_mean", 1.0)),
                float(signals.get("pair_seqgap_max", 0.0)),
                float(signals.get("pair_timegap_max", 0.0)),
                float(signals.get("pair_weightl1_mean", 0.0)),
                float(signals.get("policy_fired_ratio", 0.0)),
            ]
        vec.extend(
            [
                float(soft_score),
                float(abs(float(soft_score) - float(self.tau))),
                float(gray_flag),
                float(signals.get("policy_fired_ratio", 0.0)),
            ]
        )
        return vec

    def _resolve_gray_zone(
        self,
        *,
        batch: EvidenceBatch,
        history: VerificationHistory,
        signals: dict[str, Any],
        soft_score: float,
    ) -> VerifyResult:
        mev_decision = 1.0 if float(soft_score) >= float(self.tau) else 0.0
        self._with_decision_defaults(signals, mev_decision)
        signals["gray_zone_flag"] = 1.0

        risk = self._risk_flags(batch, signals, soft_score)
        signals["policy_fired_hint"] = float(risk["policy_hint"])
        signals["policy_ratio"] = float(risk.get("policy_ratio", signals.get("policy_fired_ratio", 0.0)))
        signals["risk_hint"] = float(risk["risk_hint"])
        signals["risk_context_low"] = float(risk["risk_context_low"])
        signals["risk_sim_low"] = float(risk["risk_sim_low"])
        signals["risk_tau_near"] = float(risk["risk_tau_near"])
        signals["risk_reasons"] = str(risk["risk_reasons"])
        signals["risk_policy_ratio_floor"] = float(self.risk_policy_ratio_floor)
        signals["risk_context_floor"] = float(self.risk_context_floor)
        signals["risk_explanation_floor"] = float(self.risk_explanation_floor)
        signals["risk_tau_band"] = float(self.risk_tau_band)

        # Veto mode only needs StrongQ on gray-zone accept candidates.
        if self.strongq_mode == "veto" and mev_decision < 0.5:
            signals["decision_path"] = "gray_zone_mev_reject"
            signals["final_decision"] = 0.0
            signals["decision_stage"] = "base"
            signals["reject_code"] = "BASE_GRAYZONE_REJECT"
            return VerifyResult(False, "gray zone rejected by MEV lower-half", signals, DropType.VERIFICATION)

        should_call = bool(risk["risk_hint"] > 0.5 and self.strongq_verifier is not None)
        if not should_call:
            signals["strongq_called"] = 0.0
            if self.strongq_verifier is None:
                signals["strongq_gate_reason"] = "no_strongq_resolver"
            else:
                signals["strongq_gate_reason"] = "risk_hint_not_fired"
            if self.grayzone_no_policy_action == "reject":
                signals["decision_path"] = "gray_zone_fallback_reject"
                signals["final_decision"] = 0.0
                signals["decision_stage"] = "base"
                signals["reject_code"] = "BASE_GRAYZONE_FALLBACK_REJECT"
                return VerifyResult(False, "gray zone rejected by fallback policy", signals, DropType.VERIFICATION)
            final_accept = bool(mev_decision >= 0.5)
            signals["decision_path"] = "gray_zone_fallback_mev_accept" if final_accept else "gray_zone_fallback_mev_reject"
            signals["final_decision"] = 1.0 if final_accept else 0.0
            signals["decision_stage"] = "base"
            signals["reject_code"] = "" if final_accept else "BASE_GRAYZONE_FALLBACK_REJECT"
            if final_accept:
                return VerifyResult(True, "gray zone resolved by MEV fallback", signals, DropType.NONE)
            return VerifyResult(False, "gray zone rejected by MEV fallback", signals, DropType.VERIFICATION)

        resolver = self.strongq_verifier
        if hasattr(resolver, "verify_from_signals"):
            strongq_result = resolver.verify_from_signals(signals)
        elif hasattr(resolver, "verify"):
            strongq_result = resolver.verify(batch, history)
        else:
            signals["decision_path"] = "gray_zone_invalid_strongq"
            signals["final_decision"] = 0.0
            signals["decision_stage"] = "strongq"
            signals["reject_code"] = "STRONGQ_INVALID_RESOLVER"
            return VerifyResult(False, "invalid strongq resolver", signals, DropType.VERIFICATION)

        merged_scores = dict(signals)
        merged_scores.update(dict(strongq_result.scores))
        strongq_score = merged_scores.get("strongq_score")
        if strongq_score is None:
            strongq_score = merged_scores.get("quantum_score", merged_scores.get("strongq_witness", float("nan")))
        merged_scores["strongq_score"] = float(strongq_score)
        strongq_decision = 1.0 if bool(strongq_result.accept) else 0.0
        merged_scores["strongq_decision"] = float(strongq_decision)
        merged_scores["strongq_called"] = 1.0
        merged_scores["strongq_gate_reason"] = "risk_hint_called"

        if self.strongq_mode == "override":
            final_accept = bool(strongq_decision > 0.5)
        else:
            final_accept = bool((mev_decision > 0.5) and (strongq_decision > 0.5))

        disagree = int(mev_decision != strongq_decision)
        merged_scores["strongq_agree"] = 1.0 - float(disagree)
        merged_scores["strongq_disagree"] = float(disagree)
        merged_scores["flip_mev_to_reject"] = 1.0 if (mev_decision > 0.5 and not final_accept) else 0.0
        merged_scores["flip_mev_to_accept"] = 1.0 if (mev_decision < 0.5 and final_accept) else 0.0
        merged_scores["final_decision"] = 1.0 if final_accept else 0.0

        if final_accept:
            merged_scores["decision_path"] = "gray_zone_strongq_accept"
            merged_scores["decision_stage"] = "strongq"
            merged_scores["reject_code"] = ""
            return VerifyResult(True, "accepted by S3 gray-zone + strongq", merged_scores, DropType.NONE)
        if merged_scores["flip_mev_to_reject"] > 0.5:
            merged_scores["decision_path"] = "gray_zone_strongq_veto_reject"
            merged_scores["reject_code"] = "STRONGQ_VETO"
        else:
            merged_scores["decision_path"] = "gray_zone_strongq_reject"
            merged_scores["reject_code"] = "STRONGQ_REJECT"
        merged_scores["decision_stage"] = "strongq"
        return VerifyResult(False, "rejected by S3 gray-zone + strongq", merged_scores, DropType.VERIFICATION)

    def _calibrate_tau(self, batch: EvidenceBatch, soft_score: float, signals: dict[str, Any]) -> None:
        if not self.auto_tau_from_a0:
            signals["tau_source"] = "config"
            signals["tau_calibrated"] = 0.0
            return
        if not bool(batch.network_state.get("calibration_mode", False)):
            signals["tau_source"] = "config"
            signals["tau_calibrated"] = 0.0
            return
        self._tau_calibration_scores.append(float(soft_score))
        n = len(self._tau_calibration_scores)
        signals["tau_calibration_samples"] = float(n)
        if n < self.tau_min_samples:
            signals["tau_source"] = "warmup"
            signals["tau_calibrated"] = 0.0
            return
        q_tau = float(np.quantile(np.asarray(self._tau_calibration_scores, dtype=float), self.tau_quantile))
        self.tau = clamp01(q_tau)
        signals["tau_source"] = "a0_quantile"
        signals["tau_quantile"] = float(self.tau_quantile)
        signals["tau_calibrated"] = 1.0

    def verify(self, batch: EvidenceBatch, history: VerificationHistory) -> VerifyResult:
        if not batch.evidences:
            scores = {"quorum": 0.0, "decision_path": "reject_empty_batch", "gray_zone_flag": 0.0, "reject_code": "BASE_EMPTY_BATCH"}
            self._with_decision_defaults(scores, 0.0)
            return VerifyResult(False, "empty evidence batch", scores, DropType.VERIFICATION)

        reason = self.replay_check(batch, history)
        if reason:
            scores = {"replay": 0.0, "decision_path": "reject_replay", "gray_zone_flag": 0.0, "reject_code": "BASE_REPLAY_REJECT"}
            self._with_decision_defaults(scores, 0.0)
            return VerifyResult(False, reason, scores, DropType.VERIFICATION)

        reason = self.time_sequence_check(batch, history)
        if reason:
            scores = {"sequence": 0.0, "decision_path": "reject_sequence", "gray_zone_flag": 0.0, "reject_code": "BASE_SEQUENCE_REJECT"}
            self._with_decision_defaults(scores, 0.0)
            return VerifyResult(False, reason, scores, DropType.VERIFICATION)

        required = self.required_quorum(len(batch.evidences))
        if len(batch.evidences) < required:
            scores = {
                "quorum_ratio": len(batch.evidences) / max(required, 1),
                "decision_path": "reject_quorum",
                "gray_zone_flag": 0.0,
                "reject_code": "BASE_QUORUM_REJECT",
            }
            self._with_decision_defaults(scores, 0.0)
            return VerifyResult(
                False,
                f"quorum failed: got {len(batch.evidences)} < required {required}",
                scores,
                DropType.VERIFICATION,
            )

        signals = self.compute_signals(batch)
        if signals["context_ratio"] < self.context_consistency_floor:
            signals["decision_path"] = "reject_context_floor"
            signals["gray_zone_flag"] = 0.0
            signals["reject_code"] = "BASE_CONTEXT_REJECT"
            self._with_decision_defaults(signals, 0.0)
            return VerifyResult(
                False,
                f"context consistency failed: ratio={signals['context_ratio']:.4f}",
                signals,
                DropType.VERIFICATION,
            )
        soft_score, corr_scaled, sim_scaled = self.compute_soft_score(signals)
        self._calibrate_tau(batch, soft_score, signals)
        signals["soft_score"] = float(soft_score)
        signals["corr_scaled"] = float(corr_scaled)
        signals["sim_scaled"] = float(sim_scaled)
        signals["quorum_ratio"] = 1.0
        signals["tau"] = float(self.tau)
        signals["gray_margin"] = float(self.gray_margin)
        signals.setdefault("tau_calibration_samples", float(len(self._tau_calibration_scores)))
        signals.setdefault("tau_quantile", float(self.tau_quantile))
        signals.setdefault("tau_source", "config")
        signals.setdefault("tau_calibrated", 0.0)

        upper = clamp01(self.tau + self.gray_margin)
        lower = clamp01(self.tau - self.gray_margin)
        signals["soft_upper"] = float(upper)
        signals["soft_lower"] = float(lower)
        if soft_score >= upper:
            signals["decision_path"] = "mev_accept"
            signals["gray_zone_flag"] = 0.0
            signals["reject_code"] = ""
            strongq_vec = self.build_strongq_feature_vector(signals, soft_score, gray_flag=0.0)
            signals["strongq_feature_vector"] = strongq_vec
            signals["strongq_feature_dim"] = float(len(strongq_vec))
            signals["strongq_feature_mean"] = float(np.mean(np.asarray(strongq_vec, dtype=float)))
            signals["strongq_feature_std"] = float(np.std(np.asarray(strongq_vec, dtype=float)))
            self._with_decision_defaults(signals, 1.0)
            return VerifyResult(True, "accepted by S3 soft score", signals, DropType.NONE)
        if soft_score <= lower:
            signals["decision_path"] = "mev_reject"
            signals["gray_zone_flag"] = 0.0
            signals["reject_code"] = "BASE_S3_SCORE_REJECT"
            strongq_vec = self.build_strongq_feature_vector(signals, soft_score, gray_flag=0.0)
            signals["strongq_feature_vector"] = strongq_vec
            signals["strongq_feature_dim"] = float(len(strongq_vec))
            signals["strongq_feature_mean"] = float(np.mean(np.asarray(strongq_vec, dtype=float)))
            signals["strongq_feature_std"] = float(np.std(np.asarray(strongq_vec, dtype=float)))
            self._with_decision_defaults(signals, 0.0)
            return VerifyResult(False, "rejected by S3 soft score", signals, DropType.VERIFICATION)

        strongq_vec = self.build_strongq_feature_vector(signals, soft_score, gray_flag=1.0)
        signals["strongq_feature_vector"] = strongq_vec
        signals["strongq_feature_dim"] = float(len(strongq_vec))
        signals["strongq_feature_mean"] = float(np.mean(np.asarray(strongq_vec, dtype=float)))
        signals["strongq_feature_std"] = float(np.std(np.asarray(strongq_vec, dtype=float)))
        return self._resolve_gray_zone(batch=batch, history=history, signals=signals, soft_score=soft_score)
