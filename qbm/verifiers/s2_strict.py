from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from qbm.model import (
    FEATURE_COLUMNS,
    DropType,
    EvidenceBatch,
    VerificationHistory,
    VerifyResult,
    compute_trust_scores,
)
from qbm.verifiers.base import BaseVerifier


class S2StrictVerifier(BaseVerifier):
    name = "s2_strict"

    def __init__(
        self,
        quorum: float = 1.0,
        confidence_floor: float = 0.70,
        variance_tol: float = 1.0e-8,
        context_consistency_floor: float = 1.0,
        strict_window_sequence: bool = True,
        trust_weights: Mapping[str, float] | None = None,
        explanation_feature_whitelist: Sequence[str] | None = None,
        expected_topk_len: int = 3,
        enforce_explanation_schema: bool = False,
    ) -> None:
        super().__init__(
            quorum=quorum,
            strict_window_sequence=strict_window_sequence,
            enable_replay_check=True,
            enable_timestamp_check=True,
        )
        self.confidence_floor = float(confidence_floor)
        self.variance_tol = float(variance_tol)
        self.context_consistency_floor = float(context_consistency_floor)
        self.trust_weights = dict(trust_weights or {"score": 0.45, "confidence": 0.40, "explanation": 0.15})
        self.explanation_feature_whitelist = set(explanation_feature_whitelist or FEATURE_COLUMNS)
        self.expected_topk_len = int(expected_topk_len)
        self.enforce_explanation_schema = bool(enforce_explanation_schema)

    def explanation_diagnostics(self, batch: EvidenceBatch) -> dict[str, float]:
        explanations = [tuple(e.explanation_topk) for e in batch.evidences]
        hashes = [e.explanation_hash for e in batch.evidences]
        if not explanations:
            return {
                "explanation_exact_match": 1.0,
                "explanation_order_match": 1.0,
                "explanation_length_match": 1.0,
                "explanation_schema_whitelist_match": 1.0,
                "explanation_schema_violations": 0.0,
                "explanation_expected_topk_len": float(max(self.expected_topk_len, 0)),
            }

        ref = explanations[0]
        lengths = [len(exp) for exp in explanations]
        expected_len = self.expected_topk_len if self.expected_topk_len > 0 else lengths[0]
        schema_violations = 0
        for exp in explanations:
            for feature_name in exp:
                if feature_name not in self.explanation_feature_whitelist:
                    schema_violations += 1

        return {
            # Existing deterministic gate: explanation hash exact match.
            "explanation_exact_match": 1.0 if len(set(hashes)) == 1 else 0.0,
            # Extra audit fields for reviewer-facing strict-format diagnostics.
            "explanation_order_match": 1.0 if all(exp == ref for exp in explanations) else 0.0,
            "explanation_length_match": 1.0 if all(length == expected_len for length in lengths) else 0.0,
            "explanation_schema_whitelist_match": 1.0 if schema_violations == 0 else 0.0,
            "explanation_schema_violations": float(schema_violations),
            "explanation_expected_topk_len": float(expected_len),
        }

    def verify(self, batch: EvidenceBatch, history: VerificationHistory) -> VerifyResult:
        if not batch.evidences:
            return VerifyResult(False, "empty evidence batch", {"quorum": 0.0}, DropType.VERIFICATION)

        reason = self.replay_check(batch, history)
        if reason:
            return VerifyResult(False, reason, {"replay": 0.0}, DropType.VERIFICATION)

        reason = self.time_sequence_check(batch, history)
        if reason:
            return VerifyResult(False, reason, {"sequence": 0.0}, DropType.VERIFICATION)

        required = self.required_quorum(len(batch.evidences))
        if len(batch.evidences) < required:
            return VerifyResult(
                False,
                f"quorum failed: got {len(batch.evidences)} < required {required}",
                {"quorum_ratio": len(batch.evidences) / max(required, 1)},
                DropType.VERIFICATION,
            )

        confidences = [float(e.confidence) for e in batch.evidences]
        conf_min = float(min(confidences))
        if conf_min < self.confidence_floor:
            return VerifyResult(
                False,
                f"hard confidence floor failed: min={conf_min:.4f}",
                {"conf_min": conf_min},
                DropType.VERIFICATION,
            )

        context_ratio = self.context_consistency_ratio(batch)
        if context_ratio < self.context_consistency_floor:
            return VerifyResult(
                False,
                f"context consistency failed: ratio={context_ratio:.4f}",
                {"context_ratio": context_ratio},
                DropType.VERIFICATION,
            )

        explanation_scores = self.explanation_diagnostics(batch)
        if explanation_scores["explanation_exact_match"] < 1.0:
            return VerifyResult(
                False,
                "deterministic explanation format check failed",
                explanation_scores,
                DropType.VERIFICATION,
            )
        if self.enforce_explanation_schema and explanation_scores["explanation_length_match"] < 1.0:
            return VerifyResult(
                False,
                "deterministic explanation length check failed",
                explanation_scores,
                DropType.VERIFICATION,
            )
        if self.enforce_explanation_schema and explanation_scores["explanation_schema_whitelist_match"] < 1.0:
            return VerifyResult(
                False,
                "deterministic explanation schema whitelist failed",
                explanation_scores,
                DropType.VERIFICATION,
            )

        trust_scores = compute_trust_scores(
            anomaly_scores=[float(e.anomaly_score) for e in batch.evidences],
            confidences=confidences,
            explanations=[e.explanation_topk for e in batch.evidences],
            weights=self.trust_weights,
        )
        variance = float(np.var(trust_scores))
        if variance > self.variance_tol:
            return VerifyResult(
                False,
                f"zero-variance (within numerical tolerance) failed: var={variance:.12f}",
                {"trust_variance": variance, "variance_tol": self.variance_tol},
                DropType.VERIFICATION,
            )

        return VerifyResult(
            True,
            "accepted by S2 strict gate",
            {
                "quorum_ratio": 1.0,
                "conf_min": conf_min,
                "trust_variance": variance,
                "variance_tol": self.variance_tol,
                "context_ratio": context_ratio,
                **explanation_scores,
            },
            DropType.NONE,
        )
