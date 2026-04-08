from __future__ import annotations

import math
from typing import Any, Mapping

from qbm.backend.qiskit_qbm import ResidualForensicEnergyValidator
from qbm.model import DropType, EvidenceBatch, VerificationHistory, VerifyResult, clamp01
from qbm.verifiers.s3_mev import S3MEVVerifier


class QBMVerifier(S3MEVVerifier):
    name = "qbm_verifier"

    def __init__(
        self,
        qbm_threshold: float = 0.55,
        shots: int = 2048,
        qiskit_config: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.qbm_threshold = float(qbm_threshold)
        self.shots = int(shots)
        self.backend_label = "qiskit_residual_forensic_energy"
        self.qiskit_config = dict(qiskit_config or {})
        self._scorer = ResidualForensicEnergyValidator(self.qiskit_config, shots=self.shots)

    def apply_calibration(self, artifact: Mapping[str, Any]) -> None:
        if not isinstance(artifact, Mapping):
            return
        if hasattr(self._scorer, "apply_calibration"):
            self._scorer.apply_calibration(artifact)
        if "qbm_threshold" in artifact:
            self.qbm_threshold = float(artifact.get("qbm_threshold", self.qbm_threshold))

    def export_calibration(self) -> dict[str, Any]:
        artifact = {}
        if hasattr(self._scorer, "export_calibration"):
            artifact.update(dict(self._scorer.export_calibration()))
        artifact["qbm_threshold"] = float(self.qbm_threshold)
        return artifact

    def _score_signals(self, signals: Mapping[str, Any]) -> dict[str, float | str]:
        return self._scorer.score(signals)

    def can_shadow_score(self, signals: Mapping[str, Any]) -> bool:
        return bool(getattr(self._scorer, "can_score")(signals))

    def shadow_score_from_signals(self, signals: Mapping[str, Any]) -> dict[str, float | str]:
        if not self.can_shadow_score(signals):
            return {}
        scores = dict(self._score_signals(signals))
        shadow_score = float(scores.get("q_score", float("nan")))
        scores["q_score_shadow"] = float(shadow_score)
        scores["qbm_shadow_available"] = 1.0
        scores["qbm_invoked_by_gray_zone"] = float(clamp01(scores.get("gray_zone_flag", 0.0)))
        scores["qbm_risk_hint_level"] = float(clamp01(scores.get("risk_hint", 0.0)))
        scores.setdefault("qbm_stage2_eligible", 0.0)
        scores.setdefault("qbm_stage2_veto", 0.0)
        scores.setdefault("qbm_stage2_decision", float("nan"))
        scores.setdefault("qbm_threshold", float(self.qbm_threshold))
        return scores

    def shadow_score_batch(self, batch: EvidenceBatch) -> dict[str, float | str]:
        if not batch.evidences:
            return {}
        signals = dict(self.compute_signals(batch))
        soft_score, corr_scaled, sim_scaled = self.compute_soft_score(signals)
        upper = clamp01(self.tau + self.gray_margin)
        lower = clamp01(self.tau - self.gray_margin)
        risk = self._risk_flags(batch, signals, soft_score)
        signals["soft_score"] = float(soft_score)
        signals["corr_scaled"] = float(corr_scaled)
        signals["sim_scaled"] = float(sim_scaled)
        signals["tau"] = float(self.tau)
        signals["gray_margin"] = float(self.gray_margin)
        signals["gray_zone_flag"] = 1.0 if (lower < soft_score < upper) else 0.0
        signals["policy_fired_hint"] = float(risk["policy_hint"])
        signals["policy_ratio"] = float(risk.get("policy_ratio", signals.get("policy_fired_ratio", 0.0)))
        signals["risk_hint"] = float(risk["risk_hint"])
        signals["risk_context_low"] = float(risk["risk_context_low"])
        signals["risk_sim_low"] = float(risk["risk_sim_low"])
        signals["risk_tau_near"] = float(risk["risk_tau_near"])
        signals["risk_reasons"] = str(risk["risk_reasons"])
        shadow = dict(self.shadow_score_from_signals(signals))
        for key, value in signals.items():
            shadow.setdefault(key, value)
        shadow.pop("q_score", None)
        return shadow

    def verify(self, batch: EvidenceBatch, history: VerificationHistory) -> VerifyResult:
        base = super().verify(batch, history)
        scores = dict(base.scores)
        scores.setdefault("base_accept", 1.0 if base.accept else 0.0)
        scores.setdefault("base_decision_path", str(scores.get("decision_path", "")))
        scores.setdefault("decision_stage", "base")
        scores.setdefault("reject_code", "" if base.accept else "BASE_REJECT")

        shadow_scores = self.shadow_score_from_signals(scores)
        if shadow_scores:
            scores.update(shadow_scores)
        else:
            scores.setdefault("q_score_shadow", float("nan"))
            scores.setdefault("qbm_shadow_available", 0.0)
            scores.setdefault("qbm_invoked_by_gray_zone", float(clamp01(scores.get("gray_zone_flag", 0.0))))
            scores.setdefault("qbm_risk_hint_level", float(clamp01(scores.get("risk_hint", 0.0))))
        scores.setdefault("qbm_stage2_eligible", 0.0)
        scores.setdefault("qbm_stage2_veto", 0.0)
        scores.setdefault("qbm_stage2_decision", float("nan"))
        scores.setdefault("qbm_invoked_by_base_accept", 1.0 if base.accept else 0.0)
        scores.setdefault("qbm_threshold", float(self.qbm_threshold))

        if not base.accept:
            scores["q_score"] = float("nan")
            scores["decision_stage"] = "base"
            return VerifyResult(base.accept, base.reason, scores, base.drop_type)

        q_score_shadow = float(scores.get("q_score_shadow", float("nan")))
        if not math.isfinite(q_score_shadow):
            scores["decision_stage"] = "qbm_stage2"
            scores["decision_path"] = "qbm_stage2_shadow_unavailable_accept"
            return VerifyResult(
                True,
                "accepted by residual forensic validator (shadow score unavailable)",
                scores,
                DropType.NONE,
            )

        scores["qbm_stage2_eligible"] = 1.0
        scores["qbm_stage2_decision"] = 1.0
        scores["q_score"] = float(q_score_shadow)
        scores["decision_stage"] = "qbm_stage2"
        if q_score_shadow < self.qbm_threshold:
            scores["qbm_stage2_veto"] = 1.0
            scores["qbm_stage2_decision"] = 0.0
            scores["decision_path"] = "qbm_stage2_veto"
            scores["reject_code"] = "QBM_STAGE2_VETO"
            return VerifyResult(
                False,
                f"residual forensic validator veto ({self.backend_label}): q_score={q_score_shadow:.4f}",
                scores,
                DropType.VERIFICATION,
            )
        scores["decision_path"] = "qbm_stage2_accept"
        scores["reject_code"] = ""
        return VerifyResult(True, "accepted by residual forensic validator", scores, DropType.NONE)
