from __future__ import annotations

from typing import Any, Mapping

from qbm.backend.qpu_sim import strongq_witness_stats
from qbm.model import DropType, EvidenceBatch, VerificationHistory, VerifyResult
from qbm.verifiers.s3_mev import S3MEVVerifier


class StrongQVerifier(S3MEVVerifier):
    name = "strongq_verifier"

    def __init__(
        self,
        witness_threshold: float = 0.58,
        shots: int = 4096,
        ci_gate: bool = False,
        ci_z_score: float = 1.96,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.witness_threshold = float(witness_threshold)
        self.shots = int(shots)
        self.ci_gate = bool(ci_gate)
        self.ci_z_score = float(ci_z_score)
        if getattr(self, "strongq_verifier", None) is None:
            # Allow standalone strongQ mode to resolve its own S3 gray-zone path.
            self.strongq_verifier = self

    @staticmethod
    def _feature_vector_from_signals(signals: Mapping[str, Any]) -> list[float]:
        vector = signals.get("strongq_feature_vector")
        if isinstance(vector, (list, tuple)):
            return [float(v) for v in vector]
        return [
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
            float(signals.get("soft_score", 0.0)),
            float(abs(float(signals.get("soft_score", 0.0)) - float(signals.get("tau", 0.0)))),
            float(signals.get("gray_zone_flag", 0.0)),
            float(signals.get("policy_fired_ratio", 0.0)),
        ]

    def score_from_signals(self, signals: Mapping[str, Any]) -> dict[str, float]:
        feature_vector = self._feature_vector_from_signals(signals)
        stats = strongq_witness_stats(
            trust_mean=float(signals.get("trust_mean", 0.0)),
            corr=float(signals.get("corr", 0.0)),
            sim=float(signals.get("sim", 0.0)),
            shots=self.shots,
            z_score=self.ci_z_score,
            feature_vector=feature_vector,
        )
        return stats

    def verify_from_signals(self, signals: Mapping[str, Any]) -> VerifyResult:
        stats = self.score_from_signals(signals)
        witness = float(stats["witness"])
        scores = dict(signals)
        scores["strongq_witness"] = witness
        scores["strongq_score"] = float(stats.get("quantum_score", witness))
        scores["quantum_score"] = float(stats.get("quantum_score", witness))
        scores["strongq_shot_std"] = float(stats["shot_std"])
        scores["strongq_ci_low"] = float(stats["ci_low"])
        scores["strongq_ci_high"] = float(stats["ci_high"])
        scores["strongq_shots"] = float(stats["shots"])
        scores["strongq_feature_mean"] = float(stats.get("feature_mean", 0.0))
        scores["strongq_feature_std"] = float(stats.get("feature_std", 0.0))
        scores["strongq_feature_dim"] = float(stats.get("feature_dim", 0.0))
        scores["decision_stage"] = "strongq"

        if witness >= self.witness_threshold:
            scores["strongq_decision"] = 0.0
            scores["decision_path"] = "strongq_veto"
            scores["reject_code"] = "STRONGQ_VETO"
            return VerifyResult(
                False,
                f"strongQ veto triggered: witness={witness:.4f}",
                scores,
                DropType.VERIFICATION,
            )
        if self.ci_gate and float(stats["ci_high"]) >= self.witness_threshold:
            scores["strongq_decision"] = 0.0
            scores["decision_path"] = "strongq_ci_veto"
            scores["reject_code"] = "STRONGQ_CI_VETO"
            return VerifyResult(
                False,
                f"strongQ confidence bound veto: ci_high={stats['ci_high']:.4f}",
                scores,
                DropType.VERIFICATION,
            )
        scores["strongq_decision"] = 1.0
        scores["decision_path"] = "strongq_accept"
        scores["reject_code"] = ""
        return VerifyResult(True, "accepted by strongQ verifier", scores, DropType.NONE)

    def verify(self, batch: EvidenceBatch, history: VerificationHistory) -> VerifyResult:
        # Keep StrongQ strictly as gray-zone resolver (or standalone S3 path)
        # to avoid unconditional extra verification on high-confidence MEV accepts.
        return super().verify(batch, history)
