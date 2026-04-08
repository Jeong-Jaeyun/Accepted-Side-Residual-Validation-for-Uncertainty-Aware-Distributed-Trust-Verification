from __future__ import annotations

"""
Residual forensic feature schema for the Qiskit-backed stage-2 validator.

The raw residual terms are built from verifier signals, then normalized so that
all encoded visible inputs follow one invariant:

  larger encoded values => stronger forensic inconsistency / more suspicious

This is the contract consumed by the shallow residual-energy circuit in
`qbm/backend/qiskit_qbm.py`.
"""

from typing import Any, Mapping

from qbm.model import clamp01


FORENSIC_FEATURE_SCHEMA_VERSION = "residual_v2"
FORENSIC_REFERENCE_MODE = "global_benign_reference"
FORENSIC_EXPECTED_CONTEXT_MODE = "benign_only_linear_reference"
FORENSIC_DIRECTIONALITY_MODE = "normalized_high_is_suspicious"

FORENSIC_FEATURE_KEYS: tuple[str, ...] = (
    "r_trust_sim_gap",
    "r_overlap_spread",
    "r_context_dev",
    "r_policy_gray_contra",
    "r_trust_sim_prod",
    "r_benign_delta",
    "r_temporal_inconsistency",
    "r_witness_disagree",
)

MINIMAL_SIGNAL_KEYS: tuple[str, ...] = (
    "trust_mean",
    "sim",
    "context_ratio",
    "pair_overlap_mean",
    "pair_overlap_min",
)

OPTIONAL_SIGNAL_KEYS: tuple[str, ...] = (
    "corr",
    "gray_zone_flag",
    "risk_hint",
    "soft_score",
    "policy_fired_ratio",
    "policy_fired_hint",
    "pair_ctxmatch_mean",
    "pair_seqgap_max",
    "pair_timegap_max",
    "pair_weightl1_mean",
    "score_conf_gap_mean",
    "context_drift_mean",
    "order_drift_mean",
)

SIGNAL_FALLBACK_POLICY: dict[str, str] = {
    "trust_mean": "required",
    "sim": "required",
    "context_ratio": "required",
    "pair_overlap_mean": "required",
    "pair_overlap_min": "fallback_to_pair_overlap_mean",
    "corr": "fallback_to_trust_mean",
    "gray_zone_flag": "fallback_to_0",
    "risk_hint": "fallback_to_0",
    "soft_score": "fallback_to_trust_mean",
    "policy_fired_ratio": "fallback_to_policy_fired_hint_or_0",
    "policy_fired_hint": "fallback_to_0",
    "pair_ctxmatch_mean": "fallback_to_1",
    "pair_seqgap_max": "fallback_to_0",
    "pair_timegap_max": "fallback_to_0",
    "pair_weightl1_mean": "fallback_to_0",
    "score_conf_gap_mean": "fallback_to_0",
    "context_drift_mean": "fallback_to_0",
    "order_drift_mean": "fallback_to_0",
}

FORENSIC_FEATURE_RAW_DIRECTIONALITY: dict[str, str] = {
    "r_trust_sim_gap": "high_is_suspicious",
    "r_overlap_spread": "high_is_suspicious",
    "r_context_dev": "high_is_suspicious",
    "r_policy_gray_contra": "high_is_suspicious",
    "r_trust_sim_prod": "low_is_suspicious",
    "r_benign_delta": "high_is_suspicious",
    "r_temporal_inconsistency": "high_is_suspicious",
    "r_witness_disagree": "high_is_suspicious",
}

FORENSIC_FEATURE_NORMALIZED_DIRECTIONALITY: dict[str, str] = {
    key: "high_is_suspicious" for key in FORENSIC_FEATURE_KEYS
}

FORENSIC_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "r_trust_sim_gap": "Absolute disagreement between trust coherence and explanation similarity.",
    "r_overlap_spread": "Spread between mean and minimum pairwise explanation overlap.",
    "r_context_dev": "Deviation from expected benign context consistency.",
    "r_policy_gray_contra": "Conflict between policy-safe signal and gray-risk evidence.",
    "r_trust_sim_prod": "Benign coherence product; lower values are more suspicious before normalization.",
    "r_benign_delta": "Mean distance from the fitted global benign reference.",
    "r_temporal_inconsistency": "Temporal or context continuity violation across node evidence.",
    "r_witness_disagree": "Witness-level disagreement across node-local anomaly evidence.",
}

DEFAULT_BENIGN_REFERENCE: dict[str, float] = {
    "trust_mean": 0.92,
    "sim": 0.92,
    "context_ratio": 0.98,
    "pair_overlap_mean": 0.92,
    "pair_overlap_min": 0.86,
    "policy_safe": 0.98,
}

DEFAULT_EXPECTED_CONTEXT_WEIGHTS: dict[str, float] = {
    "bias": 0.08,
    "trust_mean": 0.34,
    "sim": 0.34,
    "pair_overlap_mean": 0.16,
    "policy_safe": 0.08,
}


def forensic_feature_schema() -> dict[str, Any]:
    return {
        "feature_schema_version": FORENSIC_FEATURE_SCHEMA_VERSION,
        "reference_mode": FORENSIC_REFERENCE_MODE,
        "expected_context_mode": FORENSIC_EXPECTED_CONTEXT_MODE,
        "directionality_mode": FORENSIC_DIRECTIONALITY_MODE,
        "required_signal_keys": list(MINIMAL_SIGNAL_KEYS),
        "optional_signal_keys": list(OPTIONAL_SIGNAL_KEYS),
        "signal_fallback_policy": dict(SIGNAL_FALLBACK_POLICY),
        "raw_directionality": dict(FORENSIC_FEATURE_RAW_DIRECTIONALITY),
        "normalized_directionality": dict(FORENSIC_FEATURE_NORMALIZED_DIRECTIONALITY),
        "feature_descriptions": dict(FORENSIC_FEATURE_DESCRIPTIONS),
    }


def _float_or(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def policy_safe_from_signals(signals: Mapping[str, Any]) -> float:
    policy_ratio = _float_or(signals.get("policy_fired_ratio", signals.get("policy_fired_hint", 0.0)), 0.0)
    return clamp01(1.0 - policy_ratio)


def gray_risk_from_signals(signals: Mapping[str, Any]) -> float:
    gray_flag = clamp01(_float_or(signals.get("gray_zone_flag", 0.0), 0.0))
    risk_hint = clamp01(_float_or(signals.get("risk_hint", 0.0), 0.0))
    soft_score = clamp01(_float_or(signals.get("soft_score", signals.get("trust_mean", 0.0)), 0.0))
    context_ratio = clamp01(_float_or(signals.get("context_ratio", 0.0), 0.0))
    sim = clamp01(_float_or(signals.get("sim", 0.0), 0.0))
    return clamp01(
        max(
            gray_flag,
            risk_hint,
            1.0 - soft_score,
            1.0 - context_ratio,
            1.0 - sim,
        )
    )


def can_build_forensic_features(signals: Mapping[str, Any]) -> bool:
    return all(key in signals for key in MINIMAL_SIGNAL_KEYS)


def suspicious_feature_value(feature_key: str, features: Mapping[str, float]) -> float:
    value = clamp01(_float_or(features.get(feature_key, 0.0), 0.0))
    if feature_key == "r_trust_sim_prod":
        return clamp01(1.0 - value)
    return value


def build_forensic_features(
    signals: Mapping[str, Any],
    *,
    benign_reference: Mapping[str, float] | None = None,
    expected_context_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    trust_mean = clamp01(_float_or(signals.get("trust_mean", 0.0), 0.0))
    sim = clamp01(_float_or(signals.get("sim", 0.0), 0.0))
    corr = clamp01(_float_or(signals.get("corr", trust_mean), trust_mean))
    context_ratio = clamp01(_float_or(signals.get("context_ratio", 0.0), 0.0))
    pair_overlap_mean = clamp01(_float_or(signals.get("pair_overlap_mean", 0.0), 0.0))
    pair_overlap_min = clamp01(_float_or(signals.get("pair_overlap_min", pair_overlap_mean), pair_overlap_mean))
    pair_ctxmatch_mean = clamp01(_float_or(signals.get("pair_ctxmatch_mean", 1.0), 1.0))
    pair_seqgap_max = clamp01(_float_or(signals.get("pair_seqgap_max", 0.0), 0.0))
    pair_timegap_max = clamp01(_float_or(signals.get("pair_timegap_max", 0.0), 0.0))
    pair_weightl1_mean = clamp01(_float_or(signals.get("pair_weightl1_mean", 0.0), 0.0))
    score_conf_gap_mean = clamp01(_float_or(signals.get("score_conf_gap_mean", 0.0), 0.0))
    context_drift_mean = clamp01(_float_or(signals.get("context_drift_mean", 0.0), 0.0))
    order_drift_mean = clamp01(_float_or(signals.get("order_drift_mean", 0.0), 0.0))
    policy_safe = policy_safe_from_signals(signals)
    gray_risk = gray_risk_from_signals(signals)

    weights = dict(DEFAULT_EXPECTED_CONTEXT_WEIGHTS)
    if expected_context_weights:
        weights.update({str(k): _float_or(v, 0.0) for k, v in expected_context_weights.items()})
    expected_context = clamp01(
        _float_or(weights.get("bias", 0.0), 0.0)
        + (_float_or(weights.get("trust_mean", 0.0), 0.0) * trust_mean)
        + (_float_or(weights.get("sim", 0.0), 0.0) * sim)
        + (_float_or(weights.get("pair_overlap_mean", 0.0), 0.0) * pair_overlap_mean)
        + (_float_or(weights.get("policy_safe", 0.0), 0.0) * policy_safe)
    )

    ref = dict(DEFAULT_BENIGN_REFERENCE)
    if benign_reference:
        ref.update({str(k): clamp01(_float_or(v, ref.get(str(k), 0.0))) for k, v in benign_reference.items()})
    benign_delta_terms = (
        abs(trust_mean - ref["trust_mean"]),
        abs(sim - ref["sim"]),
        abs(context_ratio - ref["context_ratio"]),
        abs(pair_overlap_mean - ref["pair_overlap_mean"]),
        abs(pair_overlap_min - ref["pair_overlap_min"]),
        abs(policy_safe - ref["policy_safe"]),
    )
    benign_delta = clamp01(sum(benign_delta_terms) / max(len(benign_delta_terms), 1))

    temporal_inconsistency = clamp01(
        max(
            pair_timegap_max,
            pair_seqgap_max,
            context_drift_mean,
            clamp01(1.0 - pair_ctxmatch_mean),
        )
    )
    witness_disagree = clamp01(
        max(
            pair_weightl1_mean,
            score_conf_gap_mean,
            order_drift_mean,
            clamp01(1.0 - corr),
        )
    )

    return {
        "trust_mean": trust_mean,
        "sim": sim,
        "corr": corr,
        "context_ratio": context_ratio,
        "pair_overlap_mean": pair_overlap_mean,
        "pair_overlap_min": pair_overlap_min,
        "pair_ctxmatch_mean": pair_ctxmatch_mean,
        "pair_seqgap_max": pair_seqgap_max,
        "pair_timegap_max": pair_timegap_max,
        "pair_weightl1_mean": pair_weightl1_mean,
        "score_conf_gap_mean": score_conf_gap_mean,
        "context_drift_mean": context_drift_mean,
        "order_drift_mean": order_drift_mean,
        "policy_safe": policy_safe,
        "gray_risk": gray_risk,
        "expected_context": expected_context,
        "r_trust_sim_gap": clamp01(abs(trust_mean - sim)),
        "r_overlap_spread": clamp01(max(0.0, pair_overlap_mean - pair_overlap_min)),
        "r_context_dev": clamp01(abs(context_ratio - expected_context)),
        "r_policy_gray_contra": clamp01(policy_safe * gray_risk),
        "r_trust_sim_prod": clamp01(trust_mean * sim),
        "r_benign_delta": benign_delta,
        "r_temporal_inconsistency": temporal_inconsistency,
        "r_witness_disagree": witness_disagree,
    }


def encode_forensic_feature(feature_key: str, features: Mapping[str, float]) -> float:
    return clamp01(1.0 - suspicious_feature_value(feature_key, features))
