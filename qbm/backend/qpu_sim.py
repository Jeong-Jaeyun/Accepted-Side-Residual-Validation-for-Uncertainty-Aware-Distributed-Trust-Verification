from __future__ import annotations

import math
from typing import Sequence


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def qbm_accept_probability(trust_mean: float, corr: float, sim: float, shots: int = 2048) -> float:
    """
    Lightweight QBM-inspired acceptance score.
    """
    energy = (1.80 * trust_mean) + (1.20 * corr) + (0.90 * sim) - 1.75
    prob = _sigmoid(energy)
    shot_penalty = 0.30 / math.sqrt(max(int(shots), 1))
    return _clamp01(prob - shot_penalty)


def strongq_witness(trust_mean: float, corr: float, sim: float, shots: int = 4096) -> float:
    """
    Lightweight strongQ witness score.
    """
    benign_core = (0.42 * corr) + (0.28 * sim) + (0.30 * trust_mean)
    base = 1.0 - _clamp01(benign_core)
    shot_penalty = 0.24 / math.sqrt(max(int(shots), 1))
    return _clamp01(base - shot_penalty)


def strongq_witness_with_features(
    trust_mean: float,
    corr: float,
    sim: float,
    feature_vector: Sequence[float] | None,
    shots: int = 4096,
) -> tuple[float, float, float, float]:
    """
    StrongQ witness with micro-inconsistency risk features.
    `feature_vector` is expected to contain values where larger means "more suspicious".
    Returns: (witness, feature_mean, feature_std, feature_dim)
    """
    vals = [float(v) for v in (feature_vector or [])]
    benign_core = (0.42 * corr) + (0.28 * sim) + (0.30 * trust_mean)
    suspicion_core = 1.0 - _clamp01(benign_core)
    if vals and len(vals) >= 21:
        conf_mean = _clamp01(vals[0])
        conf_std = _clamp01(vals[1])
        conf_min = _clamp01(vals[2])
        anom_mean = _clamp01(vals[4])
        anom_std = _clamp01(vals[5])
        unc_mean = _clamp01(vals[6])
        unc_std = _clamp01(vals[7])
        overlap_mean = _clamp01(vals[8])
        overlap_min = _clamp01(vals[9])
        orderdist_mean = _clamp01(vals[10])
        orderdist_max = _clamp01(vals[11])
        ctxmatch_mean = _clamp01(vals[12])
        seqgap_max = _clamp01(vals[13])
        timegap_max = _clamp01(vals[14])
        weightl1_mean = _clamp01(vals[15])
        soft_score = _clamp01(vals[17])
        dist_to_tau = abs(float(vals[18]))
        gray_flag = _clamp01(vals[19])
        policy_ratio = _clamp01(vals[20])

        risk_node = _clamp01(
            (0.20 * (1.0 - conf_mean))
            + (0.20 * conf_std)
            + (0.10 * (1.0 - conf_min))
            + (0.25 * anom_mean)
            + (0.10 * anom_std)
            + (0.10 * unc_mean)
            + (0.05 * unc_std)
        )
        risk_pairwise = _clamp01(
            (0.18 * (1.0 - overlap_mean))
            + (0.18 * (1.0 - overlap_min))
            + (0.16 * orderdist_mean)
            + (0.16 * orderdist_max)
            + (0.16 * (1.0 - ctxmatch_mean))
            + (0.06 * seqgap_max)
            + (0.06 * timegap_max)
            + (0.04 * weightl1_mean)
        )
        # Risk rises near decision boundary, and when gray/policy hints are active.
        tau_band_risk = _clamp01(1.0 - min(1.0, dist_to_tau / 0.05))
        risk_global = _clamp01((0.35 * gray_flag) + (0.35 * policy_ratio) + (0.20 * tau_band_risk) + (0.10 * (1.0 - soft_score)))

        base = _clamp01(
            (0.30 * suspicion_core)
            + (0.20 * risk_node)
            + (0.35 * risk_pairwise)
            + (0.15 * risk_global)
        )

        feature_mean = float(sum(vals) / len(vals))
        feature_var = float(sum((v - feature_mean) ** 2 for v in vals) / len(vals))
        feature_std = math.sqrt(max(feature_var, 0.0))
    elif vals:
        feature_mean = float(sum(vals) / len(vals))
        feature_var = float(sum((v - feature_mean) ** 2 for v in vals) / len(vals))
        feature_std = math.sqrt(max(feature_var, 0.0))
        feature_spread = _clamp01(min(1.0, feature_std / 0.35))
        base = (0.55 * suspicion_core) + (0.35 * _clamp01(feature_mean)) + (0.10 * feature_spread)
    else:
        feature_mean = 0.0
        feature_std = 0.0
        base = suspicion_core
    shot_penalty = 0.24 / math.sqrt(max(int(shots), 1))
    witness = _clamp01(base - shot_penalty)
    return witness, feature_mean, feature_std, float(len(vals))


def shot_noise_std(prob_like: float, shots: int) -> float:
    """
    Binomial-style shot noise approximation for a [0,1] observable.
    """
    p = _clamp01(prob_like)
    n = max(int(shots), 1)
    return math.sqrt((p * (1.0 - p)) / n)


def strongq_witness_stats(
    trust_mean: float,
    corr: float,
    sim: float,
    shots: int = 4096,
    z_score: float = 1.96,
    feature_vector: Sequence[float] | None = None,
) -> dict[str, float]:
    """
    StrongQ witness with simple shot-noise uncertainty diagnostics.
    """
    witness, feature_mean, feature_std, feature_dim = strongq_witness_with_features(
        trust_mean=trust_mean,
        corr=corr,
        sim=sim,
        feature_vector=feature_vector,
        shots=shots,
    )
    std = shot_noise_std(witness, shots)
    z = float(max(z_score, 0.0))
    lo = _clamp01(witness - z * std)
    hi = _clamp01(witness + z * std)
    return {
        "witness": float(witness),
        "quantum_score": float(witness),
        "shot_std": float(std),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "shots": float(max(int(shots), 1)),
        "feature_mean": float(feature_mean),
        "feature_std": float(feature_std),
        "feature_dim": float(feature_dim),
    }
