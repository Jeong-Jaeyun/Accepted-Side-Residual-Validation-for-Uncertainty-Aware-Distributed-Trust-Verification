from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import json
import math

import numpy as np
import pandas as pd
import yaml

from preprocess.feature_extraction import compute_features
from preprocess.filter_by_port import filter_by_port
from preprocess.grid_mapping import add_grid_indices
from preprocess.load_and_map_schema import load_and_prepare
from preprocess.windowing import add_windows
from qbm.attacks import AttackGenerator, build_attack_plan
from qbm.backend.forensic_features import (
    FORENSIC_EXPECTED_CONTEXT_MODE,
    FORENSIC_FEATURE_KEYS,
    FORENSIC_FEATURE_SCHEMA_VERSION,
    FORENSIC_REFERENCE_MODE,
    suspicious_feature_value,
)
from qbm.model import (
    FEATURE_COLUMNS,
    DropType,
    Evidence,
    EvidenceBatch,
    SimulationSummary,
    VerificationHistory,
    VerifyResult,
    clamp01,
    explanation_hash,
    stable_hash,
)
from qbm.verifiers import QBMVerifier, S2StrictVerifier, S3MEVVerifier, StrongQVerifier


DEFAULTS: Dict[str, Any] = {
    "project": {
        "seed": 42,
        "processed_dir": "data/processed/busan",
        "raw_path": "data/raw/Busan_anonymized.csv",
        "results_dir": "results",
        "port": "busan",
        "artifacts_dir": "artifacts",
    },
    "time": {"timezone": "Asia/Seoul", "dt_minutes": 5},
    "port_filter": {"use_polygon": False, "bbox_override": None},
    "grid": {"nx": 10, "ny": 10},
    "experiments": {
        "scenario": "S3",
        "enable_injection": True,
        "attack_id": "",
        "capture_k": 2,
        "injection_window": {"start_window": 500, "end_window": 800},
        "A4": {
            # A4 = time-shifted replay. near-replay perturbations are controlled under A4P.
            "time_jitter_slots": 1,
            "recent_windows": 120,
        },
        "A4P": {
            # A4P = near-replay (partial perturbation)
            "level": 3,
            "p_topk_mutate": 0.20,
            "ctx_drift_strength": 0.08,
            "conf_jitter": 0.04,
            "time_jitter_slots": 1,
            "recent_windows": 120,
        },
    },
    "verification": {
        "implementation": None,
        "pipeline_mode": "auto",
        "quorum": 2.0 / 3.0,
        "confidence_floor": 0.70,
        "variance_tol": 1.0e-8,
        "corr_threshold": 0.78,
        "explanation_threshold": 0.50,
        "context_consistency_floor": 0.67,
        "soft_weights": {"corr": 0.35, "sim": 0.25, "context_ratio": 0.25, "trust_mean": 0.15},
        "s3_soft_tau": 0.72,
        "s3_soft_gray_margin": 0.05,
        "s3_grayzone_use_strongq": False,
        "s3_auto_tau_from_a0": True,
        "s3_tau_freeze_after_calibration": True,
        "s3_tau_calibration_windows": 300,
        "s3_tau_quantile": 0.95,
        "s3_tau_min_samples": 20,
        "s3_auto_gray_margin_from_a0": True,
        "s3_target_gray_rate_a0": 0.03,
        "s3_gray_margin_min": 0.001,
        "s3_gray_margin_max": 0.05,
        "s3_grayzone_requires_policy_hint": False,
        "s3_grayzone_no_policy_action": "mev",
        "s3_strongq_mode": "veto",
        "s3_risk_policy_ratio_floor": 0.10,
        "s3_risk_context_floor": 0.92,
        "s3_risk_explanation_floor": 0.90,
        "s3_risk_tau_band": 0.00012,
        "s3_auto_strongq_threshold": True,
        "s3_strongq_threshold_quantile": 0.50,
        "s3_strongq_threshold_min_samples": 12,
        "qbm_threshold": 0.55,
        "qbm_auto_threshold_from_a0": True,
        "qbm_threshold_quantile": 0.05,
        "qbm_threshold_min_samples": 20,
        "qbm_threshold_calibration_windows": 300,
        "qbm_threshold_stage": "accepted_only",
        "qbm_calibration_mode": "benign_only",
        "qbm_contrastive_attack_ids": [],
        "qbm_use_saved_calibration": True,
        "qbm_save_calibration": True,
        "qbm_calibration_artifact": None,
        "strongq_witness_threshold": 0.58,
        "strongq_ci_gate": False,
        "strongq_ci_z_score": 1.96,
        "explanation_feature_whitelist": list(FEATURE_COLUMNS),
        "s2_expected_topk_len": 3,
        "s2_enforce_explanation_schema": False,
        "trust_weights": {"score": 0.45, "confidence": 0.40, "explanation": 0.15},
    },
    "inference": {
        "anomaly_threshold": 0.65,
        "node_noise_std": 0.02,
        "explanation_jitter_prob": 0.08,
        "weights": [0.16, 0.19, 0.23, 0.19, 0.13, 0.10],
        "train_fraction": 0.70,
    },
    "simulation": {
        "base_capacity_tps": 180.0,
        "max_backlog": 350000.0,
        "partition_loss_extra": 0.06,
        "tcp_backlog_threshold": 250000.0,
        "tcp_drop_ratio_threshold": 0.80,
        "overhead_mult": {
            "none": 1.00,
            "s2_strict": 1.03,
            "s3_mev": 1.04,
            "s3_only": 1.04,
            "s3_strongq": 1.04,
            "qbm_verifier": 1.06,
            "s3_qbm": 1.06,
            "s3_qbm_strongq": 1.06,
            "strongq_verifier": 1.08,
        },
    },
    "evaluation": {
        # Window-level classification metrics are computed against the verifier-visible
        # malicious label by default so network loss does not count as a verifier miss.
        "label_source": "malicious_visible",   # malicious_visible | malicious_injected
        "prediction_source": "reject",         # reject | policy_fired | score
        "score_threshold": 0.50,
        "cost_weights": {
            "false_positive": 1.0,
            "false_negative": 5.0,
        },
    },
}

FALSE_TRUST_ATTACK_IDS = {"A3", "A4", "A4P", "A5"}


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    default_cfg_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    base_cfg = dict(DEFAULTS)
    if default_cfg_path.exists():
        with default_cfg_path.open("r", encoding="utf-8") as f:
            file_defaults = yaml.safe_load(f) or {}
        base_cfg = _deep_merge(base_cfg, file_defaults)
    if cfg_path.resolve() == default_cfg_path.resolve():
        return base_cfg
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _deep_merge(base_cfg, raw)


def apply_overrides(
    cfg: Dict[str, Any],
    *,
    scenario: Optional[str] = None,
    attack_id: Optional[str] = None,
    verifier_impl: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    out = dict(cfg)
    out["experiments"] = dict(cfg.get("experiments", {}))
    out["verification"] = dict(cfg.get("verification", {}))
    out["project"] = dict(cfg.get("project", {}))
    if scenario:
        out["experiments"]["scenario"] = scenario.upper()
    if attack_id:
        out["experiments"]["attack_id"] = attack_id.upper()
    if verifier_impl:
        out["verification"]["implementation"] = verifier_impl
    if seed is not None:
        out["project"]["seed"] = int(seed)
    return out


def _safe_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _load_features_from_processed(processed_dir: Path) -> Optional[pd.DataFrame]:
    parquet = processed_dir / "features.parquet"
    csv_path = processed_dir / "features.csv"
    df = _safe_read_parquet(parquet)
    if df is not None:
        return df
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def _build_features_from_raw(cfg: Dict[str, Any]) -> pd.DataFrame:
    project = cfg["project"]
    raw_path = Path(project["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing raw data: {raw_path}. "
            "Install pyarrow/fastparquet to read existing parquet outputs or provide raw CSV."
        )

    schema_mapping = cfg.get("schema_mapping", {})
    tz = cfg.get("time", {}).get("timezone", "UTC")
    df = load_and_prepare(str(raw_path), schema_mapping, tz_hint=tz)

    df = filter_by_port(
        df=df,
        port_name=str(project.get("port", "busan")),
        ports_path="ports/ports.yaml",
        use_polygon=bool(cfg.get("port_filter", {}).get("use_polygon", False)),
        bbox_override=cfg.get("port_filter", {}).get("bbox_override"),
    )

    grid_cfg = cfg.get("grid", {})
    bbox_override = cfg.get("port_filter", {}).get("bbox_override")
    if not bbox_override:
        with open("ports/ports.yaml", "r", encoding="utf-8") as f:
            ports = yaml.safe_load(f) or {}
        bbox_override = ports[project.get("port", "busan")]["bbox"]
    df = add_grid_indices(df, bbox_override, nx=int(grid_cfg.get("nx", 10)), ny=int(grid_cfg.get("ny", 10)))

    time_cfg = cfg.get("time", {})
    df, _t0 = add_windows(df, dt_minutes=int(time_cfg.get("dt_minutes", 5)), t0=time_cfg.get("t0"))
    feats = compute_features(df, cfg)
    return feats


def load_feature_windows(cfg: Dict[str, Any]) -> pd.DataFrame:
    processed_dir = Path(cfg["project"]["processed_dir"])
    df = _load_features_from_processed(processed_dir)
    if df is None:
        df = _build_features_from_raw(cfg)
        processed_dir.mkdir(parents=True, exist_ok=True)
        (processed_dir / "features.csv").parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_dir / "features.csv", index=False)

    required = ["window_id", *FEATURE_COLUMNS]
    missing = set(required) - set(df.columns)
    if missing:
        raise KeyError(f"Missing feature columns: {sorted(missing)}")

    out = df[required].copy()
    out = out.sort_values("window_id").drop_duplicates("window_id").reset_index(drop=True)
    return out


def _zscore(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = np.where(std <= 1e-12, 1.0, std)
    return (values - mean) / safe_std


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def build_feature_statistics(features_df: pd.DataFrame, train_fraction: float) -> Dict[str, np.ndarray]:
    n = len(features_df)
    train_n = max(8, int(round(n * float(train_fraction))))
    train = features_df.iloc[: min(train_n, n)]
    matrix = train[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    mins = features_df[list(FEATURE_COLUMNS)].min().to_numpy(dtype=float)
    maxs = features_df[list(FEATURE_COLUMNS)].max().to_numpy(dtype=float)
    return {"mean": mean, "std": std, "min": mins, "max": maxs}


def build_node_ids(cfg: Dict[str, Any]) -> List[str]:
    n = int(cfg.get("blockchain_net", {}).get("validators", 7))
    return [f"node_{i+1}" for i in range(max(1, n))]


def _rotate_tuple(values: Tuple[str, ...]) -> Tuple[str, ...]:
    if not values:
        return values
    return values[1:] + values[:1]


def generate_evidence_batch(
    window_row: pd.Series,
    *,
    step_index: int,
    cfg: Dict[str, Any],
    stats: Dict[str, np.ndarray],
    node_ids: List[str],
    prev_hash_by_node: Dict[str, str],
    rng: np.random.Generator,
) -> EvidenceBatch:
    inf_cfg = cfg.get("inference", {})
    node_noise = float(inf_cfg.get("node_noise_std", 0.02))
    exp_jitter = float(inf_cfg.get("explanation_jitter_prob", 0.08))
    weights = np.asarray(inf_cfg.get("weights", [0.16, 0.19, 0.23, 0.19, 0.13, 0.10]), dtype=float)
    if weights.size != len(FEATURE_COLUMNS):
        raise ValueError("inference.weights must have 6 entries (F1..F6).")
    weights = weights / max(weights.sum(), 1e-9)

    x = np.asarray([float(window_row[c]) for c in FEATURE_COLUMNS], dtype=float)
    z = _zscore(x, stats["mean"], stats["std"])
    contributions = z * weights
    base_score = clamp01(_sigmoid(float(np.dot(z, weights))))
    base_conf = clamp01(0.55 + 0.45 * (abs(base_score - 0.5) * 2.0))

    top_idx = np.argsort(np.abs(contributions))[::-1][:3]
    topk = tuple(FEATURE_COLUMNS[i] for i in top_idx)

    dt_minutes = int(cfg.get("time", {}).get("dt_minutes", 5))
    timestamp_ms = int(step_index * dt_minutes * 60_000)
    window_id = int(window_row["window_id"])
    context_hash = stable_hash(
        {
            "window_id": window_id,
            "features": {c: round(float(window_row[c]), 6) for c in FEATURE_COLUMNS},
        }
    )
    model_id = "qbm_node_local_v1"
    policy_id = "policy_default_v1"
    params_hash = stable_hash(
        {
            "weights": [round(float(w), 6) for w in weights.tolist()],
            "noise": round(node_noise, 6),
        }
    )

    evidences: List[Evidence] = []
    for node_id in node_ids:
        score = clamp01(base_score + float(rng.normal(0.0, node_noise)))
        conf = clamp01(base_conf + float(rng.normal(0.0, node_noise * 0.5)))
        uncertainty = clamp01(1.0 - conf)
        exp = topk
        if rng.random() < exp_jitter:
            exp = _rotate_tuple(topk)
        exp_hash = explanation_hash(exp)
        event_id = stable_hash({"window_id": window_id, "node_id": node_id, "step": step_index})
        prev_hash = prev_hash_by_node.get(node_id, "genesis")
        evidence = Evidence(
            event_id=event_id,
            window_id=window_id,
            node_id=node_id,
            decision="anomaly" if score >= float(inf_cfg.get("anomaly_threshold", 0.65)) else "normal",
            anomaly_score=score,
            confidence=conf,
            uncertainty=uncertainty,
            explanation_topk=exp,
            explanation_hash=exp_hash,
            model_id=model_id,
            policy_id=policy_id,
            params_hash=params_hash,
            context_hash=context_hash,
            timestamp_ms=timestamp_ms,
            prev_evidence_hash=prev_hash,
            malicious=False,
            attack_label="A0",
        )
        prev_hash_by_node[node_id] = evidence.payload_hash()
        evidences.append(evidence)

    return EvidenceBatch(window_id=window_id, evidences=evidences, network_state={})


def _base_s3_verifier_kwargs(
    *,
    cfg: Dict[str, Any],
    trust_weights: Mapping[str, float],
    soft_weights: Mapping[str, float],
    soft_tau: float,
    soft_gray_margin: float,
    auto_tau_from_a0: bool,
    tau_quantile: float,
    tau_min_samples: int,
    grayzone_requires_policy_hint: bool,
    grayzone_no_policy_action: str,
    strongq_mode: str,
    risk_policy_ratio_floor: float,
    risk_context_floor: float,
    risk_explanation_floor: float,
    risk_tau_band: float,
    strongq_verifier: Any = None,
) -> Dict[str, Any]:
    vcfg = cfg.get("verification", {})
    return {
        "quorum": float(vcfg.get("quorum", 2.0 / 3.0)),
        "corr_threshold": float(vcfg.get("corr_threshold", 0.78)),
        "explanation_threshold": float(vcfg.get("explanation_threshold", 0.50)),
        "context_consistency_floor": float(vcfg.get("context_consistency_floor", 0.67)),
        "soft_weights": dict(soft_weights),
        "tau": float(soft_tau),
        "gray_margin": float(soft_gray_margin),
        "auto_tau_from_a0": bool(auto_tau_from_a0),
        "tau_quantile": float(tau_quantile),
        "tau_min_samples": int(tau_min_samples),
        "grayzone_requires_policy_hint": bool(grayzone_requires_policy_hint),
        "grayzone_no_policy_action": str(grayzone_no_policy_action),
        "strongq_mode": str(strongq_mode),
        "risk_policy_ratio_floor": float(risk_policy_ratio_floor),
        "risk_context_floor": float(risk_context_floor),
        "risk_explanation_floor": float(risk_explanation_floor),
        "risk_tau_band": float(risk_tau_band),
        "strongq_verifier": strongq_verifier,
        "trust_weights": dict(trust_weights),
    }


def _build_strongq_resolver(
    *,
    cfg: Dict[str, Any],
    trust_weights: Mapping[str, float],
    soft_weights: Mapping[str, float],
    soft_tau: float,
    soft_gray_margin: float,
    auto_tau_from_a0: bool,
    tau_quantile: float,
    tau_min_samples: int,
    grayzone_requires_policy_hint: bool,
    grayzone_no_policy_action: str,
    strongq_mode: str,
    risk_policy_ratio_floor: float,
    risk_context_floor: float,
    risk_explanation_floor: float,
    risk_tau_band: float,
) -> StrongQVerifier:
    vcfg = cfg.get("verification", {})
    kwargs = _base_s3_verifier_kwargs(
        cfg=cfg,
        trust_weights=trust_weights,
        soft_weights=soft_weights,
        soft_tau=soft_tau,
        soft_gray_margin=soft_gray_margin,
        auto_tau_from_a0=auto_tau_from_a0,
        tau_quantile=tau_quantile,
        tau_min_samples=tau_min_samples,
        grayzone_requires_policy_hint=grayzone_requires_policy_hint,
        grayzone_no_policy_action=grayzone_no_policy_action,
        strongq_mode=strongq_mode,
        risk_policy_ratio_floor=risk_policy_ratio_floor,
        risk_context_floor=risk_context_floor,
        risk_explanation_floor=risk_explanation_floor,
        risk_tau_band=risk_tau_band,
        strongq_verifier=None,
    )
    return StrongQVerifier(
        **kwargs,
        witness_threshold=float(vcfg.get("strongq_witness_threshold", 0.58)),
        shots=int(cfg.get("qbm", {}).get("shots", 4096)),
        ci_gate=bool(vcfg.get("strongq_ci_gate", False)),
        ci_z_score=float(vcfg.get("strongq_ci_z_score", 1.96)),
    )


def build_verifier(cfg: Dict[str, Any], scenario: str):
    vcfg = cfg.get("verification", {})
    trust_weights = dict(vcfg.get("trust_weights", {"score": 0.45, "confidence": 0.40, "explanation": 0.15}))
    soft_weights = dict(vcfg.get("soft_weights", {"corr": 0.35, "sim": 0.25, "context_ratio": 0.25, "trust_mean": 0.15}))
    soft_tau = float(vcfg.get("s3_soft_tau", 0.72))
    soft_gray_margin = float(vcfg.get("s3_soft_gray_margin", 0.05))
    auto_tau_from_a0 = bool(vcfg.get("s3_auto_tau_from_a0", True))
    tau_quantile = float(vcfg.get("s3_tau_quantile", 0.95))
    tau_min_samples = int(vcfg.get("s3_tau_min_samples", 20))
    grayzone_requires_policy_hint = bool(vcfg.get("s3_grayzone_requires_policy_hint", False))
    grayzone_no_policy_action = str(vcfg.get("s3_grayzone_no_policy_action", "mev"))
    strongq_mode = str(vcfg.get("s3_strongq_mode", "veto"))
    risk_policy_ratio_floor = float(vcfg.get("s3_risk_policy_ratio_floor", 0.10))
    risk_context_floor = float(vcfg.get("s3_risk_context_floor", 0.90))
    risk_explanation_floor = float(vcfg.get("s3_risk_explanation_floor", 0.75))
    risk_tau_band = float(vcfg.get("s3_risk_tau_band", 0.00))
    pipeline_mode = str(vcfg.get("pipeline_mode", "auto")).strip().lower() or "auto"
    qbm_cfg = dict(cfg.get("qbm", {}))

    impl = vcfg.get("implementation")
    if impl is None:
        impl = {"S2": "s2_strict", "S3": "s3_mev"}.get(scenario, "none")
    impl = str(impl).lower()

    if scenario in {"S0", "S1"} or impl == "none":
        # S1 is intentionally "logging only": no replay/time-sequence checks, no verifier gate.
        return None, "none"

    if impl == "s2_strict":
        return (
            S2StrictVerifier(
                quorum=float(vcfg.get("quorum", 1.0)),
                confidence_floor=float(vcfg.get("confidence_floor", 0.70)),
                variance_tol=float(vcfg.get("variance_tol", 1.0e-8)),
                context_consistency_floor=float(vcfg.get("context_consistency_floor", 1.0)),
                strict_window_sequence=bool(vcfg.get("strict_window_sequence", True)),
                trust_weights=trust_weights,
                explanation_feature_whitelist=vcfg.get("explanation_feature_whitelist", FEATURE_COLUMNS),
                expected_topk_len=int(vcfg.get("s2_expected_topk_len", 3)),
                enforce_explanation_schema=bool(vcfg.get("s2_enforce_explanation_schema", False)),
            ),
            "s2_strict",
        )

    if scenario == "S3" and pipeline_mode in {"s3_only", "s3_qbm", "s3_strongq", "s3_qbm_strongq"}:
        strongq_resolver = None
        if pipeline_mode in {"s3_strongq", "s3_qbm_strongq"}:
            strongq_resolver = _build_strongq_resolver(
                cfg=cfg,
                trust_weights=trust_weights,
                soft_weights=soft_weights,
                soft_tau=soft_tau,
                soft_gray_margin=soft_gray_margin,
                auto_tau_from_a0=auto_tau_from_a0,
                tau_quantile=tau_quantile,
                tau_min_samples=tau_min_samples,
                grayzone_requires_policy_hint=grayzone_requires_policy_hint,
                grayzone_no_policy_action=grayzone_no_policy_action,
                strongq_mode=strongq_mode,
                risk_policy_ratio_floor=risk_policy_ratio_floor,
                risk_context_floor=risk_context_floor,
                risk_explanation_floor=risk_explanation_floor,
                risk_tau_band=risk_tau_band,
            )
        base_kwargs = _base_s3_verifier_kwargs(
            cfg=cfg,
            trust_weights=trust_weights,
            soft_weights=soft_weights,
            soft_tau=soft_tau,
            soft_gray_margin=soft_gray_margin,
            auto_tau_from_a0=auto_tau_from_a0,
            tau_quantile=tau_quantile,
            tau_min_samples=tau_min_samples,
            grayzone_requires_policy_hint=grayzone_requires_policy_hint,
            grayzone_no_policy_action=grayzone_no_policy_action,
            strongq_mode=strongq_mode,
            risk_policy_ratio_floor=risk_policy_ratio_floor,
            risk_context_floor=risk_context_floor,
            risk_explanation_floor=risk_explanation_floor,
            risk_tau_band=risk_tau_band,
            strongq_verifier=strongq_resolver,
        )
        if pipeline_mode == "s3_only":
            verifier = S3MEVVerifier(**base_kwargs)
        elif pipeline_mode == "s3_qbm":
            verifier = QBMVerifier(
                **base_kwargs,
                qbm_threshold=float(vcfg.get("qbm_threshold", 0.55)),
                shots=int(qbm_cfg.get("shots", 2048)),
                qiskit_config=qbm_cfg,
            )
        elif pipeline_mode == "s3_strongq":
            verifier = S3MEVVerifier(**base_kwargs)
        else:
            verifier = QBMVerifier(
                **base_kwargs,
                qbm_threshold=float(vcfg.get("qbm_threshold", 0.55)),
                shots=int(qbm_cfg.get("shots", 2048)),
                qiskit_config=qbm_cfg,
            )
        setattr(verifier, "pipeline_mode", pipeline_mode)
        return verifier, pipeline_mode

    if impl == "qbm":
        return (
            QBMVerifier(
                **_base_s3_verifier_kwargs(
                    cfg=cfg,
                    trust_weights=trust_weights,
                    soft_weights=soft_weights,
                    soft_tau=soft_tau,
                    soft_gray_margin=soft_gray_margin,
                    auto_tau_from_a0=auto_tau_from_a0,
                    tau_quantile=tau_quantile,
                    tau_min_samples=tau_min_samples,
                    grayzone_requires_policy_hint=grayzone_requires_policy_hint,
                    grayzone_no_policy_action=grayzone_no_policy_action,
                    strongq_mode=strongq_mode,
                    risk_policy_ratio_floor=risk_policy_ratio_floor,
                    risk_context_floor=risk_context_floor,
                    risk_explanation_floor=risk_explanation_floor,
                    risk_tau_band=risk_tau_band,
                    strongq_verifier=None,
                ),
                qbm_threshold=float(vcfg.get("qbm_threshold", 0.55)),
                shots=int(qbm_cfg.get("shots", 2048)),
                qiskit_config=qbm_cfg,
            ),
            "qbm_verifier",
        )
    if impl == "strongq":
        return (
            _build_strongq_resolver(
                cfg=cfg,
                trust_weights=trust_weights,
                soft_weights=soft_weights,
                soft_tau=soft_tau,
                soft_gray_margin=soft_gray_margin,
                auto_tau_from_a0=auto_tau_from_a0,
                tau_quantile=tau_quantile,
                tau_min_samples=tau_min_samples,
                grayzone_requires_policy_hint=grayzone_requires_policy_hint,
                grayzone_no_policy_action=grayzone_no_policy_action,
                strongq_mode=strongq_mode,
                risk_policy_ratio_floor=risk_policy_ratio_floor,
                risk_context_floor=risk_context_floor,
                risk_explanation_floor=risk_explanation_floor,
                risk_tau_band=risk_tau_band,
            ),
            "strongq_verifier",
        )

    strongq_fallback = None
    if bool(vcfg.get("s3_grayzone_use_strongq", True)):
        strongq_fallback = _build_strongq_resolver(
            cfg=cfg,
            trust_weights=trust_weights,
            soft_weights=soft_weights,
            soft_tau=soft_tau,
            soft_gray_margin=soft_gray_margin,
            auto_tau_from_a0=auto_tau_from_a0,
            tau_quantile=tau_quantile,
            tau_min_samples=tau_min_samples,
            grayzone_requires_policy_hint=grayzone_requires_policy_hint,
            grayzone_no_policy_action=grayzone_no_policy_action,
            strongq_mode=strongq_mode,
            risk_policy_ratio_floor=risk_policy_ratio_floor,
            risk_context_floor=risk_context_floor,
            risk_explanation_floor=risk_explanation_floor,
            risk_tau_band=risk_tau_band,
        )
    return (
        S3MEVVerifier(
            **_base_s3_verifier_kwargs(
                cfg=cfg,
                trust_weights=trust_weights,
                soft_weights=soft_weights,
                soft_tau=soft_tau,
                soft_gray_margin=soft_gray_margin,
                auto_tau_from_a0=auto_tau_from_a0,
                tau_quantile=tau_quantile,
                tau_min_samples=tau_min_samples,
                grayzone_requires_policy_hint=grayzone_requires_policy_hint,
                grayzone_no_policy_action=grayzone_no_policy_action,
                strongq_mode=strongq_mode,
                risk_policy_ratio_floor=risk_policy_ratio_floor,
                risk_context_floor=risk_context_floor,
                risk_explanation_floor=risk_explanation_floor,
                risk_tau_band=risk_tau_band,
                strongq_verifier=strongq_fallback,
            ),
        ),
        "s3_mev",
    )


def _offered_load(window_row: pd.Series, stats: Dict[str, np.ndarray], rng: np.random.Generator) -> float:
    idx = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
    mins = stats["min"]
    maxs = stats["max"]

    def norm(col: str) -> float:
        i = idx[col]
        lo, hi = mins[i], maxs[i]
        x = float(window_row[col])
        if hi <= lo:
            return 0.0
        return max(0.0, min(1.0, (x - lo) / (hi - lo)))

    v1 = norm("F1_unique_mmsi_count")
    v3 = norm("F3_message_burstiness")
    v4 = norm("F4_position_jump_rate")
    offered = 140.0 + (120.0 * v1) + (220.0 * v3) + (80.0 * v4) + float(rng.normal(0.0, 8.0))
    return max(20.0, offered)


def apply_network_visibility(
    batch: EvidenceBatch,
    *,
    base_loss_rate: float,
    partition_loss_extra: float,
    rng: np.random.Generator,
) -> tuple[EvidenceBatch, Dict[str, float]]:
    total = len(batch.evidences)
    if total <= 0:
        return batch, {"loss_rate": 0.0, "visibility_ratio": 0.0}

    loss_rate = base_loss_rate + float(batch.network_state.get("extra_loss_rate", 0.0))
    if bool(batch.network_state.get("partition", False)):
        loss_rate += partition_loss_extra
    loss_rate = max(0.0, min(0.95, loss_rate))

    delivered = [e for e in batch.evidences if rng.random() > loss_rate]
    if not delivered:
        visible: List[Evidence] = []
    elif bool(batch.network_state.get("partition", False)):
        partition_nodes = set(batch.network_state.get("partition_nodes", []))
        if not partition_nodes:
            ids = sorted({e.node_id for e in delivered})
            half = max(1, len(ids) // 2)
            partition_nodes = set(ids[half:])
        view = "A" if rng.random() < 0.5 else "B"
        if view == "A":
            visible = [e for e in delivered if e.node_id not in partition_nodes]
        else:
            visible = [e for e in delivered if e.node_id in partition_nodes]
        # Avoid pathological empty-view with non-empty delivered set.
        if not visible:
            visible = [delivered[int(rng.integers(0, len(delivered)))]]
    else:
        visible = delivered

    ratio = len(visible) / max(total, 1)
    visible_batch = EvidenceBatch(
        window_id=batch.window_id,
        evidences=visible,
        network_state=dict(batch.network_state),
    )
    return visible_batch, {"loss_rate": float(loss_rate), "visibility_ratio": float(ratio)}


def _policy_hint_from_network(
    batch: EvidenceBatch,
    network_stats: Mapping[str, float],
    base_loss: float,
) -> bool:
    return (
        bool(batch.network_state.get("partition", False))
        or (float(network_stats.get("loss_rate", 0.0)) >= (base_loss + 0.03))
        or (float(network_stats.get("visibility_ratio", 1.0)) <= 0.85)
    )


def _precalibrate_tau_from_a0(
    *,
    verifier: Any,
    cfg: Dict[str, Any],
    features: pd.DataFrame,
    stats: Dict[str, np.ndarray],
    node_ids: List[str],
    base_loss: float,
    partition_loss_extra: float,
    seed: int,
) -> Dict[str, Any]:
    if verifier is None or not isinstance(verifier, S3MEVVerifier):
        return {}
    vcfg = cfg.get("verification", {})
    if not bool(vcfg.get("s3_auto_tau_from_a0", True)):
        return {}

    cal_windows = int(vcfg.get("s3_tau_calibration_windows", 300))
    cal_windows = max(cal_windows, int(getattr(verifier, "tau_min_samples", 1)))
    cal_windows = max(1, min(len(features), cal_windows))

    rng_cal = np.random.default_rng(seed + 1001)
    prev_hash_by_node: Dict[str, str] = {}
    soft_scores: List[float] = []
    for step in range(cal_windows):
        row = features.iloc[step]
        batch = generate_evidence_batch(
            row,
            step_index=step,
            cfg=cfg,
            stats=stats,
            node_ids=node_ids,
            prev_hash_by_node=prev_hash_by_node,
            rng=rng_cal,
        )
        verify_batch, network_stats = apply_network_visibility(
            batch,
            base_loss_rate=base_loss,
            partition_loss_extra=partition_loss_extra,
            rng=rng_cal,
        )
        verify_batch.network_state["policy_fired_hint"] = _policy_hint_from_network(verify_batch, network_stats, base_loss)
        if not verify_batch.evidences:
            continue
        signals = verifier.compute_signals(verify_batch)
        soft_score, _corr_scaled, _sim_scaled = verifier.compute_soft_score(signals)
        soft_scores.append(float(soft_score))

    min_samples = int(getattr(verifier, "tau_min_samples", 1))
    calibrated = False
    # soft_score is trust-like (higher => easier accept). For tau calibration we
    # convert the configured upper-tail quantile into the equivalent lower-tail
    # cutoff used by accept/reject gating.
    effective_quantile = clamp01(1.0 - float(getattr(verifier, "tau_quantile", 0.95)))
    if len(soft_scores) >= min_samples:
        verifier.tau = clamp01(
            float(np.quantile(np.asarray(soft_scores, dtype=float), effective_quantile))
        )
        calibrated = True
        if bool(vcfg.get("s3_auto_gray_margin_from_a0", True)):
            target_gray = clamp01(float(vcfg.get("s3_target_gray_rate_a0", 0.03)))
            dist = np.abs(np.asarray(soft_scores, dtype=float) - float(verifier.tau))
            margin = float(np.quantile(dist, target_gray))
            margin_min = float(max(vcfg.get("s3_gray_margin_min", 0.005), 0.0))
            margin_max = float(max(vcfg.get("s3_gray_margin_max", 0.05), margin_min))
            verifier.gray_margin = float(np.clip(margin, margin_min, margin_max))

    if bool(vcfg.get("s3_tau_freeze_after_calibration", True)):
        verifier.auto_tau_from_a0 = False

    return {
        "tau_precalibrated": 1.0 if calibrated else 0.0,
        "tau_precalibration_samples": float(len(soft_scores)),
        "tau_precalibrated_value": float(verifier.tau),
        "tau_precalibration_quantile_config": float(getattr(verifier, "tau_quantile", 0.95)),
        "tau_precalibration_quantile_effective": float(effective_quantile),
        "gray_margin_precalibrated_value": float(getattr(verifier, "gray_margin", float("nan"))),
    }


def _resolve_strongq_resolver(verifier: Any) -> Optional[StrongQVerifier]:
    if isinstance(verifier, StrongQVerifier):
        return verifier
    candidate = getattr(verifier, "strongq_verifier", None)
    if isinstance(candidate, StrongQVerifier):
        return candidate
    return None


def _precalibrate_strongq_threshold(
    *,
    verifier: Any,
    cfg: Dict[str, Any],
    features: pd.DataFrame,
    stats: Dict[str, np.ndarray],
    node_ids: List[str],
    base_loss: float,
    partition_loss_extra: float,
    plan: Any,
    seed: int,
) -> Dict[str, Any]:
    if verifier is None or not isinstance(verifier, S3MEVVerifier):
        return {}
    if str(getattr(plan, "attack_id", "A0")).upper() == "A0":
        return {}
    vcfg = cfg.get("verification", {})
    if not bool(vcfg.get("s3_auto_strongq_threshold", True)):
        return {}
    strongq_resolver = _resolve_strongq_resolver(verifier)
    if strongq_resolver is None:
        return {}

    quantile = clamp01(float(vcfg.get("s3_strongq_threshold_quantile", 0.50)))
    min_samples = max(1, int(vcfg.get("s3_strongq_threshold_min_samples", 12)))

    rng_cal = np.random.default_rng(seed + 2001)
    history = VerificationHistory(replay_buffer_max=2048)
    attacker = AttackGenerator(plan)
    prev_hash_by_node: Dict[str, str] = {}
    witness_scores: List[float] = []
    for step, (_, row) in enumerate(features.iterrows()):
        batch = generate_evidence_batch(
            row,
            step_index=step,
            cfg=cfg,
            stats=stats,
            node_ids=node_ids,
            prev_hash_by_node=prev_hash_by_node,
            rng=rng_cal,
        )
        batch, net_delta, _mal_count = attacker.apply(batch, history, rng_cal)
        if net_delta:
            batch.network_state.update(net_delta)

        verify_batch, network_stats = apply_network_visibility(
            batch,
            base_loss_rate=base_loss,
            partition_loss_extra=partition_loss_extra,
            rng=rng_cal,
        )
        verify_batch.network_state["policy_fired_hint"] = _policy_hint_from_network(verify_batch, network_stats, base_loss)

        if not verify_batch.evidences:
            continue

        signals = verifier.compute_signals(verify_batch)
        soft_score, _corr_scaled, _sim_scaled = verifier.compute_soft_score(signals)
        upper = clamp01(float(verifier.tau) + float(verifier.gray_margin))
        lower = clamp01(float(verifier.tau) - float(verifier.gray_margin))
        if soft_score >= upper or soft_score <= lower:
            history.record_commit(verify_batch)
            continue

        mev_decision = soft_score >= float(verifier.tau)
        if getattr(verifier, "strongq_mode", "veto") == "veto" and not mev_decision:
            history.record_commit(verify_batch)
            continue
        signals = dict(signals)
        signals["soft_score"] = float(soft_score)
        signals["tau"] = float(verifier.tau)
        signals["gray_zone_flag"] = 1.0
        signals["gray_margin"] = float(verifier.gray_margin)
        if hasattr(verifier, "build_strongq_feature_vector"):
            signals["strongq_feature_vector"] = verifier.build_strongq_feature_vector(
                signals,
                soft_score=float(soft_score),
                gray_flag=1.0,
            )
        stats_map = strongq_resolver.score_from_signals(signals)
        witness = float(stats_map.get("witness", stats_map.get("quantum_score", float("nan"))))
        if np.isfinite(witness):
            witness_scores.append(witness)
        history.record_commit(verify_batch)

    calibrated = False
    if len(witness_scores) >= min_samples:
        strongq_resolver.witness_threshold = clamp01(
            float(np.quantile(np.asarray(witness_scores, dtype=float), quantile))
        )
        calibrated = True

    return {
        "strongq_threshold_precalibrated": 1.0 if calibrated else 0.0,
        "strongq_threshold_samples": float(len(witness_scores)),
        "strongq_threshold_quantile": float(quantile),
        "strongq_threshold_value": float(strongq_resolver.witness_threshold),
    }


def _qbm_calibration_artifact_path(cfg: Dict[str, Any], scenario: str, verifier: QBMVerifier) -> Path:
    vcfg = cfg.get("verification", {})
    explicit = vcfg.get("qbm_calibration_artifact")
    if explicit:
        return Path(str(explicit))
    artifacts_dir = Path(cfg.get("project", {}).get("artifacts_dir", "artifacts")) / "qbm_calibration"
    port = str(cfg.get("project", {}).get("port", "busan")).replace(" ", "_").lower()
    mode = str(vcfg.get("qbm_calibration_mode", "benign_only")).strip().lower() or "benign_only"
    backend_mode = str(getattr(verifier._scorer, "backend_mode", "exact_state")).replace("/", "_").replace(" ", "_")
    return artifacts_dir / f"{port}_{scenario.lower()}_qiskit_{backend_mode}_{mode}.json"


def _collect_qbm_shadow_records(
    *,
    verifier: QBMVerifier,
    cfg: Dict[str, Any],
    features: pd.DataFrame,
    stats: Dict[str, np.ndarray],
    node_ids: List[str],
    base_loss: float,
    partition_loss_extra: float,
    seed: int,
    attack_id: str,
    max_windows: int,
) -> pd.DataFrame:
    cfg_local = copy.deepcopy(cfg)
    cfg_local.setdefault("experiments", {})
    cfg_local["experiments"]["attack_id"] = str(attack_id).upper()
    cfg_local["experiments"]["enable_injection"] = str(attack_id).upper() != "A0"
    plan = build_attack_plan(cfg_local)
    attacker = AttackGenerator(plan)
    history = VerificationHistory(replay_buffer_max=2048)
    prev_hash_by_node: Dict[str, str] = {}
    attack_seed = 1000 if str(attack_id).upper() == "A0" else (2000 + sum(ord(ch) for ch in str(attack_id).upper()) % 1000)
    rng_cal = np.random.default_rng(seed + attack_seed)

    rows: List[Dict[str, Any]] = []
    for step in range(min(len(features), max_windows)):
        row = features.iloc[step]
        batch = generate_evidence_batch(
            row,
            step_index=step,
            cfg=cfg_local,
            stats=stats,
            node_ids=node_ids,
            prev_hash_by_node=prev_hash_by_node,
            rng=rng_cal,
        )
        batch, net_delta, mal_count = attacker.apply(batch, history, rng_cal)
        if net_delta:
            batch.network_state.update(net_delta)

        verify_batch, network_stats = apply_network_visibility(
            batch,
            base_loss_rate=base_loss,
            partition_loss_extra=partition_loss_extra,
            rng=rng_cal,
        )
        verify_batch.network_state["policy_fired_hint"] = _policy_hint_from_network(verify_batch, network_stats, base_loss)

        shadow_scores = verifier.shadow_score_batch(verify_batch)
        if not shadow_scores:
            continue

        base_result = S3MEVVerifier.verify(verifier, verify_batch, history)
        if base_result.accept:
            history.record_commit(verify_batch)

        row_map = dict(shadow_scores)
        row_map["window_id"] = int(verify_batch.window_id)
        row_map["attack_id"] = str(attack_id).upper()
        row_map["step"] = int(step)
        row_map["base_accept"] = 1.0 if base_result.accept else 0.0
        row_map["malicious_visible"] = int(verify_batch.malicious_count)
        row_map["malicious_injected"] = int(mal_count)
        rows.append(row_map)

    return pd.DataFrame(rows)


def _fit_expected_context_weights_from_records(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "bias": 0.08,
            "trust_mean": 0.34,
            "sim": 0.34,
            "pair_overlap_mean": 0.16,
            "policy_safe": 0.08,
        }
    benign = df[pd.to_numeric(df.get("malicious_visible", 0.0), errors="coerce").fillna(0.0) <= 0]
    if benign.empty:
        benign = df
    cols = ["qbm_trust_mean", "qbm_sim", "qbm_pair_overlap_mean", "qbm_policy_safe"]
    X = benign[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(benign.get("qbm_context_ratio", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if X.size == 0 or y.size == 0:
        return {
            "bias": 0.08,
            "trust_mean": 0.34,
            "sim": 0.34,
            "pair_overlap_mean": 0.16,
            "policy_safe": 0.08,
        }
    X_aug = np.column_stack([np.ones(len(X)), X])
    coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    coeffs = np.asarray(coeffs, dtype=float)
    return {
        "bias": float(coeffs[0]),
        "trust_mean": float(coeffs[1]),
        "sim": float(coeffs[2]),
        "pair_overlap_mean": float(coeffs[3]),
        "policy_safe": float(coeffs[4]),
    }


def _fit_benign_reference_from_records(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    benign = df[pd.to_numeric(df.get("malicious_visible", 0.0), errors="coerce").fillna(0.0) <= 0]
    if benign.empty:
        benign = df
    out: Dict[str, float] = {}
    for src, dst in (
        ("qbm_trust_mean", "trust_mean"),
        ("qbm_sim", "sim"),
        ("qbm_context_ratio", "context_ratio"),
        ("qbm_pair_overlap_mean", "pair_overlap_mean"),
        ("qbm_pair_overlap_min", "pair_overlap_min"),
        ("qbm_policy_safe", "policy_safe"),
    ):
        series = pd.to_numeric(benign.get(src, pd.Series(dtype=float)), errors="coerce").dropna()
        if not series.empty:
            out[dst] = float(series.median())
    return out


def _fit_visible_weights_from_records(df: pd.DataFrame, feature_keys: Sequence[str], current_weights: Sequence[float], mode: str) -> list[float]:
    if df.empty:
        return [float(w) for w in current_weights]
    benign = df[pd.to_numeric(df.get("malicious_visible", 0.0), errors="coerce").fillna(0.0) <= 0]
    if benign.empty:
        benign = df
    malicious = df[pd.to_numeric(df.get("malicious_visible", 0.0), errors="coerce").fillna(0.0) > 0]

    cols = [f"qbm_{key}" for key in feature_keys]
    benign_matrix = benign[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if benign_matrix.size == 0:
        return [float(w) for w in current_weights]

    suspicious_benign = benign_matrix.copy()
    for idx, key in enumerate(feature_keys):
        if key == "r_trust_sim_prod":
            suspicious_benign[:, idx] = 1.0 - suspicious_benign[:, idx]
    q75 = np.quantile(suspicious_benign, 0.75, axis=0)
    q25 = np.quantile(suspicious_benign, 0.25, axis=0)
    iqr = np.maximum(q75 - q25, 0.03)
    base = 1.0 / iqr

    if mode == "contrastive" and not malicious.empty:
        malicious_matrix = malicious[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        suspicious_mal = malicious_matrix.copy()
        for idx, key in enumerate(feature_keys):
            if key == "r_trust_sim_prod":
                suspicious_mal[:, idx] = 1.0 - suspicious_mal[:, idx]
        effect = np.abs(suspicious_mal.mean(axis=0) - suspicious_benign.mean(axis=0)) / iqr
        base = base * (1.0 + effect)

    target_mean = float(np.mean(np.asarray(current_weights, dtype=float))) if current_weights else 0.82
    scaled = base / max(float(np.mean(base)), 1.0e-9)
    scaled = np.clip(scaled * target_mean, 0.30, 1.60)
    return [float(v) for v in scaled.tolist()]


def _fit_pair_terms_from_records(df: pd.DataFrame, feature_keys: Sequence[str]) -> tuple[list[list[float]], list[list[float]]]:
    if df.empty:
        return [], []
    cols = [f"qbm_{key}" for key in feature_keys]
    matrix = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if matrix.shape[0] < 4 or matrix.shape[1] < 2:
        return [], []
    suspicious = matrix.copy()
    for idx, key in enumerate(feature_keys):
        if key == "r_trust_sim_prod":
            suspicious[:, idx] = 1.0 - suspicious[:, idx]
    std = np.std(suspicious, axis=0)
    valid_idx = [idx for idx, value in enumerate(std.tolist()) if float(value) > 1.0e-8]
    if len(valid_idx) < 2:
        return [], []
    corr = np.corrcoef(suspicious[:, valid_idx], rowvar=False)
    pairs: list[tuple[float, int, int]] = []
    for local_i in range(corr.shape[0]):
        for local_j in range(local_i + 1, corr.shape[1]):
            i = valid_idx[local_i]
            j = valid_idx[local_j]
            value = abs(float(corr[local_i, local_j]))
            if np.isfinite(value):
                pairs.append((value, i, j))
    pairs.sort(reverse=True)
    couplings: list[list[float]] = []
    weights: list[list[float]] = []
    for value, i, j in pairs[: min(6, len(pairs))]:
        if value < 0.05:
            continue
        couplings.append([int(i), int(j), float(np.clip(0.08 + (0.20 * value), 0.08, 0.30))])
        weights.append([int(i), int(j), float(np.clip(0.05 + (0.14 * value), 0.05, 0.22))])
    return couplings, weights


def _apply_qbm_calibration_to_records(verifier: QBMVerifier, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    scorer = getattr(verifier, "_scorer", None)
    if scorer is None or not hasattr(scorer, "score_many"):
        return df.copy()

    signals_batch: List[Dict[str, Any]] = [dict(row) for _, row in df.iterrows()]

    rescored = scorer.score_many(signals_batch)
    out = df.copy()
    for idx, score_map in enumerate(rescored):
        for key, value in score_map.items():
            out.at[out.index[idx], key] = value
        if "q_score" in score_map:
            out.at[out.index[idx], "q_score_shadow"] = score_map["q_score"]
    return out


def _fit_qbm_calibration_artifact(
    *,
    verifier: QBMVerifier,
    records: pd.DataFrame,
    mode: str,
    quantile: float,
    stage: str,
) -> dict[str, Any]:
    scorer = getattr(verifier, "_scorer", None)
    if scorer is None:
        return {"qbm_threshold": float(verifier.qbm_threshold)}

    def _select_score_source(df: pd.DataFrame) -> pd.DataFrame:
        if stage == "accepted_only":
            return df.loc[pd.to_numeric(df.get("base_accept", 0.0), errors="coerce").fillna(0.0) > 0]
        return df

    benign_reference = _fit_benign_reference_from_records(records)
    expected_context_weights = _fit_expected_context_weights_from_records(records)
    visible_weights = _fit_visible_weights_from_records(
        records,
        feature_keys=getattr(scorer, "feature_keys", FORENSIC_FEATURE_KEYS),
        current_weights=getattr(scorer, "visible_weights", [0.8] * len(getattr(scorer, "feature_keys", FORENSIC_FEATURE_KEYS))),
        mode=mode,
    )
    pair_couplings, pair_weights = _fit_pair_terms_from_records(records, getattr(scorer, "feature_keys", FORENSIC_FEATURE_KEYS))

    artifact: dict[str, Any] = {
        "version": 1,
        "mode": str(mode),
        "benign_reference": benign_reference,
        "expected_context_weights": expected_context_weights,
        "visible_weights": visible_weights,
    }
    if pair_couplings:
        artifact["visible_pair_couplings"] = pair_couplings
    if pair_weights:
        artifact["visible_pair_weights"] = pair_weights

    verifier.apply_calibration(artifact)
    rescored = _apply_qbm_calibration_to_records(verifier, records)

    score_source = _select_score_source(rescored)

    benign_scores = pd.to_numeric(
        score_source.loc[pd.to_numeric(score_source.get("malicious_visible", 0.0), errors="coerce").fillna(0.0) <= 0, "q_score_shadow"],
        errors="coerce",
    ).dropna()
    malicious_scores = pd.to_numeric(
        score_source.loc[pd.to_numeric(score_source.get("malicious_visible", 0.0), errors="coerce").fillna(0.0) > 0, "q_score_shadow"],
        errors="coerce",
    ).dropna()
    benign_energy = pd.to_numeric(
        rescored.loc[pd.to_numeric(rescored.get("malicious_visible", 0.0), errors="coerce").fillna(0.0) <= 0, "qbm_energy"],
        errors="coerce",
    ).dropna()
    malicious_energy = pd.to_numeric(
        rescored.loc[pd.to_numeric(rescored.get("malicious_visible", 0.0), errors="coerce").fillna(0.0) > 0, "qbm_energy"],
        errors="coerce",
    ).dropna()

    if not benign_energy.empty:
        benign_median = float(benign_energy.median())
        if not malicious_energy.empty:
            gap = abs(float(benign_energy.mean()) - float(malicious_energy.mean()))
            artifact["score_scale"] = float(np.clip(2.20 / max(gap, 0.8), 0.70, 2.20))
        else:
            artifact["score_scale"] = float(getattr(scorer, "score_scale", 1.28))
        target_median = 0.90
        target_logit = math.log(target_median / (1.0 - target_median))
        artifact["score_offset"] = float((artifact["score_scale"] * benign_median) - target_logit)
        verifier.apply_calibration(artifact)
        rescored = _apply_qbm_calibration_to_records(verifier, records)
        score_source = _select_score_source(rescored)
        benign_scores = pd.to_numeric(
            score_source.loc[pd.to_numeric(score_source.get("malicious_visible", 0.0), errors="coerce").fillna(0.0) <= 0, "q_score_shadow"],
            errors="coerce",
        ).dropna()
        malicious_scores = pd.to_numeric(
            score_source.loc[pd.to_numeric(score_source.get("malicious_visible", 0.0), errors="coerce").fillna(0.0) > 0, "q_score_shadow"],
            errors="coerce",
        ).dropna()

    threshold = float(verifier.qbm_threshold)
    if mode == "contrastive" and not benign_scores.empty and not malicious_scores.empty:
        candidates = sorted({float(v) for v in pd.concat([benign_scores, malicious_scores], ignore_index=True).tolist()})
        best_score = -float("inf")
        for candidate in candidates:
            benign_accept = (benign_scores >= candidate).mean()
            malicious_reject = (malicious_scores < candidate).mean()
            objective = float(benign_accept + malicious_reject)
            if objective > best_score:
                best_score = objective
                threshold = float(candidate)
    elif not benign_scores.empty:
        threshold = float(np.quantile(np.asarray(benign_scores, dtype=float), quantile))

    artifact["qbm_threshold"] = float(clamp01(threshold))
    artifact["qbm_threshold_stage"] = str(stage)
    artifact["qbm_threshold_quantile"] = float(quantile)
    artifact["shadow_samples"] = int(len(rescored))
    artifact["eligible_samples"] = int(len(score_source))
    artifact["benign_samples"] = int(len(benign_scores))
    artifact["malicious_samples"] = int(len(malicious_scores))
    artifact["benign_energy_mean"] = float(benign_energy.mean()) if not benign_energy.empty else float("nan")
    artifact["benign_energy_std"] = float(benign_energy.std(ddof=0)) if len(benign_energy) > 1 else 0.0
    artifact["malicious_energy_mean"] = float(malicious_energy.mean()) if not malicious_energy.empty else float("nan")
    artifact["malicious_energy_std"] = float(malicious_energy.std(ddof=0)) if len(malicious_energy) > 1 else 0.0
    artifact["benign_q_score_quantile"] = float(np.quantile(np.asarray(benign_scores, dtype=float), quantile)) if not benign_scores.empty else float("nan")
    artifact["feature_schema_version"] = FORENSIC_FEATURE_SCHEMA_VERSION
    artifact["reference_mode"] = FORENSIC_REFERENCE_MODE
    artifact["expected_context_mode"] = FORENSIC_EXPECTED_CONTEXT_MODE
    artifact["feature_keys"] = list(getattr(scorer, "feature_keys", FORENSIC_FEATURE_KEYS))
    return artifact


def _load_qbm_calibration_artifact(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_qbm_calibration_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(artifact), f, indent=2, ensure_ascii=False)


def _ensure_qbm_calibration(
    *,
    verifier: Any,
    cfg: Dict[str, Any],
    features: pd.DataFrame,
    stats: Dict[str, np.ndarray],
    node_ids: List[str],
    base_loss: float,
    partition_loss_extra: float,
    seed: int,
    scenario: str,
) -> Dict[str, Any]:
    if verifier is None or not isinstance(verifier, QBMVerifier):
        return {}
    vcfg = cfg.get("verification", {})

    artifact_path = _qbm_calibration_artifact_path(cfg, scenario, verifier)
    use_saved = bool(vcfg.get("qbm_use_saved_calibration", True))
    save_artifact = bool(vcfg.get("qbm_save_calibration", True))
    mode = str(vcfg.get("qbm_calibration_mode", "benign_only")).strip().lower() or "benign_only"
    quantile = clamp01(float(vcfg.get("qbm_threshold_quantile", 0.05)))
    min_samples = max(1, int(vcfg.get("qbm_threshold_min_samples", 20)))
    cal_windows = max(1, min(len(features), int(vcfg.get("qbm_threshold_calibration_windows", 300))))
    stage = str(vcfg.get("qbm_threshold_stage", "accepted_only")).strip().lower() or "accepted_only"
    calibration_attack_ids = ["A0"]

    if use_saved and artifact_path.exists():
        artifact = _load_qbm_calibration_artifact(artifact_path)
        verifier.apply_calibration(artifact)
        return {
            "qbm_threshold_precalibrated": 1.0,
            "qbm_calibration_loaded": 1.0,
            "qbm_calibration_artifact": str(artifact_path),
            "qbm_threshold_stage": str(artifact.get("qbm_threshold_stage", stage)),
            "qbm_threshold_quantile": float(artifact.get("qbm_threshold_quantile", quantile)),
            "qbm_threshold_value": float(artifact.get("qbm_threshold", verifier.qbm_threshold)),
            "qbm_threshold_samples_shadow": float(artifact.get("shadow_samples", artifact.get("benign_samples", 0.0))),
            "qbm_threshold_samples_eligible": float(artifact.get("eligible_samples", artifact.get("benign_samples", 0.0))),
            "qbm_feature_schema_version": str(artifact.get("feature_schema_version", "")),
            "qbm_calibration_version": str(artifact.get("qbm_calibration_version", "")),
            "qbm_reference_mode": str(artifact.get("reference_mode", "")),
            "qbm_expected_context_mode": str(artifact.get("expected_context_mode", "")),
            "qbm_calibration_attack_ids": "|".join(str(item) for item in artifact.get("calibration_attack_ids", [])),
            "qbm_calibration_config_hash": str(artifact.get("config_hash", "")),
        }

    if not bool(vcfg.get("qbm_auto_threshold_from_a0", True)):
        return {
            "qbm_threshold_precalibrated": 0.0,
            "qbm_calibration_loaded": 0.0,
            "qbm_calibration_artifact": str(artifact_path),
            "qbm_threshold_stage": str(stage),
            "qbm_threshold_quantile": float(quantile),
            "qbm_threshold_value": float(verifier.qbm_threshold),
            "qbm_threshold_samples_shadow": 0.0,
            "qbm_threshold_samples_eligible": 0.0,
        }

    records = _collect_qbm_shadow_records(
        verifier=verifier,
        cfg=cfg,
        features=features,
        stats=stats,
        node_ids=node_ids,
        base_loss=base_loss,
        partition_loss_extra=partition_loss_extra,
        seed=seed,
        attack_id="A0",
        max_windows=cal_windows,
    )
    if mode == "contrastive":
        attack_ids = [str(item).upper() for item in vcfg.get("qbm_contrastive_attack_ids", []) if str(item).strip()]
        for idx, attack_id in enumerate(attack_ids):
            calibration_attack_ids.append(str(attack_id).upper())
            attack_records = _collect_qbm_shadow_records(
                verifier=verifier,
                cfg=cfg,
                features=features,
                stats=stats,
                node_ids=node_ids,
                base_loss=base_loss,
                partition_loss_extra=partition_loss_extra,
                seed=seed + (idx + 1) * 101,
                attack_id=attack_id,
                max_windows=cal_windows,
            )
            if not attack_records.empty:
                records = pd.concat([records, attack_records], ignore_index=True)

    if len(records) < min_samples:
        return {
            "qbm_threshold_precalibrated": 0.0,
            "qbm_calibration_loaded": 0.0,
            "qbm_calibration_artifact": str(artifact_path),
            "qbm_threshold_stage": str(stage),
            "qbm_threshold_quantile": float(quantile),
            "qbm_threshold_value": float(verifier.qbm_threshold),
            "qbm_threshold_samples_shadow": float(len(records)),
            "qbm_threshold_samples_eligible": float(
                pd.to_numeric(records.get("base_accept", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
            ),
        }

    artifact = _fit_qbm_calibration_artifact(
        verifier=verifier,
        records=records,
        mode=mode,
        quantile=quantile,
        stage=stage,
    )
    artifact["artifact_version"] = 2
    artifact["qbm_feature_set_version"] = FORENSIC_FEATURE_SCHEMA_VERSION
    artifact["qbm_calibration_version"] = f"{mode}_v1"
    artifact["qbm_backend_mode"] = str(getattr(verifier._scorer, "backend_mode", "exact_state"))
    artifact["qbm_backend_label"] = str(getattr(verifier._scorer, "qbm_backend_name", "qiskit"))
    artifact["qbm_seed"] = int(getattr(verifier._scorer, "seed", seed))
    artifact["scenario"] = str(scenario)
    artifact["calibration_window_count"] = int(cal_windows)
    artifact["calibration_attack_ids"] = list(calibration_attack_ids)
    artifact["config_hash"] = stable_hash(
        {
            "scenario": str(scenario),
            "verification": cfg.get("verification", {}),
            "qbm": cfg.get("qbm", {}),
            "calibration_attack_ids": list(calibration_attack_ids),
            "calibration_window_count": int(cal_windows),
        }
    )
    verifier.apply_calibration(artifact)
    if save_artifact:
        _save_qbm_calibration_artifact(artifact_path, artifact)

    base_accept = pd.to_numeric(records.get("base_accept", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    return {
        "qbm_threshold_precalibrated": 1.0,
        "qbm_calibration_loaded": 0.0,
        "qbm_calibration_artifact": str(artifact_path),
        "qbm_threshold_stage": str(artifact.get("qbm_threshold_stage", stage)),
        "qbm_threshold_quantile": float(artifact.get("qbm_threshold_quantile", quantile)),
        "qbm_threshold_value": float(artifact.get("qbm_threshold", verifier.qbm_threshold)),
        "qbm_threshold_samples_shadow": float(artifact.get("shadow_samples", len(records))),
        "qbm_threshold_samples_eligible": float(artifact.get("eligible_samples", base_accept.sum())),
        "qbm_calibration_mode": str(mode),
        "qbm_benign_reference_fitted": 1.0 if artifact.get("benign_reference") else 0.0,
        "qbm_pair_terms_fitted": float(len(artifact.get("visible_pair_couplings", []))),
        "qbm_feature_schema_version": str(artifact.get("feature_schema_version", "")),
        "qbm_calibration_version": str(artifact.get("qbm_calibration_version", "")),
        "qbm_reference_mode": str(artifact.get("reference_mode", "")),
        "qbm_expected_context_mode": str(artifact.get("expected_context_mode", "")),
        "qbm_calibration_attack_ids": "|".join(str(item) for item in artifact.get("calibration_attack_ids", [])),
        "qbm_calibration_config_hash": str(artifact.get("config_hash", "")),
    }


def _safe_ratio(num: float, den: float, *, default: float = 0.0) -> float:
    den = float(den)
    if den <= 0.0:
        return float(default)
    return float(num) / den


def _numeric_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def _clip01_series(series: pd.Series) -> pd.Series:
    return series.astype(float).clip(lower=0.0, upper=1.0)


def _build_detection_score(df: pd.DataFrame) -> pd.Series:
    components: List[pd.Series] = []

    soft_score = _numeric_col(df, "soft_score")
    if not soft_score.isna().all():
        components.append((1.0 - _clip01_series(soft_score)).rename("soft_risk"))

    q_score_shadow = _numeric_col(df, "q_score_shadow")
    if not q_score_shadow.isna().all():
        components.append((1.0 - _clip01_series(q_score_shadow)).rename("qbm_shadow_risk"))

    q_score = _numeric_col(df, "q_score")
    if not q_score.isna().all():
        components.append((1.0 - _clip01_series(q_score)).rename("qbm_stage2_risk"))

    strongq_score = _numeric_col(df, "strongq_score")
    if not strongq_score.isna().all():
        components.append(_clip01_series(strongq_score).rename("strongq_risk"))

    fallback_parts: List[pd.Series] = []
    context_ratio = _numeric_col(df, "context_ratio")
    if not context_ratio.isna().all():
        fallback_parts.append((1.0 - _clip01_series(context_ratio)).rename("context_risk"))

    explanation_exact = _numeric_col(df, "s2_explanation_exact_match")
    if not explanation_exact.isna().all():
        fallback_parts.append((1.0 - _clip01_series(explanation_exact)).rename("s2_explanation_risk"))

    trust_variance = _numeric_col(df, "trust_variance")
    variance_tol = _numeric_col(df, "s2_variance_tol")
    if not trust_variance.isna().all():
        denom = variance_tol.where(variance_tol > 0.0, np.nan)
        variance_risk = (trust_variance / denom).clip(lower=0.0, upper=1.0)
        fallback_parts.append(variance_risk.fillna(0.0).rename("variance_risk"))

    for col in ("risk_hint", "risk_context_low", "risk_sim_low", "risk_tau_near", "gray_zone_flag", "policy_fired_hint"):
        part = _numeric_col(df, col)
        if not part.isna().all():
            fallback_parts.append(_clip01_series(part).rename(col))

    final_decision = _numeric_col(df, "final_decision")
    if not final_decision.isna().all():
        fallback_parts.append((1.0 - _clip01_series(final_decision)).rename("final_decision_risk"))

    commit_risk = (~df["commit"].astype(bool)).astype(float)
    fallback_parts.append(commit_risk.rename("reject_risk"))

    fallback = pd.concat(fallback_parts, axis=1).max(axis=1, skipna=True).fillna(0.0)
    if components:
        return pd.concat([*components, fallback.rename("fallback_risk")], axis=1).max(axis=1, skipna=True).fillna(0.0)
    return fallback


def _ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_score)
    y_true = y_true[mask].astype(int)
    y_score = y_score[mask].astype(float)

    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0 or y_score.size == 0:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "fpr_at_tpr95": float("nan"),
        }

    ranks = pd.Series(y_score).rank(method="average").to_numpy(dtype=float)
    pos_rank_sum = float(ranks[y_true == 1].sum())
    auroc = (pos_rank_sum - (pos * (pos + 1) / 2.0)) / max(pos * neg, 1)

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]

    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)
    distinct_idx = np.where(np.diff(y_score))[0]
    threshold_idx = np.r_[distinct_idx, y_true.size - 1]
    tps = tps[threshold_idx].astype(float)
    fps = fps[threshold_idx].astype(float)

    tpr = tps / max(pos, 1)
    fpr = fps / max(neg, 1)
    precision = tps / np.maximum(tps + fps, 1.0)
    recall = tpr

    recall_prev = np.r_[0.0, recall[:-1]]
    auprc = float(np.sum((recall - recall_prev) * precision))

    tpr95_mask = tpr >= 0.95
    fpr_at_tpr95 = float(np.min(fpr[tpr95_mask])) if np.any(tpr95_mask) else float("nan")
    return {
        "auroc": float(auroc),
        "auprc": auprc,
        "fpr_at_tpr95": fpr_at_tpr95,
    }


def _window_classification_metrics(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    if df.empty or "commit" not in df.columns:
        empty_int = pd.Series(dtype=int)
        empty_float = pd.Series(dtype=float)
        return {
            "actual_attack": empty_int,
            "predicted_attack": empty_int,
            "detection_score": empty_float,
            "tp_window": empty_int,
            "fp_window": empty_int,
            "tn_window": empty_int,
            "fn_window": empty_int,
            "precision": float("nan"),
            "recall": float("nan"),
            "f1_score": float("nan"),
            "mcc": float("nan"),
            "detection_delay_windows": float("nan"),
            "cost_sensitive_score": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "fpr_at_tpr95": float("nan"),
        }

    eval_cfg = dict(cfg.get("evaluation", {}))
    label_source = str(eval_cfg.get("label_source", "malicious_visible")).strip().lower()
    prediction_source = str(eval_cfg.get("prediction_source", "reject")).strip().lower()
    score_threshold = clamp01(float(eval_cfg.get("score_threshold", 0.50)))
    cost_cfg = dict(eval_cfg.get("cost_weights", {}))
    fp_cost = max(0.0, float(cost_cfg.get("false_positive", 1.0)))
    fn_cost = max(0.0, float(cost_cfg.get("false_negative", 5.0)))

    label_map = {
        "malicious_visible": "malicious_visible",
        "malicious_injected": "malicious_injected",
    }
    label_col = label_map.get(label_source, "malicious_visible")
    y_true = (_numeric_col(df, label_col).fillna(0.0) > 0.0).astype(int)

    detection_score = _clip01_series(_build_detection_score(df)).fillna(0.0)
    if prediction_source == "policy_fired":
        y_pred = df["policy_fired"].astype(bool).astype(int)
    elif prediction_source == "score":
        y_pred = (detection_score >= score_threshold).astype(int)
    else:
        y_pred = (~df["commit"].astype(bool)).astype(int)

    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)
    fn_mask = (y_true == 1) & (y_pred == 0)

    tp = int(tp_mask.sum())
    fp = int(fp_mask.sum())
    tn = int(tn_mask.sum())
    fn = int(fn_mask.sum())

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1_score = _safe_ratio(2.0 * precision * recall, precision + recall)
    mcc_den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0))
    mcc = _safe_ratio((tp * tn) - (fp * fn), mcc_den)

    weighted_error = (fp_cost * fp) + (fn_cost * fn)
    weighted_norm = (fp_cost * int((y_true == 0).sum())) + (fn_cost * int((y_true == 1).sum()))
    cost_sensitive_score = 1.0 - _safe_ratio(weighted_error, weighted_norm, default=0.0)
    cost_sensitive_score = clamp01(cost_sensitive_score)

    positive_idx = np.flatnonzero(y_true.to_numpy(dtype=int) == 1)
    detected_idx = np.flatnonzero(tp_mask.to_numpy(dtype=bool))
    if positive_idx.size > 0 and detected_idx.size > 0:
        detection_delay_windows = float(max(0, int(detected_idx[0]) - int(positive_idx[0])))
    else:
        detection_delay_windows = float("nan")

    ranking = _ranking_metrics(
        y_true.to_numpy(dtype=int),
        detection_score.to_numpy(dtype=float),
    )

    return {
        "actual_attack": y_true.astype(int),
        "predicted_attack": y_pred.astype(int),
        "detection_score": detection_score.astype(float),
        "tp_window": tp_mask.astype(int),
        "fp_window": fp_mask.astype(int),
        "tn_window": tn_mask.astype(int),
        "fn_window": fn_mask.astype(int),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "mcc": float(mcc),
        "detection_delay_windows": float(detection_delay_windows),
        "cost_sensitive_score": float(cost_sensitive_score),
        **ranking,
    }


def summarize_window_metrics(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    scenario: str,
    attack_id: str,
    verifier_name: str,
    malicious_injected: int,
    malicious_accepted: int,
    false_trust_committed: int,
    false_trust_attempted: int,
    collapse_count: int,
    first_malicious_step: Optional[int],
    first_detected_step: Optional[int],
    n_gray_zone: int,
    n_strongq_called: int,
    n_flip_mev_to_reject: int,
    n_flip_mev_to_accept: int,
    n_strongq_agree: int,
    n_strongq_disagree: int,
) -> SimulationSummary:
    window_metrics = _window_classification_metrics(df, cfg)
    asr = (malicious_accepted / malicious_injected) if malicious_injected > 0 else 0.0
    ftr = (false_trust_committed / false_trust_attempted) if false_trust_attempted > 0 else 0.0
    tcp = (collapse_count / len(df)) if len(df) > 0 else 0.0
    if first_malicious_step is None or first_detected_step is None:
        ttd = float("nan")
    else:
        ttd = float(max(0, first_detected_step - first_malicious_step))

    return SimulationSummary(
        scenario=scenario,
        attack_id=attack_id,
        verifier_name=verifier_name,
        processed_tps_mean=float(df["processed_tps"].mean()),
        latency_ms_mean=float(df["latency_ms"].mean()),
        backlog_max=float(df["backlog"].max()),
        dropped_sum=float(df["dropped"].sum()),
        dropped_by_verification_sum=float(df["dropped_by_verification"].sum()),
        dropped_by_network_sum=float(df["dropped_by_network"].sum()),
        dropped_by_overflow_sum=float(df["dropped_by_overflow"].sum()),
        policy_fired_ratio=float(df["policy_fired"].mean()),
        asr=float(asr),
        ftr=float(ftr),
        tcp=float(tcp),
        ttd_windows=float(ttd),
        auroc=float(window_metrics["auroc"]),
        auprc=float(window_metrics["auprc"]),
        mcc=float(window_metrics["mcc"]),
        precision=float(window_metrics["precision"]),
        recall=float(window_metrics["recall"]),
        f1_score=float(window_metrics["f1_score"]),
        fpr_at_tpr95=float(window_metrics["fpr_at_tpr95"]),
        detection_delay_windows=float(window_metrics["detection_delay_windows"]),
        cost_sensitive_score=float(window_metrics["cost_sensitive_score"]),
        n_gray_zone=int(n_gray_zone),
        n_strongq_called=int(n_strongq_called),
        n_flip_mev_to_reject=int(n_flip_mev_to_reject),
        n_flip_mev_to_accept=int(n_flip_mev_to_accept),
        n_strongq_agree=int(n_strongq_agree),
        n_strongq_disagree=int(n_strongq_disagree),
    )


def _upsert_csv(path: Path, key_col: str, key_value: str, row_dict: Dict[str, Any]) -> None:
    if path.exists():
        try:
            old = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            old = pd.DataFrame(columns=[key_col])
        if key_col not in old.columns:
            old = pd.DataFrame(columns=[key_col])
        old = old[old[key_col].notna()].copy()
        old[key_col] = old[key_col].astype(str)
        value_cols = [c for c in old.columns if c != key_col]
        if value_cols:
            old = old[old[value_cols].notna().any(axis=1)]
        old = old[old[key_col] != key_value]
        new = pd.concat([old, pd.DataFrame([row_dict])], ignore_index=True)
    else:
        new = pd.DataFrame([row_dict])
    new.to_csv(path, index=False)


def run_simulation(cfg: Dict[str, Any], max_windows: Optional[int] = None) -> Tuple[pd.DataFrame, SimulationSummary]:
    cfg = copy.deepcopy(cfg)
    attack_id_for_window = str(cfg.get("experiments", {}).get("attack_id", "A0")).upper()
    if max_windows is not None and attack_id_for_window != "A0":
        cfg.setdefault("experiments", {})
        iw = cfg["experiments"].setdefault("injection_window", {})
        start_w = int(iw.get("start_window", 0))
        end_w = int(iw.get("end_window", 0))
        if start_w >= int(max_windows) or end_w <= start_w:
            new_start = max(5, int(max_windows * 0.35))
            new_end = max(new_start + 1, int(max_windows * 0.75))
            iw["start_window"] = new_start
            iw["end_window"] = new_end

    features = load_feature_windows(cfg)
    if max_windows is not None:
        features = features.head(int(max_windows)).copy()

    scenario = str(cfg.get("experiments", {}).get("scenario", "S3")).upper()
    seed = int(cfg.get("project", {}).get("seed", 42))
    rng = np.random.default_rng(seed)
    stats = build_feature_statistics(features, train_fraction=float(cfg.get("inference", {}).get("train_fraction", 0.70)))
    node_ids = build_node_ids(cfg)
    verifier, verifier_name = build_verifier(cfg, scenario)
    sim_cfg = cfg.get("simulation", {})
    base_loss = float(cfg.get("blockchain_net", {}).get("base_loss_rate", 0.01))
    base_delay = float(cfg.get("blockchain_net", {}).get("base_delay_ms", 50.0))
    capacity_base = float(sim_cfg.get("base_capacity_tps", 180.0))
    max_backlog = float(sim_cfg.get("max_backlog", 350000.0))
    partition_loss_extra = float(sim_cfg.get("partition_loss_extra", 0.06))
    tcp_backlog_threshold = float(sim_cfg.get("tcp_backlog_threshold", 250000.0))
    tcp_drop_ratio_threshold = float(sim_cfg.get("tcp_drop_ratio_threshold", 0.80))
    overhead_map = sim_cfg.get("overhead_mult", {})
    overhead_mult_base = float(overhead_map.get(verifier_name, overhead_map.get("none", 1.0)))
    overhead_mult_s3 = float(overhead_map.get("s3_mev", overhead_mult_base))
    overhead_mult_qbm = float(overhead_map.get("qbm_verifier", max(overhead_mult_base, overhead_mult_s3)))
    overhead_mult_strongq = float(overhead_map.get("strongq_verifier", max(overhead_mult_base, overhead_mult_s3)))
    pipeline_mode_runtime = str(getattr(verifier, "pipeline_mode", cfg.get("verification", {}).get("pipeline_mode", "auto"))) if verifier is not None else "none"

    plan = build_attack_plan(cfg)
    attacker = AttackGenerator(plan)
    history = VerificationHistory(replay_buffer_max=2048)
    prev_hash_by_node: Dict[str, str] = {}
    verifier_runtime: Dict[str, Any] = {}
    if scenario == "S3" and verifier is not None and isinstance(verifier, S3MEVVerifier):
        verifier_runtime["verification_pipeline_mode"] = str(pipeline_mode_runtime)
        verifier_runtime.update(
            _precalibrate_tau_from_a0(
                verifier=verifier,
                cfg=cfg,
                features=features,
                stats=stats,
                node_ids=node_ids,
                base_loss=base_loss,
                partition_loss_extra=partition_loss_extra,
                seed=seed,
            )
        )
        verifier_runtime.update(
            _ensure_qbm_calibration(
                verifier=verifier,
                cfg=cfg,
                features=features,
                stats=stats,
                node_ids=node_ids,
                base_loss=base_loss,
                partition_loss_extra=partition_loss_extra,
                seed=seed,
                scenario=scenario,
            )
        )
        verifier_runtime.update(
            _precalibrate_strongq_threshold(
                verifier=verifier,
                cfg=cfg,
                features=features,
                stats=stats,
                node_ids=node_ids,
                base_loss=base_loss,
                partition_loss_extra=partition_loss_extra,
                plan=plan,
                seed=seed,
            )
        )

    rows: List[Dict[str, Any]] = []
    backlog = 0.0
    malicious_injected = 0
    malicious_accepted = 0
    false_trust_committed = 0
    false_trust_attempted = 0
    first_malicious_step: Optional[int] = None
    first_detected_step: Optional[int] = None
    collapse_count = 0
    n_gray_zone = 0
    n_strongq_called = 0
    n_flip_mev_to_reject = 0
    n_flip_mev_to_accept = 0
    n_strongq_agree = 0
    n_strongq_disagree = 0

    for step, (_, row) in enumerate(features.iterrows()):
        batch = generate_evidence_batch(
            row,
            step_index=step,
            cfg=cfg,
            stats=stats,
            node_ids=node_ids,
            prev_hash_by_node=prev_hash_by_node,
            rng=rng,
        )

        batch, net_delta, mal_count = attacker.apply(batch, history, rng)
        if net_delta:
            batch.network_state.update(net_delta)

        verify_batch, network_stats = apply_network_visibility(
            batch,
            base_loss_rate=base_loss,
            partition_loss_extra=partition_loss_extra,
            rng=rng,
        )
        calibration_mode = bool(
            scenario == "S3"
            and verifier is not None
            and bool(getattr(verifier, "auto_tau_from_a0", False))
            and plan.attack_id == "A0"
        )
        verify_batch.network_state["calibration_mode"] = calibration_mode
        policy_fired_hint = _policy_hint_from_network(verify_batch, network_stats, base_loss)
        verify_batch.network_state["policy_fired_hint"] = bool(policy_fired_hint)
        malicious_visible = verify_batch.malicious_count

        pre_shadow_scores: Dict[str, Any] = {}
        if isinstance(verifier, QBMVerifier):
            pre_shadow_scores = dict(verifier.shadow_score_batch(verify_batch))

        verify_result = VerifyResult(accept=True, reason="raft commit", scores={"quorum_ratio": 1.0}, drop_type=DropType.NONE)
        if verifier is not None:
            verify_result = verifier.verify(verify_batch, history)
        scores = dict(verify_result.scores or {})
        for key, value in pre_shadow_scores.items():
            if key == "q_score":
                continue
            scores.setdefault(key, value)
        n_gray_zone += int(float(scores.get("gray_zone_flag", 0.0)) > 0.5)
        n_strongq_called += int(float(scores.get("strongq_called", 0.0)) > 0.5)
        n_flip_mev_to_reject += int(float(scores.get("flip_mev_to_reject", 0.0)) > 0.5)
        n_flip_mev_to_accept += int(float(scores.get("flip_mev_to_accept", 0.0)) > 0.5)
        n_strongq_agree += int(float(scores.get("strongq_agree", 0.0)) > 0.5)
        n_strongq_disagree += int(float(scores.get("strongq_disagree", 0.0)) > 0.5)
        commit = bool(verify_result.accept)
        strongq_called_flag = bool(float(scores.get("strongq_called", 0.0)) > 0.5)

        offered = _offered_load(row, stats, rng)
        dropped_by_network = offered * (1.0 - float(network_stats.get("visibility_ratio", 0.0)))
        dropped_by_verification = 0.0
        if not commit:
            dropped_by_verification = max(0.0, offered - dropped_by_network)
        admitted = max(0.0, offered - dropped_by_network - dropped_by_verification)

        effective_overhead_mult = overhead_mult_base
        # Call-aware overhead model:
        # - strongq_verifier pays strong overhead only when StrongQ is actually called,
        #   otherwise pays S3-MEV overhead.
        # - s3_mev with optional StrongQ fallback pays strong overhead on called windows only.
        if verifier_name == "strongq_verifier":
            effective_overhead_mult = overhead_mult_strongq if strongq_called_flag else overhead_mult_s3
        elif verifier_name == "s3_mev" and strongq_called_flag:
            effective_overhead_mult = max(overhead_mult_base, overhead_mult_strongq)
        elif verifier_name == "s3_strongq":
            effective_overhead_mult = overhead_mult_strongq if strongq_called_flag else overhead_mult_s3
        elif verifier_name == "s3_qbm":
            effective_overhead_mult = max(overhead_mult_base, overhead_mult_qbm)
        elif verifier_name == "s3_qbm_strongq":
            effective_overhead_mult = max(overhead_mult_qbm, overhead_mult_strongq) if strongq_called_flag else max(overhead_mult_base, overhead_mult_qbm)

        # Overhead is a cost: higher overhead must reduce effective capacity.
        capacity = capacity_base / max(effective_overhead_mult, 1.0e-9)
        queue_total = backlog + admitted
        processed = min(queue_total, capacity)
        backlog = max(0.0, queue_total - processed)
        dropped_by_overflow = 0.0
        if backlog > max_backlog:
            dropped_by_overflow = backlog - max_backlog
            backlog = max_backlog

        latency_ms = base_delay + (backlog / max(capacity, 1e-9)) * 100.0
        dropped = dropped_by_network + dropped_by_verification + dropped_by_overflow
        policy_fired = bool(policy_fired_hint) or (not commit)

        if commit:
            history.record_commit(verify_batch)

        if mal_count > 0:
            malicious_injected += int(mal_count)
            if first_malicious_step is None:
                first_malicious_step = step
            if plan.attack_id in FALSE_TRUST_ATTACK_IDS:
                false_trust_attempted += int(mal_count)
            if commit and malicious_visible > 0:
                malicious_accepted += int(malicious_visible)
                if plan.attack_id in FALSE_TRUST_ATTACK_IDS:
                    false_trust_committed += int(malicious_visible)
            elif (not commit) and malicious_visible > 0 and first_detected_step is None:
                first_detected_step = step

        drop_ratio = dropped / max(offered, 1.0)
        if backlog > tcp_backlog_threshold or drop_ratio > tcp_drop_ratio_threshold:
            collapse_count += 1

        rows.append(
            {
                "window_id": int(batch.window_id),
                "offered": float(offered),
                "admitted": float(admitted),
                "processed_tps": float(processed),
                "backlog": float(backlog),
                "latency_ms": float(latency_ms),
                "dropped": float(dropped),
                "dropped_by_verification": float(dropped_by_verification),
                "dropped_by_network": float(dropped_by_network),
                "dropped_by_overflow": float(dropped_by_overflow),
                "policy_fired": bool(policy_fired),
                "policy_fired_hint": float(scores.get("policy_fired_hint", float(policy_fired_hint))),
                "overhead_mult": float(effective_overhead_mult),
                "overhead_mult_base": float(overhead_mult_base),
                "overhead_strongq_called": float(1.0 if strongq_called_flag else 0.0),
                "verification_mode": "logging_only" if (scenario == "S1" and verifier is None) else ("none" if verifier is None else "active"),
                "verification_pipeline_mode": str(pipeline_mode_runtime),
                "commit": bool(commit),
                "verify_reason": verify_result.reason,
                "corr": float(scores.get("corr", np.nan)),
                "corr_pairwise": float(scores.get("corr_pairwise", np.nan)),
                "corr_scaled": float(scores.get("corr_scaled", np.nan)),
                "sim": float(scores.get("sim", np.nan)),
                "sim_scaled": float(scores.get("sim_scaled", np.nan)),
                "context_ratio": float(scores.get("context_ratio", np.nan)),
                "trust_mean": float(scores.get("trust_mean", np.nan)),
                "trust_variance": float(scores.get("trust_variance", np.nan)),
                "node_conf_mean": float(scores.get("node_conf_mean", np.nan)),
                "node_conf_std": float(scores.get("node_conf_std", np.nan)),
                "node_conf_min": float(scores.get("node_conf_min", np.nan)),
                "node_conf_max": float(scores.get("node_conf_max", np.nan)),
                "node_anom_mean": float(scores.get("node_anom_mean", np.nan)),
                "node_anom_std": float(scores.get("node_anom_std", np.nan)),
                "node_unc_mean": float(scores.get("node_unc_mean", np.nan)),
                "node_unc_std": float(scores.get("node_unc_std", np.nan)),
                "pair_overlap_mean": float(scores.get("pair_overlap_mean", np.nan)),
                "pair_overlap_min": float(scores.get("pair_overlap_min", np.nan)),
                "pair_orderdist_mean": float(scores.get("pair_orderdist_mean", np.nan)),
                "pair_orderdist_max": float(scores.get("pair_orderdist_max", np.nan)),
                "pair_ctxmatch_mean": float(scores.get("pair_ctxmatch_mean", np.nan)),
                "pair_seqgap_max": float(scores.get("pair_seqgap_max", np.nan)),
                "pair_timegap_max": float(scores.get("pair_timegap_max", np.nan)),
                "pair_weightl1_mean": float(scores.get("pair_weightl1_mean", np.nan)),
                "policy_fired_ratio_signal": float(scores.get("policy_fired_ratio", np.nan)),
                "soft_score": float(scores.get("soft_score", np.nan)),
                "tau": float(scores.get("tau", np.nan)),
                "gray_margin": float(scores.get("gray_margin", np.nan)),
                "tau_calibration_samples": float(scores.get("tau_calibration_samples", np.nan)),
                "tau_calibrated": float(scores.get("tau_calibrated", np.nan)),
                "tau_quantile": float(scores.get("tau_quantile", np.nan)),
                "tau_source": str(scores.get("tau_source", "")),
                "decision_path": str(scores.get("decision_path", "")),
                "decision_stage": str(scores.get("decision_stage", "")),
                "reject_code": str(scores.get("reject_code", "")),
                "base_accept": float(scores.get("base_accept", np.nan)),
                "base_decision_path": str(scores.get("base_decision_path", "")),
                "gray_zone_flag": float(scores.get("gray_zone_flag", 0.0)),
                "mev_decision": float(scores.get("mev_decision", np.nan)),
                "final_decision": float(scores.get("final_decision", 1.0 if commit else 0.0)),
                "strongq_called": float(scores.get("strongq_called", 0.0)),
                "strongq_agree": float(scores.get("strongq_agree", 0.0)),
                "strongq_disagree": float(scores.get("strongq_disagree", 0.0)),
                "flip_mev_to_reject": float(scores.get("flip_mev_to_reject", 0.0)),
                "flip_mev_to_accept": float(scores.get("flip_mev_to_accept", 0.0)),
                "strongq_gate_reason": str(scores.get("strongq_gate_reason", "")),
                "risk_hint": float(scores.get("risk_hint", 0.0)),
                "risk_context_low": float(scores.get("risk_context_low", 0.0)),
                "risk_sim_low": float(scores.get("risk_sim_low", 0.0)),
                "risk_tau_near": float(scores.get("risk_tau_near", 0.0)),
                "risk_reasons": str(scores.get("risk_reasons", "")),
                "q_score": float(scores.get("q_score", np.nan)),
                "q_score_shadow": float(scores.get("q_score_shadow", np.nan)),
                "qbm_score_raw": float(scores.get("qbm_score_raw", np.nan)),
                "qbm_score_adjusted": float(scores.get("qbm_score_adjusted", np.nan)),
                "qbm_energy": float(scores.get("qbm_energy", np.nan)),
                "qbm_energy_raw": float(scores.get("qbm_energy_raw", np.nan)),
                "qbm_energy_adjusted": float(scores.get("qbm_energy_adjusted", np.nan)),
                "qbm_energy_std": float(scores.get("qbm_energy_std", np.nan)),
                "qbm_uncertainty_proxy": float(scores.get("qbm_uncertainty_proxy", np.nan)),
                "qbm_shot_penalty_applied": float(scores.get("qbm_shot_penalty_applied", np.nan)),
                "qbm_shot_std": float(scores.get("qbm_shot_std", np.nan)),
                "qbm_circuit_depth": float(scores.get("qbm_circuit_depth", np.nan)),
                "qbm_circuit_size": float(scores.get("qbm_circuit_size", np.nan)),
                "qbm_n_qubits": float(scores.get("qbm_n_qubits", np.nan)),
                "qbm_layers": float(scores.get("qbm_layers", np.nan)),
                "qbm_hidden_count": float(scores.get("qbm_hidden_count", np.nan)),
                "qbm_feature_dim": float(scores.get("qbm_feature_dim", np.nan)),
                "qbm_pair_coupling_count": float(scores.get("qbm_pair_coupling_count", np.nan)),
                "qbm_visible_pair_weight_count": float(scores.get("qbm_visible_pair_weight_count", np.nan)),
                "qbm_feature_keys": str(scores.get("qbm_feature_keys", "")),
                "qbm_backend": str(scores.get("qbm_backend", "")),
                "qbm_backend_label": str(scores.get("qbm_backend_label", "")),
                "qbm_backend_mode": str(scores.get("qbm_backend_mode", "")),
                "qbm_uncertainty_mode": str(scores.get("qbm_uncertainty_mode", "")),
                "qbm_model_family": str(scores.get("qbm_model_family", "")),
                "qbm_model_note": str(scores.get("qbm_model_note", "")),
                "qbm_score_name": str(scores.get("qbm_score_name", "")),
                "qbm_visible_values": str(scores.get("qbm_visible_values", "")),
                "qbm_optional_signal_keys": str(scores.get("qbm_optional_signal_keys", "")),
                "qbm_feature_schema_version": str(scores.get("qbm_feature_schema_version", "")),
                "qbm_feature_set_version": str(scores.get("qbm_feature_set_version", "")),
                "qbm_calibration_version": str(scores.get("qbm_calibration_version", "")),
                "qbm_feature_directionality_mode": str(scores.get("qbm_feature_directionality_mode", "")),
                "qbm_reference_mode": str(scores.get("qbm_reference_mode", "")),
                "qbm_expected_context_mode": str(scores.get("qbm_expected_context_mode", "")),
                "qbm_noise_enabled": float(scores.get("qbm_noise_enabled", np.nan)),
                "qbm_measurement_shots": float(scores.get("qbm_measurement_shots", np.nan)),
                "qbm_template_reused": float(scores.get("qbm_template_reused", np.nan)),
                "qbm_batch_eval_enabled": float(scores.get("qbm_batch_eval_enabled", np.nan)),
                "qbm_shadow_available": float(scores.get("qbm_shadow_available", 0.0)),
                "qbm_stage2_eligible": float(scores.get("qbm_stage2_eligible", 0.0)),
                "qbm_stage2_veto": float(scores.get("qbm_stage2_veto", 0.0)),
                "qbm_stage2_decision": float(scores.get("qbm_stage2_decision", np.nan)),
                "qbm_invoked_by_base_accept": float(scores.get("qbm_invoked_by_base_accept", 0.0)),
                "qbm_invoked_by_gray_zone": float(scores.get("qbm_invoked_by_gray_zone", 0.0)),
                "qbm_risk_hint_level": float(scores.get("qbm_risk_hint_level", np.nan)),
                "qbm_threshold": float(scores.get("qbm_threshold", np.nan)),
                "qbm_expected_context": float(scores.get("qbm_expected_context", np.nan)),
                "qbm_signal_policy_safe": float(scores.get("qbm_policy_safe", np.nan)),
                "qbm_signal_gray_risk": float(scores.get("qbm_gray_risk", np.nan)),
                "qbm_feat_r_trust_sim_gap": float(scores.get("qbm_r_trust_sim_gap", np.nan)),
                "qbm_feat_r_overlap_spread": float(scores.get("qbm_r_overlap_spread", np.nan)),
                "qbm_feat_r_context_dev": float(scores.get("qbm_r_context_dev", np.nan)),
                "qbm_feat_r_policy_gray_contra": float(scores.get("qbm_r_policy_gray_contra", np.nan)),
                "qbm_feat_r_trust_sim_prod": float(scores.get("qbm_r_trust_sim_prod", np.nan)),
                "qbm_feat_r_benign_delta": float(scores.get("qbm_r_benign_delta", np.nan)),
                "qbm_feat_r_temporal_inconsistency": float(scores.get("qbm_r_temporal_inconsistency", np.nan)),
                "qbm_feat_r_witness_disagree": float(scores.get("qbm_r_witness_disagree", np.nan)),
                "strongq_score": float(scores.get("strongq_score", np.nan)),
                "strongq_decision": float(scores.get("strongq_decision", np.nan)),
                "strongq_witness": float(scores.get("strongq_witness", np.nan)),
                "strongq_shot_std": float(scores.get("strongq_shot_std", np.nan)),
                "strongq_ci_low": float(scores.get("strongq_ci_low", np.nan)),
                "strongq_ci_high": float(scores.get("strongq_ci_high", np.nan)),
                "strongq_shots": float(scores.get("strongq_shots", np.nan)),
                "strongq_feature_mean": float(scores.get("strongq_feature_mean", np.nan)),
                "strongq_feature_std": float(scores.get("strongq_feature_std", np.nan)),
                "strongq_feature_dim": float(scores.get("strongq_feature_dim", np.nan)),
                "s2_explanation_exact_match": float(scores.get("explanation_exact_match", np.nan)),
                "s2_variance_tol": float(scores.get("variance_tol", np.nan)),
                "s2_explanation_order_match": float(scores.get("explanation_order_match", np.nan)),
                "s2_explanation_length_match": float(scores.get("explanation_length_match", np.nan)),
                "s2_explanation_schema_whitelist_match": float(
                    scores.get("explanation_schema_whitelist_match", np.nan)
                ),
                "s2_explanation_schema_violations": float(scores.get("explanation_schema_violations", np.nan)),
                "s2_explanation_expected_topk_len": float(
                    scores.get("explanation_expected_topk_len", np.nan)
                ),
                "attack_id": plan.attack_id if mal_count > 0 else "A0",
                "malicious_injected": int(mal_count),
                "malicious_visible": int(malicious_visible),
                "malicious_committed": int(malicious_visible if commit else 0),
                "false_trust_candidate": int(mal_count if plan.attack_id in FALSE_TRUST_ATTACK_IDS else 0),
                "false_trust_committed": int(malicious_visible if (commit and plan.attack_id in FALSE_TRUST_ATTACK_IDS) else 0),
                "network_visibility_ratio": float(network_stats.get("visibility_ratio", np.nan)),
                "effective_loss_rate": float(network_stats.get("loss_rate", np.nan)),
            }
        )

    sim_df = pd.DataFrame(rows)
    window_metrics = _window_classification_metrics(sim_df, cfg)
    for col in (
        "actual_attack",
        "predicted_attack",
        "detection_score",
        "tp_window",
        "fp_window",
        "tn_window",
        "fn_window",
    ):
        sim_df[col] = window_metrics[col]
    effective_attack_id = plan.attack_id if malicious_injected > 0 else "A0"
    summary = summarize_window_metrics(
        sim_df,
        cfg,
        scenario=scenario,
        attack_id=effective_attack_id,
        verifier_name=verifier_name,
        malicious_injected=malicious_injected,
        malicious_accepted=malicious_accepted,
        false_trust_committed=false_trust_committed,
        false_trust_attempted=false_trust_attempted,
        collapse_count=collapse_count,
        first_malicious_step=first_malicious_step,
        first_detected_step=first_detected_step,
        n_gray_zone=n_gray_zone,
        n_strongq_called=n_strongq_called,
        n_flip_mev_to_reject=n_flip_mev_to_reject,
        n_flip_mev_to_accept=n_flip_mev_to_accept,
        n_strongq_agree=n_strongq_agree,
        n_strongq_disagree=n_strongq_disagree,
    )
    sim_df.attrs["verifier_runtime"] = verifier_runtime
    return sim_df, summary


def save_outputs(cfg: Dict[str, Any], sim_df: pd.DataFrame, summary: SimulationSummary) -> Dict[str, str]:
    results_dir = Path(cfg.get("project", {}).get("results_dir", "results"))
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    sim_name = f"sim_{summary.scenario}_{summary.attack_id}_{summary.verifier_name}.csv"
    sim_path = tables_dir / sim_name
    sim_df.to_csv(sim_path, index=False)

    extended_summary_path = tables_dir / "summary_end2end_extended.csv"
    key = f"{summary.scenario}_{summary.attack_id}_{summary.verifier_name}"
    _upsert_csv(extended_summary_path, "key", key, {"key": key, **asdict(summary)})

    meta_path = tables_dir / f"meta_{summary.scenario}_{summary.attack_id}_{summary.verifier_name}.json"
    vcfg = cfg.get("verification", {})
    bcfg = cfg.get("blockchain_net", {})
    tau_series = pd.to_numeric(sim_df.get("tau", pd.Series([], dtype=float)), errors="coerce").dropna()
    gray_series = pd.to_numeric(sim_df.get("gray_zone_flag", pd.Series([], dtype=float)), errors="coerce").fillna(0.0)
    strongq_called_series = pd.to_numeric(
        sim_df.get("strongq_called", pd.Series([], dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    qbm_threshold_series = pd.to_numeric(sim_df.get("qbm_threshold", pd.Series([], dtype=float)), errors="coerce").dropna()
    qbm_shadow_series = pd.to_numeric(sim_df.get("qbm_shadow_available", pd.Series([], dtype=float)), errors="coerce").fillna(0.0)
    qbm_stage2_series = pd.to_numeric(sim_df.get("qbm_stage2_eligible", pd.Series([], dtype=float)), errors="coerce").fillna(0.0)
    qbm_veto_series = pd.to_numeric(sim_df.get("qbm_stage2_veto", pd.Series([], dtype=float)), errors="coerce").fillna(0.0)
    runtime_attrs = dict(sim_df.attrs.get("verifier_runtime", {}))
    pipeline_mode_meta = str(runtime_attrs.get("verification_pipeline_mode", vcfg.get("pipeline_mode", "auto")))
    runtime_state = {
        "tau_final": float(tau_series.iloc[-1]) if not tau_series.empty else float("nan"),
        "tau_mean": float(tau_series.mean()) if not tau_series.empty else float("nan"),
        "gray_rate": float((gray_series > 0).mean()) if len(gray_series) > 0 else float("nan"),
        "strongq_called_rate": float((strongq_called_series > 0).mean()) if len(strongq_called_series) > 0 else float("nan"),
        "qbm_threshold_final": float(qbm_threshold_series.iloc[-1]) if not qbm_threshold_series.empty else float("nan"),
        "qbm_shadow_rate": float((qbm_shadow_series > 0).mean()) if len(qbm_shadow_series) > 0 else float("nan"),
        "qbm_stage2_rate": float((qbm_stage2_series > 0).mean()) if len(qbm_stage2_series) > 0 else float("nan"),
        "qbm_stage2_veto_rate": float((qbm_veto_series > 0).mean()) if len(qbm_veto_series) > 0 else float("nan"),
        "n_gray_zone": int(summary.n_gray_zone),
        "n_strongq_called": int(summary.n_strongq_called),
        "n_flip_mev_to_reject": int(summary.n_flip_mev_to_reject),
        "n_flip_mev_to_accept": int(summary.n_flip_mev_to_accept),
        "n_strongq_agree": int(summary.n_strongq_agree),
        "n_strongq_disagree": int(summary.n_strongq_disagree),
        "verification_pipeline_mode": str(pipeline_mode_meta),
    }
    runtime_state.update(runtime_attrs)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario": summary.scenario,
                "attack_id": summary.attack_id,
                "verifier": summary.verifier_name,
                "reproducibility": {
                    "epsilon_corr_threshold": float(vcfg.get("corr_threshold", float("nan"))),
                    "theta_explanation_threshold": float(vcfg.get("explanation_threshold", float("nan"))),
                    "node_count": int(bcfg.get("validators", 0)),
                    "s3_soft_tau": float(vcfg.get("s3_soft_tau", float("nan"))),
                    "s3_soft_gray_margin": float(vcfg.get("s3_soft_gray_margin", float("nan"))),
                    "s3_auto_tau_from_a0": bool(vcfg.get("s3_auto_tau_from_a0", True)),
                    "s3_tau_freeze_after_calibration": bool(vcfg.get("s3_tau_freeze_after_calibration", True)),
                    "s3_tau_calibration_windows": int(vcfg.get("s3_tau_calibration_windows", 0)),
                    "s3_tau_quantile": float(vcfg.get("s3_tau_quantile", float("nan"))),
                    "s3_tau_min_samples": int(vcfg.get("s3_tau_min_samples", 0)),
                    "s3_auto_gray_margin_from_a0": bool(vcfg.get("s3_auto_gray_margin_from_a0", True)),
                    "s3_target_gray_rate_a0": float(vcfg.get("s3_target_gray_rate_a0", float("nan"))),
                    "s3_gray_margin_min": float(vcfg.get("s3_gray_margin_min", float("nan"))),
                    "s3_gray_margin_max": float(vcfg.get("s3_gray_margin_max", float("nan"))),
                    "s3_grayzone_requires_policy_hint": bool(
                        vcfg.get("s3_grayzone_requires_policy_hint", False)
                    ),
                    "s3_grayzone_no_policy_action": str(vcfg.get("s3_grayzone_no_policy_action", "mev")),
                    "s3_strongq_mode": str(vcfg.get("s3_strongq_mode", "veto")),
                    "s3_risk_policy_ratio_floor": float(vcfg.get("s3_risk_policy_ratio_floor", float("nan"))),
                    "s3_risk_context_floor": float(vcfg.get("s3_risk_context_floor", float("nan"))),
                    "s3_risk_explanation_floor": float(vcfg.get("s3_risk_explanation_floor", float("nan"))),
                    "s3_risk_tau_band": float(vcfg.get("s3_risk_tau_band", float("nan"))),
                    # Paper notation aliases for the StrongQ call gate thresholds.
                    "p0": float(vcfg.get("s3_risk_policy_ratio_floor", float("nan"))),
                    "c0": float(vcfg.get("s3_risk_context_floor", float("nan"))),
                    "s0": float(vcfg.get("s3_risk_explanation_floor", float("nan"))),
                    "band": float(vcfg.get("s3_risk_tau_band", float("nan"))),
                    "s3_auto_strongq_threshold": bool(vcfg.get("s3_auto_strongq_threshold", True)),
                    "s3_strongq_threshold_quantile": float(vcfg.get("s3_strongq_threshold_quantile", float("nan"))),
                    "s3_strongq_threshold_min_samples": int(vcfg.get("s3_strongq_threshold_min_samples", 0)),
                    "qbm_threshold": float(vcfg.get("qbm_threshold", float("nan"))),
                    "qbm_auto_threshold_from_a0": bool(vcfg.get("qbm_auto_threshold_from_a0", True)),
                    "qbm_threshold_quantile": float(vcfg.get("qbm_threshold_quantile", float("nan"))),
                    "qbm_threshold_min_samples": int(vcfg.get("qbm_threshold_min_samples", 0)),
                    "qbm_threshold_calibration_windows": int(vcfg.get("qbm_threshold_calibration_windows", 0)),
                    "qbm_threshold_stage": str(vcfg.get("qbm_threshold_stage", "accepted_only")),
                    "qbm_calibration_mode": str(vcfg.get("qbm_calibration_mode", "benign_only")),
                    "qbm_contrastive_attack_ids": list(vcfg.get("qbm_contrastive_attack_ids", [])),
                    "qbm_use_saved_calibration": bool(vcfg.get("qbm_use_saved_calibration", True)),
                    "qbm_save_calibration": bool(vcfg.get("qbm_save_calibration", True)),
                    "qbm_calibration_artifact": vcfg.get("qbm_calibration_artifact"),
                    "qbm_feature_schema_version": str(cfg.get("qbm", {}).get("qbm_feature_set_version", FORENSIC_FEATURE_SCHEMA_VERSION)),
                    "qbm_feature_set_version": str(cfg.get("qbm", {}).get("qbm_feature_set_version", FORENSIC_FEATURE_SCHEMA_VERSION)),
                    "qbm_calibration_version": str(cfg.get("qbm", {}).get("qbm_calibration_version", "benign_only_v1")),
                    "qbm_backend_mode": str(cfg.get("qbm", {}).get("qiskit_backend_mode", "exact_state")),
                    "qbm_reference_mode": FORENSIC_REFERENCE_MODE,
                    "qbm_expected_context_mode": FORENSIC_EXPECTED_CONTEXT_MODE,
                    "verification_pipeline_mode": str(pipeline_mode_meta),
                },
                "runtime_verifier_state": runtime_state,
                "summary": asdict(summary),
                "config": cfg,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "sim_csv": str(sim_path),
        "summary_extended_csv": str(extended_summary_path),
        "meta_json": str(meta_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AIS trust-plane simulation (S0~S3, A0~A5 plus A4P).")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to yaml config.")
    parser.add_argument("--scenario", default=None, help="Override architecture scenario: S0|S1|S2|S3")
    parser.add_argument("--attack-id", default=None, help="Override attack id: A0..A5|A4P")
    parser.add_argument("--verifier", default=None, help="Override verifier impl: s2_strict|s3_mev|qbm|strongq")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--max-windows", type=int, default=None, help="Run only first N windows.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(
        cfg,
        scenario=args.scenario,
        attack_id=args.attack_id,
        verifier_impl=args.verifier,
        seed=args.seed,
    )
    sim_df, summary = run_simulation(cfg, max_windows=args.max_windows)
    paths = save_outputs(cfg, sim_df, summary)

    print(f"scenario={summary.scenario} attack={summary.attack_id} verifier={summary.verifier_name}")
    print(f"processed_tps_mean={summary.processed_tps_mean:.3f}")
    print(f"latency_ms_mean={summary.latency_ms_mean:.3f}")
    print(f"backlog_max={summary.backlog_max:.3f}")
    print(f"dropped_sum={summary.dropped_sum:.3f}")
    print(f"ASR={summary.asr:.4f} FTR={summary.ftr:.4f} TCP={summary.tcp:.4f} TTD={summary.ttd_windows}")
    print(
        f"AUROC={summary.auroc:.4f} AUPRC={summary.auprc:.4f} MCC={summary.mcc:.4f} "
        f"Precision={summary.precision:.4f} Recall={summary.recall:.4f} "
        f"F1={summary.f1_score:.4f} FPR@TPR95={summary.fpr_at_tpr95}"
    )
    print(
        f"DetectionDelay={summary.detection_delay_windows} "
        f"CostSensitiveScore={summary.cost_sensitive_score:.4f}"
    )
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
