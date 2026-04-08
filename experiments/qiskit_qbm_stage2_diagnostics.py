from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any, Dict, List

import pandas as pd

from qbm.train import apply_overrides, load_config, run_simulation, save_outputs


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df.get(column, pd.Series(dtype=float)), errors="coerce")


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    return _numeric_series(df, column).fillna(0.0) > 0.0


def _numeric_last(df: pd.DataFrame, column: str) -> float:
    series = _numeric_series(df, column).dropna()
    if series.empty:
        return float("nan")
    return float(series.iloc[-1])


def _backend_suffix(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _copy_backend_artifacts(paths: Dict[str, str], backend_mode: str) -> Dict[str, str]:
    copied = dict(paths)
    suffix = _backend_suffix(backend_mode)
    for key in ("sim_csv", "meta_json"):
        raw_path = copied.get(key)
        if not raw_path:
            continue
        src = Path(raw_path)
        if not src.exists():
            continue
        dst = src.with_name(f"{src.stem}_{suffix}{src.suffix}")
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        copied[key] = str(dst)
    return copied


def _describe_subset(
    *,
    scenario: str,
    attack_id: str,
    backend_mode: str,
    subset: str,
    scores: pd.Series,
    threshold: float,
) -> Dict[str, Any]:
    clean = pd.to_numeric(scores, errors="coerce").dropna()
    if clean.empty:
        return {
            "scenario": scenario,
            "attack_id": attack_id,
            "backend_mode": backend_mode,
            "subset": subset,
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "q05": float("nan"),
            "q25": float("nan"),
            "q50": float("nan"),
            "q75": float("nan"),
            "q95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "below_threshold_rate": float("nan"),
            "mean_minus_threshold": float("nan"),
        }
    below_rate = float((clean < threshold).mean()) if math.isfinite(threshold) else float("nan")
    return {
        "scenario": scenario,
        "attack_id": attack_id,
        "backend_mode": backend_mode,
        "subset": subset,
        "count": int(clean.shape[0]),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "q05": float(clean.quantile(0.05)),
        "q25": float(clean.quantile(0.25)),
        "q50": float(clean.quantile(0.50)),
        "q75": float(clean.quantile(0.75)),
        "q95": float(clean.quantile(0.95)),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "below_threshold_rate": below_rate,
        "mean_minus_threshold": float(clean.mean() - threshold) if math.isfinite(threshold) else float("nan"),
    }


def _prepare_cfg(
    cfg_path: str,
    *,
    backend_mode: str,
    attack_id: str,
    scenario: str,
    seed: int | None,
    qbm_threshold: float | None,
    max_windows: int | None,
    force_refit_calibration: bool,
) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    cfg = apply_overrides(cfg, scenario=scenario, attack_id=attack_id, verifier_impl="qbm", seed=seed)
    cfg.setdefault("qbm", {})
    cfg["qbm"]["qiskit_backend_mode"] = backend_mode
    cfg.setdefault("verification", {})
    if force_refit_calibration:
        cfg["verification"]["qbm_use_saved_calibration"] = False
        cfg["verification"]["qbm_save_calibration"] = True
    if qbm_threshold is not None:
        cfg["verification"]["qbm_threshold"] = float(qbm_threshold)
        cfg["verification"]["qbm_auto_threshold_from_a0"] = False

    cfg.setdefault("experiments", {})
    cfg["experiments"]["enable_injection"] = str(attack_id).upper() != "A0"
    cfg["experiments"]["attack_id"] = str(attack_id).upper()

    if max_windows is not None and str(attack_id).upper() != "A0":
        iw = cfg["experiments"].setdefault("injection_window", {})
        start_w = int(iw.get("start_window", 0))
        end_w = int(iw.get("end_window", 0))
        if start_w >= int(max_windows) or end_w <= start_w:
            new_start = max(5, int(max_windows * 0.35))
            new_end = max(new_start + 1, int(max_windows * 0.75))
            iw["start_window"] = new_start
            iw["end_window"] = new_end
    return cfg


def run_stage2_diagnostics(
    cfg_path: str,
    *,
    backend_modes: List[str],
    attack_ids: List[str],
    scenario: str,
    max_windows: int | None,
    seed: int | None,
    qbm_threshold: float | None,
    force_refit_calibration: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    dist_rows: List[Dict[str, Any]] = []
    window_frames: List[pd.DataFrame] = []

    for backend_mode in backend_modes:
        for attack_id in attack_ids:
            cfg = _prepare_cfg(
                cfg_path,
                backend_mode=backend_mode,
                attack_id=attack_id,
                scenario=scenario,
                seed=seed,
                qbm_threshold=qbm_threshold,
                max_windows=max_windows,
                force_refit_calibration=force_refit_calibration,
            )
            started_at = perf_counter()
            sim_df, summary = run_simulation(cfg, max_windows=max_windows)
            elapsed_s = perf_counter() - started_at
            paths = _copy_backend_artifacts(save_outputs(cfg, sim_df, summary), backend_mode)

            threshold = _numeric_last(sim_df, "qbm_threshold")
            shadow_scores = _numeric_series(sim_df, "q_score_shadow")
            stage2_scores = _numeric_series(sim_df, "q_score")
            malicious_mask = _bool_series(sim_df, "malicious_visible")
            base_accept_mask = _bool_series(sim_df, "base_accept")
            shadow_mask = shadow_scores.notna()
            eligible_mask = _bool_series(sim_df, "qbm_stage2_eligible")
            veto_mask = _bool_series(sim_df, "qbm_stage2_veto")
            benign_mask = ~malicious_mask

            runtime = dict(sim_df.attrs.get("verifier_runtime", {}))

            summary_rows.append(
                {
                    "scenario": summary.scenario,
                    "attack_id": summary.attack_id,
                    "backend_mode": backend_mode,
                    "verifier_name": summary.verifier_name,
                    "elapsed_s": float(elapsed_s),
                    "n_windows": int(len(sim_df)),
                    "malicious_visible_windows": int(malicious_mask.sum()),
                    "base_accept_windows": int(base_accept_mask.sum()),
                    "qbm_shadow_windows": int(shadow_mask.sum()),
                    "qbm_stage2_eligible_windows": int(eligible_mask.sum()),
                    "qbm_stage2_veto_windows": int(veto_mask.sum()),
                    "qbm_stage2_veto_windows_malicious": int((veto_mask & malicious_mask).sum()),
                    "qbm_stage2_veto_windows_benign": int((veto_mask & benign_mask).sum()),
                    "qbm_threshold_final": float(threshold),
                    "qbm_threshold_samples_shadow": float(runtime.get("qbm_threshold_samples_shadow", float("nan"))),
                    "qbm_threshold_samples_eligible": float(runtime.get("qbm_threshold_samples_eligible", float("nan"))),
                    "qbm_calibration_loaded": float(runtime.get("qbm_calibration_loaded", float("nan"))),
                    "qbm_calibration_artifact": str(runtime.get("qbm_calibration_artifact", "")),
                    "qbm_shadow_mean_benign_all": float(shadow_scores[benign_mask].dropna().mean()),
                    "qbm_shadow_mean_malicious_all": float(shadow_scores[malicious_mask].dropna().mean()),
                    "qbm_shadow_gap_all": float(
                        shadow_scores[benign_mask].dropna().mean() - shadow_scores[malicious_mask].dropna().mean()
                    ),
                    "qbm_shadow_mean_benign_eligible": float(shadow_scores[benign_mask & eligible_mask].dropna().mean()),
                    "qbm_shadow_mean_malicious_eligible": float(shadow_scores[malicious_mask & eligible_mask].dropna().mean()),
                    "qbm_shadow_gap_eligible": float(
                        shadow_scores[benign_mask & eligible_mask].dropna().mean()
                        - shadow_scores[malicious_mask & eligible_mask].dropna().mean()
                    ),
                    "qbm_shadow_below_threshold_rate_benign_all": float(
                        (shadow_scores[benign_mask].dropna() < threshold).mean()
                    ) if shadow_scores[benign_mask].dropna().shape[0] > 0 and math.isfinite(threshold) else float("nan"),
                    "qbm_shadow_below_threshold_rate_malicious_all": float(
                        (shadow_scores[malicious_mask].dropna() < threshold).mean()
                    ) if shadow_scores[malicious_mask].dropna().shape[0] > 0 and math.isfinite(threshold) else float("nan"),
                    "qbm_shadow_below_threshold_rate_benign_eligible": float(
                        (shadow_scores[benign_mask & eligible_mask].dropna() < threshold).mean()
                    ) if shadow_scores[benign_mask & eligible_mask].dropna().shape[0] > 0 and math.isfinite(threshold) else float("nan"),
                    "qbm_shadow_below_threshold_rate_malicious_eligible": float(
                        (shadow_scores[malicious_mask & eligible_mask].dropna() < threshold).mean()
                    ) if shadow_scores[malicious_mask & eligible_mask].dropna().shape[0] > 0 and math.isfinite(threshold) else float("nan"),
                    "qbm_veto_given_eligible_rate": float(veto_mask[eligible_mask].mean()) if eligible_mask.any() else float("nan"),
                    "qbm_veto_given_malicious_eligible_rate": float(veto_mask[malicious_mask & eligible_mask].mean())
                    if (malicious_mask & eligible_mask).any()
                    else float("nan"),
                    "qbm_veto_given_benign_eligible_rate": float(veto_mask[benign_mask & eligible_mask].mean())
                    if (benign_mask & eligible_mask).any()
                    else float("nan"),
                    "qbm_energy_mean_benign_all": float(_numeric_series(sim_df, "qbm_energy")[benign_mask].dropna().mean()),
                    "qbm_energy_mean_malicious_all": float(_numeric_series(sim_df, "qbm_energy")[malicious_mask].dropna().mean()),
                    "qbm_shot_std_mean": float(_numeric_series(sim_df, "qbm_shot_std").dropna().mean()),
                    "qbm_uncertainty_proxy_mean": float(_numeric_series(sim_df, "qbm_uncertainty_proxy").dropna().mean()),
                    "processed_tps_mean": float(summary.processed_tps_mean),
                    "latency_ms_mean": float(summary.latency_ms_mean),
                    "asr": float(summary.asr),
                    "ftr": float(summary.ftr),
                    "tcp": float(summary.tcp),
                    "ttd_windows": float(summary.ttd_windows),
                    **paths,
                }
            )

            subsets = {
                "all_benign_shadow": benign_mask & shadow_mask,
                "all_malicious_shadow": malicious_mask & shadow_mask,
                "eligible_benign_shadow": benign_mask & eligible_mask & shadow_mask,
                "eligible_malicious_shadow": malicious_mask & eligible_mask & shadow_mask,
                "veto_benign_shadow": benign_mask & veto_mask & shadow_mask,
                "veto_malicious_shadow": malicious_mask & veto_mask & shadow_mask,
                "stage2_benign_score": benign_mask & stage2_scores.notna(),
                "stage2_malicious_score": malicious_mask & stage2_scores.notna(),
            }
            for subset_name, subset_mask in subsets.items():
                source = stage2_scores if subset_name.startswith("stage2_") else shadow_scores
                dist_rows.append(
                    _describe_subset(
                        scenario=summary.scenario,
                        attack_id=summary.attack_id,
                        backend_mode=backend_mode,
                        subset=subset_name,
                        scores=source[subset_mask],
                        threshold=threshold,
                    )
                )

            keep_columns = [
                "window_id",
                "commit",
                "malicious_injected",
                "malicious_visible",
                "base_accept",
                "soft_score",
                "gray_zone_flag",
                "decision_stage",
                "decision_path",
                "reject_code",
                "qbm_shadow_available",
                "qbm_stage2_eligible",
                "qbm_stage2_veto",
                "qbm_threshold",
                "q_score_shadow",
                "q_score",
                "qbm_score_raw",
                "qbm_score_adjusted",
                "qbm_energy",
                "qbm_energy_std",
                "qbm_shot_std",
                "qbm_uncertainty_proxy",
                "qbm_invoked_by_base_accept",
                "qbm_invoked_by_gray_zone",
                "qbm_risk_hint_level",
            ]
            window_df = sim_df[[col for col in keep_columns if col in sim_df.columns]].copy()
            window_df.insert(0, "backend_mode", backend_mode)
            window_df.insert(0, "attack_id", summary.attack_id)
            window_df.insert(0, "scenario", summary.scenario)
            if "q_score_shadow" in window_df.columns:
                window_df["q_score_shadow_minus_threshold"] = (
                    pd.to_numeric(window_df["q_score_shadow"], errors="coerce") - float(threshold)
                )
            window_frames.append(window_df)

            print(
                f"scenario={summary.scenario} attack={summary.attack_id} backend={backend_mode} "
                f"eligible={int(eligible_mask.sum())} veto={int(veto_mask.sum())} "
                f"shadow_gap={summary_rows[-1]['qbm_shadow_gap_all']:.4f} "
                f"below_thr_mal={summary_rows[-1]['qbm_shadow_below_threshold_rate_malicious_all']:.4f}"
            )

    summary_df = pd.DataFrame(summary_rows)
    dist_df = pd.DataFrame(dist_rows)
    windows_df = pd.concat(window_frames, ignore_index=True) if window_frames else pd.DataFrame()

    results_dir = Path(load_config(cfg_path).get("project", {}).get("results_dir", "results")) / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "qiskit_qbm_stage2_diagnostics_summary.csv"
    dist_path = results_dir / "qiskit_qbm_stage2_diagnostics_distributions.csv"
    windows_path = results_dir / "qiskit_qbm_stage2_diagnostics_windows.csv"
    summary_df.to_csv(summary_path, index=False)
    dist_df.to_csv(dist_path, index=False)
    windows_df.to_csv(windows_path, index=False)
    print(f"saved: {summary_path}")
    print(f"saved: {dist_path}")
    print(f"saved: {windows_path}")
    return summary_df, dist_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose why QBM stage-2 vetoes do or do not fire for each backend.")
    parser.add_argument("--config", default="configs/experiments/qiskit_qbm_backend_compare.yaml", help="Path to configuration yaml.")
    parser.add_argument("--scenario", default="S3", help="Scenario to run. Default: S3")
    parser.add_argument(
        "--backend-modes",
        default="exact_state,aer_shot",
        help="Comma-separated Qiskit backend modes. Example: exact_state,aer_shot",
    )
    parser.add_argument(
        "--attack-ids",
        default="A0,A4P,A5",
        help="Comma-separated attack ids. Example: A0,A4P,A5",
    )
    parser.add_argument("--max-windows", type=int, default=180, help="Run first N windows.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--qbm-threshold", type=float, default=None, help="Optional manual threshold override.")
    parser.add_argument(
        "--force-refit-calibration",
        action="store_true",
        help="Ignore saved QBM calibration artifacts and rebuild them for this run.",
    )
    args = parser.parse_args()

    backend_modes = [item.strip().lower() for item in str(args.backend_modes).split(",") if item.strip()]
    attack_ids = [item.strip().upper() for item in str(args.attack_ids).split(",") if item.strip()]
    run_stage2_diagnostics(
        cfg_path=args.config,
        backend_modes=backend_modes,
        attack_ids=attack_ids,
        scenario=str(args.scenario).upper(),
        max_windows=args.max_windows,
        seed=args.seed,
        qbm_threshold=args.qbm_threshold,
        force_refit_calibration=bool(args.force_refit_calibration),
    )


if __name__ == "__main__":
    main()
