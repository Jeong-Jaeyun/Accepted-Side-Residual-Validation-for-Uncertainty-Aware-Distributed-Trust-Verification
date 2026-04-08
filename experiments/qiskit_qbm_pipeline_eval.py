from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from qbm.train import apply_overrides, load_config, run_simulation, save_outputs


def _numeric_mean(df: pd.DataFrame, column: str) -> float:
    series = pd.to_numeric(df.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
    if series.empty:
        return float("nan")
    return float(series.mean())


def _numeric_last(df: pd.DataFrame, column: str) -> float:
    series = pd.to_numeric(df.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
    if series.empty:
        return float("nan")
    return float(series.iloc[-1])


def _bool_mask(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df.get(column, pd.Series(dtype=float)), errors="coerce").fillna(0.0) > 0.0


def _prepare_cfg(
    cfg_path: str,
    *,
    pipeline_mode: str,
    backend_mode: str,
    attack_id: str,
    scenario: str,
    seed: int | None,
    max_windows: int | None,
) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    cfg = apply_overrides(cfg, scenario=scenario, attack_id=attack_id, verifier_impl="s3_mev", seed=seed)
    cfg.setdefault("verification", {})
    cfg["verification"]["pipeline_mode"] = str(pipeline_mode).strip().lower()
    cfg.setdefault("qbm", {})
    cfg["qbm"]["qiskit_backend_mode"] = str(backend_mode).strip().lower()
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


def run_pipeline_eval(
    cfg_path: str,
    *,
    pipeline_modes: List[str],
    attack_ids: List[str],
    scenario: str,
    backend_mode: str,
    max_windows: int | None,
    seed: int | None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for pipeline_mode in pipeline_modes:
        for attack_id in attack_ids:
            cfg = _prepare_cfg(
                cfg_path,
                pipeline_mode=pipeline_mode,
                backend_mode=backend_mode,
                attack_id=attack_id,
                scenario=scenario,
                seed=seed,
                max_windows=max_windows,
            )
            sim_df, summary = run_simulation(cfg, max_windows=max_windows)
            paths = save_outputs(cfg, sim_df, summary)
            veto_mask = _bool_mask(sim_df, "qbm_stage2_veto")
            malicious_mask = _bool_mask(sim_df, "malicious_visible")
            benign_mask = ~malicious_mask
            rows.append(
                {
                    "scenario": summary.scenario,
                    "attack_id": summary.attack_id,
                    "pipeline_mode": pipeline_mode,
                    "backend_mode": backend_mode,
                    "verifier_name": summary.verifier_name,
                    "gray_rate": _numeric_mean(sim_df, "gray_zone_flag"),
                    "strongq_called_rate": _numeric_mean(sim_df, "strongq_called"),
                    "qbm_shadow_mean": _numeric_mean(sim_df, "q_score_shadow"),
                    "qbm_stage2_mean": _numeric_mean(sim_df, "q_score"),
                    "qbm_veto_rate": _numeric_mean(sim_df, "qbm_stage2_veto"),
                    "qbm_veto_count": int(veto_mask.sum()),
                    "qbm_benign_veto_count": int((veto_mask & benign_mask).sum()),
                    "qbm_malicious_veto_count": int((veto_mask & malicious_mask).sum()),
                    "qbm_threshold_final": _numeric_last(sim_df, "qbm_threshold"),
                    "asr": float(summary.asr),
                    "ftr": float(summary.ftr),
                    "tcp": float(summary.tcp),
                    "ttd_windows": float(summary.ttd_windows),
                    "processed_tps_mean": float(summary.processed_tps_mean),
                    "latency_ms_mean": float(summary.latency_ms_mean),
                    **paths,
                }
            )
            print(
                f"pipeline={pipeline_mode} attack={summary.attack_id} backend={backend_mode} "
                f"ASR={summary.asr:.4f} FTR={summary.ftr:.4f} TCP={summary.tcp:.4f}"
            )

    out = pd.DataFrame(rows)
    out_path = Path(load_config(cfg_path).get("project", {}).get("results_dir", "results")) / "tables" / "qiskit_qbm_pipeline_eval.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved: {out_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate explicit S3/QBM/StrongQ pipeline modes.")
    parser.add_argument("--config", default="configs/experiments/qiskit_qbm_tuned.yaml", help="Path to configuration yaml.")
    parser.add_argument("--scenario", default="S3", help="Scenario to run. Default: S3")
    parser.add_argument(
        "--pipeline-modes",
        default="s3_only,s3_qbm,s3_strongq,s3_qbm_strongq",
        help="Comma-separated pipeline modes.",
    )
    parser.add_argument("--backend-mode", default="exact_state", help="Qiskit backend mode.")
    parser.add_argument("--attack-ids", default="A0,A4P,A5", help="Comma-separated attack ids.")
    parser.add_argument("--max-windows", type=int, default=180, help="Run first N windows.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    args = parser.parse_args()

    pipeline_modes = [item.strip().lower() for item in str(args.pipeline_modes).split(",") if item.strip()]
    attack_ids = [item.strip().upper() for item in str(args.attack_ids).split(",") if item.strip()]
    run_pipeline_eval(
        cfg_path=args.config,
        pipeline_modes=pipeline_modes,
        attack_ids=attack_ids,
        scenario=str(args.scenario).upper(),
        backend_mode=str(args.backend_mode).strip().lower(),
        max_windows=args.max_windows,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
