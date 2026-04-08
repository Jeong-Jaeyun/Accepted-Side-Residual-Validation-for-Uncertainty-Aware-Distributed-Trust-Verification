from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any, Dict, List

import pandas as pd

from qbm.train import apply_overrides, load_config, run_simulation, save_outputs


def _numeric_mean(df: pd.DataFrame, column: str) -> float:
    series = pd.to_numeric(df.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
    if series.empty:
        return float("nan")
    return float(series.mean())


def _numeric_std(df: pd.DataFrame, column: str) -> float:
    series = pd.to_numeric(df.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
    if series.empty:
        return float("nan")
    return float(series.std(ddof=0))


def _numeric_last(df: pd.DataFrame, column: str) -> float:
    series = pd.to_numeric(df.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
    if series.empty:
        return float("nan")
    return float(series.iloc[-1])


def _mask_malicious(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df.get("malicious_injected", pd.Series(dtype=float)), errors="coerce").fillna(0.0) > 0


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


def _prepare_cfg(
    cfg_path: str,
    *,
    backend_mode: str,
    attack_id: str,
    scenario: str,
    seed: int | None,
    qbm_threshold: float | None,
    max_windows: int | None,
) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    cfg = apply_overrides(cfg, scenario=scenario, attack_id=attack_id, verifier_impl="qbm", seed=seed)
    cfg.setdefault("qbm", {})
    cfg["qbm"]["qiskit_backend_mode"] = backend_mode
    if qbm_threshold is not None:
        cfg.setdefault("verification", {})
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


def run_feasibility(
    cfg_path: str,
    *,
    backend_modes: List[str],
    attack_ids: List[str],
    scenario: str,
    max_windows: int | None,
    seed: int | None,
    qbm_threshold: float | None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
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
            )
            started_at = perf_counter()
            sim_df, summary = run_simulation(cfg, max_windows=max_windows)
            elapsed_s = perf_counter() - started_at
            paths = _copy_backend_artifacts(save_outputs(cfg, sim_df, summary), backend_mode)

            row = {
                "scenario": summary.scenario,
                "attack_id": summary.attack_id,
                "qiskit_backend_mode": backend_mode,
                "verifier_name": summary.verifier_name,
                "elapsed_s": float(elapsed_s),
                "qbm_shadow_eval_windows": int(pd.to_numeric(sim_df.get("q_score_shadow", pd.Series(dtype=float)), errors="coerce").notna().sum()),
                "qbm_stage2_eval_windows": int(pd.to_numeric(sim_df.get("q_score", pd.Series(dtype=float)), errors="coerce").notna().sum()),
                "malicious_windows": int(_mask_malicious(sim_df).sum()),
                "q_score_shadow_mean": _numeric_mean(sim_df, "q_score_shadow"),
                "q_score_shadow_std": _numeric_std(sim_df, "q_score_shadow"),
                "q_score_shadow_mean_benign": _numeric_mean(sim_df.loc[~_mask_malicious(sim_df)], "q_score_shadow"),
                "q_score_shadow_mean_malicious": _numeric_mean(sim_df.loc[_mask_malicious(sim_df)], "q_score_shadow"),
                "q_score_stage2_mean": _numeric_mean(sim_df, "q_score"),
                "q_score_stage2_std": _numeric_std(sim_df, "q_score"),
                "q_score_stage2_mean_benign": _numeric_mean(sim_df.loc[~_mask_malicious(sim_df)], "q_score"),
                "q_score_stage2_mean_malicious": _numeric_mean(sim_df.loc[_mask_malicious(sim_df)], "q_score"),
                "qbm_energy_mean": _numeric_mean(sim_df, "qbm_energy"),
                "qbm_energy_mean_benign": _numeric_mean(sim_df.loc[~_mask_malicious(sim_df)], "qbm_energy"),
                "qbm_energy_mean_malicious": _numeric_mean(sim_df.loc[_mask_malicious(sim_df)], "qbm_energy"),
                "qbm_shot_std_mean": _numeric_mean(sim_df, "qbm_shot_std"),
                "qbm_depth_mean": _numeric_mean(sim_df, "qbm_circuit_depth"),
                "qbm_qubits_mean": _numeric_mean(sim_df, "qbm_n_qubits"),
                "qbm_threshold_final": _numeric_last(sim_df, "qbm_threshold"),
                "qbm_stage2_veto_rate": _numeric_mean(sim_df, "qbm_stage2_veto"),
                "processed_tps_mean": float(summary.processed_tps_mean),
                "latency_ms_mean": float(summary.latency_ms_mean),
                "asr": float(summary.asr),
                "ftr": float(summary.ftr),
                "tcp": float(summary.tcp),
                "ttd_windows": float(summary.ttd_windows),
                **paths,
            }
            rows.append(row)
            print(
                f"scenario={summary.scenario} attack={summary.attack_id} backend_mode={backend_mode} "
                f"shadow_q={row['q_score_shadow_mean']:.4f} stage2_q={row['q_score_stage2_mean']:.4f} "
                f"ASR={summary.asr:.4f} FTR={summary.ftr:.4f}"
            )

    out = pd.DataFrame(rows)
    results_dir = Path(load_config(cfg_path).get("project", {}).get("results_dir", "results"))
    out_path = results_dir / "tables" / "qiskit_qbm_feasibility.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved: {out_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Qiskit backend modes for the shallow forensic QBM.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to configuration yaml.")
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
    args = parser.parse_args()

    backend_modes = [item.strip().lower() for item in str(args.backend_modes).split(",") if item.strip()]
    attack_ids = [item.strip().upper() for item in str(args.attack_ids).split(",") if item.strip()]
    run_feasibility(
        cfg_path=args.config,
        backend_modes=backend_modes,
        attack_ids=attack_ids,
        scenario=str(args.scenario).upper(),
        max_windows=args.max_windows,
        seed=args.seed,
        qbm_threshold=args.qbm_threshold,
    )


if __name__ == "__main__":
    main()
