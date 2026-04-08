from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Any, Dict, List

import pandas as pd

from qbm.train import apply_overrides, load_config, run_simulation, save_outputs


ATTACK_ORDER = {"A0": 0, "A4P": 1, "A5": 2}


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


def _sweep_suffix(sweep_mode: str, sweep_value: float) -> str:
    value = str(f"{float(sweep_value):.4f}").replace(".", "p")
    return f"{str(sweep_mode).strip().lower()}_{value}"


def _copy_sweep_artifacts(paths: Dict[str, str], sweep_mode: str, sweep_value: float) -> Dict[str, str]:
    copied = dict(paths)
    suffix = _sweep_suffix(sweep_mode, sweep_value)
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
    scenario: str,
    attack_id: str,
    pipeline_mode: str,
    backend_mode: str,
    threshold: float | None,
    quantile: float | None,
    seed: int | None,
    max_windows: int | None,
) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    cfg = apply_overrides(cfg, scenario=scenario, attack_id=attack_id, verifier_impl="s3_mev", seed=seed)
    cfg.setdefault("verification", {})
    cfg["verification"]["pipeline_mode"] = str(pipeline_mode).strip().lower()
    cfg["verification"]["qbm_use_saved_calibration"] = False
    cfg["verification"]["qbm_save_calibration"] = False
    if quantile is not None:
        cfg["verification"]["qbm_auto_threshold_from_a0"] = True
        cfg["verification"]["qbm_threshold_quantile"] = float(quantile)
    if threshold is not None:
        cfg["verification"]["qbm_threshold"] = float(threshold)
        cfg["verification"]["qbm_auto_threshold_from_a0"] = False

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


def _tradeoff_row(
    *,
    sim_df: pd.DataFrame,
    summary: Any,
    scenario: str,
    backend_mode: str,
    pipeline_mode: str,
    sweep_mode: str,
    sweep_value: float,
    paths: Dict[str, str],
) -> Dict[str, Any]:
    veto_mask = _bool_mask(sim_df, "qbm_stage2_veto")
    malicious_mask = _bool_mask(sim_df, "malicious_visible")
    benign_mask = ~malicious_mask
    eligible_mask = _bool_mask(sim_df, "qbm_stage2_eligible")

    benign_veto_count = int((veto_mask & benign_mask).sum())
    malicious_veto_count = int((veto_mask & malicious_mask).sum())

    return {
        "scenario": scenario,
        "attack_id": summary.attack_id,
        "pipeline_mode": pipeline_mode,
        "backend_mode": backend_mode,
        "sweep_mode": sweep_mode,
        "sweep_value": float(sweep_value),
        "qbm_threshold_final": _numeric_last(sim_df, "qbm_threshold"),
        "qbm_threshold_quantile_config": float(sweep_value) if sweep_mode == "quantile" else float("nan"),
        "qbm_stage2_eligible_windows": int(eligible_mask.sum()),
        "qbm_stage2_veto_windows": int(veto_mask.sum()),
        "benign_veto_count": benign_veto_count,
        "malicious_veto_count": malicious_veto_count,
        "qbm_veto_rate": _numeric_mean(sim_df, "qbm_stage2_veto"),
        "qbm_veto_given_eligible_rate": float(veto_mask[eligible_mask].mean()) if eligible_mask.any() else float("nan"),
        "ftr": float(summary.ftr),
        "asr": float(summary.asr),
        "tcp": float(summary.tcp),
        "processed_tps_mean": float(summary.processed_tps_mean),
        "latency_ms_mean": float(summary.latency_ms_mean),
        **paths,
    }


def _build_tradeoff_table(out: pd.DataFrame) -> pd.DataFrame:
    if out.empty:
        return out.copy()
    tradeoff = out[
        [
            "scenario",
            "pipeline_mode",
            "backend_mode",
            "sweep_mode",
            "sweep_value",
            "attack_id",
            "qbm_threshold_final",
            "ftr",
            "tcp",
            "qbm_veto_rate",
            "benign_veto_count",
            "malicious_veto_count",
            "qbm_stage2_veto_windows",
            "qbm_stage2_eligible_windows",
        ]
    ].copy()
    tradeoff["_attack_order"] = tradeoff["attack_id"].astype(str).map(ATTACK_ORDER).fillna(99)
    tradeoff = tradeoff.sort_values(["sweep_mode", "sweep_value", "_attack_order"], kind="mergesort")
    return tradeoff.drop(columns=["_attack_order"]).reset_index(drop=True)


def run_threshold_sweep(
    cfg_path: str,
    *,
    scenario: str,
    attack_ids: List[str],
    pipeline_mode: str,
    backend_mode: str,
    thresholds: List[float],
    quantiles: List[float],
    max_windows: int | None,
    seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []

    sweep_plan: List[tuple[str, float]] = []
    sweep_plan.extend(("quantile", float(value)) for value in quantiles)
    sweep_plan.extend(("threshold", float(value)) for value in thresholds)
    if not sweep_plan:
        raise ValueError("Provide at least one quantile or threshold value to sweep.")

    for sweep_mode, sweep_value in sweep_plan:
        threshold = sweep_value if sweep_mode == "threshold" else None
        quantile = sweep_value if sweep_mode == "quantile" else None
        for attack_id in attack_ids:
            cfg = _prepare_cfg(
                cfg_path,
                scenario=scenario,
                attack_id=attack_id,
                pipeline_mode=pipeline_mode,
                backend_mode=backend_mode,
                threshold=threshold,
                quantile=quantile,
                seed=seed,
                max_windows=max_windows,
            )
            sim_df, summary = run_simulation(cfg, max_windows=max_windows)
            paths = _copy_sweep_artifacts(save_outputs(cfg, sim_df, summary), sweep_mode, sweep_value)
            row = _tradeoff_row(
                sim_df=sim_df,
                summary=summary,
                scenario=scenario,
                backend_mode=backend_mode,
                pipeline_mode=pipeline_mode,
                sweep_mode=sweep_mode,
                sweep_value=sweep_value,
                paths=paths,
            )
            rows.append(row)
            print(
                f"{sweep_mode}={sweep_value:.4f} attack={summary.attack_id} backend={backend_mode} "
                f"FTR={summary.ftr:.4f} TCP={summary.tcp:.4f} "
                f"veto={row['qbm_stage2_veto_windows']} benign_veto={row['benign_veto_count']} "
                f"mal_veto={row['malicious_veto_count']}"
            )

    out = pd.DataFrame(rows)
    tradeoff = _build_tradeoff_table(out)
    results_dir = Path(load_config(cfg_path).get("project", {}).get("results_dir", "results")) / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "qiskit_qbm_threshold_sweep.csv"
    tradeoff_path = results_dir / "qiskit_qbm_exact_tradeoff_table.csv"
    out.to_csv(out_path, index=False)
    tradeoff.to_csv(tradeoff_path, index=False)
    print(f"saved: {out_path}")
    print(f"saved: {tradeoff_path}")
    return out, tradeoff


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the exact-state QBM operating point inside the main S3+QBM+StrongQ pipeline.")
    parser.add_argument("--config", default="configs/experiments/qiskit_qbm_main.yaml", help="Path to configuration yaml.")
    parser.add_argument("--scenario", default="S3", help="Scenario to run. Default: S3")
    parser.add_argument("--pipeline-mode", default="s3_qbm_strongq", help="Pipeline mode. Default: s3_qbm_strongq")
    parser.add_argument("--backend-mode", default="exact_state", help="Qiskit backend mode. Default: exact_state")
    parser.add_argument("--attack-ids", default="A0,A4P,A5", help="Comma-separated attack ids.")
    parser.add_argument(
        "--quantiles",
        default="0.01,0.02,0.03,0.05,0.07",
        help="Comma-separated qbm threshold quantiles for auto-calibration.",
    )
    parser.add_argument(
        "--thresholds",
        default="",
        help="Optional comma-separated manual qbm thresholds. Leave empty to sweep quantiles only.",
    )
    parser.add_argument("--max-windows", type=int, default=180, help="Run first N windows.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    args = parser.parse_args()

    attack_ids = [item.strip().upper() for item in str(args.attack_ids).split(",") if item.strip()]
    quantiles = [float(item.strip()) for item in str(args.quantiles).split(",") if item.strip()]
    thresholds = [float(item.strip()) for item in str(args.thresholds).split(",") if item.strip()]

    run_threshold_sweep(
        cfg_path=args.config,
        scenario=str(args.scenario).upper(),
        attack_ids=attack_ids,
        pipeline_mode=str(args.pipeline_mode).strip().lower(),
        backend_mode=str(args.backend_mode).strip().lower(),
        thresholds=thresholds,
        quantiles=quantiles,
        max_windows=args.max_windows,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
