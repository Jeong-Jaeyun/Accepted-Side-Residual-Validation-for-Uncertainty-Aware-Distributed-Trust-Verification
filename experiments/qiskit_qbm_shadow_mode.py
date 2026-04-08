from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from qbm.train import apply_overrides, load_config, run_simulation


def _prepare_cfg(
    cfg_path: str,
    *,
    attack_id: str,
    scenario: str,
    seed: int | None,
    max_windows: int | None,
) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    cfg = apply_overrides(cfg, scenario=scenario, attack_id=attack_id, verifier_impl="qbm", seed=seed)
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


def run_shadow_mode(
    cfg_path: str,
    *,
    attack_ids: List[str],
    scenario: str,
    max_windows: int | None,
    seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []
    for attack_id in attack_ids:
        cfg = _prepare_cfg(cfg_path, attack_id=attack_id, scenario=scenario, seed=seed, max_windows=max_windows)
        sim_df, summary = run_simulation(cfg, max_windows=max_windows)
        shadow_df = sim_df[
            [
                "window_id",
                "attack_id",
                "commit",
                "malicious_visible",
                "q_score_shadow",
                "qbm_energy",
                "qbm_threshold",
                "qbm_stage2_eligible",
                "qbm_stage2_veto",
                "soft_score",
                "gray_zone_flag",
            ]
        ].copy()
        shadow_df["scenario"] = summary.scenario
        window_rows.append(shadow_df)

        summary_rows.append(
            {
                "scenario": summary.scenario,
                "attack_id": summary.attack_id,
                "shadow_windows": int(pd.to_numeric(shadow_df["q_score_shadow"], errors="coerce").notna().sum()),
                "q_score_shadow_mean": float(pd.to_numeric(shadow_df["q_score_shadow"], errors="coerce").dropna().mean()),
                "q_score_shadow_mean_benign": float(
                    pd.to_numeric(shadow_df.loc[pd.to_numeric(shadow_df["malicious_visible"], errors="coerce").fillna(0.0) <= 0, "q_score_shadow"], errors="coerce").dropna().mean()
                ),
                "q_score_shadow_mean_malicious": float(
                    pd.to_numeric(shadow_df.loc[pd.to_numeric(shadow_df["malicious_visible"], errors="coerce").fillna(0.0) > 0, "q_score_shadow"], errors="coerce").dropna().mean()
                ),
                "qbm_energy_mean": float(pd.to_numeric(shadow_df["qbm_energy"], errors="coerce").dropna().mean()),
                "qbm_stage2_veto_rate": float(pd.to_numeric(shadow_df["qbm_stage2_veto"], errors="coerce").fillna(0.0).mean()),
                "ftr": float(summary.ftr),
                "asr": float(summary.asr),
            }
        )

    windows = pd.concat(window_rows, ignore_index=True) if window_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    results_dir = Path(load_config(cfg_path).get("project", {}).get("results_dir", "results")) / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    windows_path = results_dir / "qiskit_qbm_shadow_windows.csv"
    summary_path = results_dir / "qiskit_qbm_shadow_summary.csv"
    windows.to_csv(windows_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"saved: {windows_path}")
    print(f"saved: {summary_path}")
    return windows, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all-window QBM shadow-mode evaluation without using MEV gating as the metric.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to configuration yaml.")
    parser.add_argument("--scenario", default="S3", help="Scenario to run. Default: S3")
    parser.add_argument("--attack-ids", default="A0,A4P,A5", help="Comma-separated attack ids.")
    parser.add_argument("--max-windows", type=int, default=180, help="Run first N windows.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    args = parser.parse_args()

    attack_ids = [item.strip().upper() for item in str(args.attack_ids).split(",") if item.strip()]
    run_shadow_mode(
        cfg_path=args.config,
        attack_ids=attack_ids,
        scenario=str(args.scenario).upper(),
        max_windows=args.max_windows,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
