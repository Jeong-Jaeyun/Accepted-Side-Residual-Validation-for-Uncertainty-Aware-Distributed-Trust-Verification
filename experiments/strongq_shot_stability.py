"""
StrongQ shot-count stability sweep.

Example:
  python -m experiments.strongq_shot_stability --config configs/default.yaml --scenario S3 --attack-id A4 --max-windows 120
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from qbm.train import apply_overrides, load_config, run_simulation, save_outputs


def _parse_shots(spec: str) -> List[int]:
    values: List[int] = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(max(1, int(token)))
    if not values:
        values = [512, 1024, 2048, 4096, 8192]
    return values


def _adjust_injection_window(cfg: Dict[str, Any], max_windows: int | None) -> None:
    if max_windows is None:
        return
    attack_id = str(cfg.get("experiments", {}).get("attack_id", "A0")).upper()
    if attack_id == "A0":
        return
    iw = cfg["experiments"].setdefault("injection_window", {})
    start_w = int(iw.get("start_window", 0))
    end_w = int(iw.get("end_window", 0))
    if start_w >= int(max_windows) or end_w <= start_w:
        new_start = max(5, int(max_windows * 0.35))
        new_end = max(new_start + 1, int(max_windows * 0.75))
        iw["start_window"] = new_start
        iw["end_window"] = new_end


def run_shot_sweep(
    *,
    cfg_path: str,
    scenario: str,
    attack_id: str,
    shots_list: List[int],
    max_windows: int | None,
    seed: int | None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for shots in shots_list:
        cfg = load_config(cfg_path)
        cfg = apply_overrides(
            cfg,
            scenario=scenario,
            attack_id=attack_id,
            verifier_impl="strongq",
            seed=seed,
        )
        cfg["experiments"]["enable_injection"] = str(attack_id).upper() != "A0"
        cfg["experiments"]["attack_id"] = str(attack_id).upper()
        cfg["qbm"] = dict(cfg.get("qbm", {}))
        cfg["qbm"]["shots"] = int(shots)
        _adjust_injection_window(cfg, max_windows)

        sim_df, summary = run_simulation(cfg, max_windows=max_windows)
        paths = save_outputs(cfg, sim_df, summary)

        rows.append(
            {
                "shots": int(shots),
                **asdict(summary),
                "strongq_witness_mean": float(sim_df["strongq_witness"].mean()),
                "strongq_witness_std": float(sim_df["strongq_witness"].std(ddof=0)),
                "strongq_shot_std_mean": float(sim_df["strongq_shot_std"].mean()),
                "strongq_shot_std_max": float(sim_df["strongq_shot_std"].max()),
                "strongq_ci_width_mean": float((sim_df["strongq_ci_high"] - sim_df["strongq_ci_low"]).mean()),
                **paths,
            }
        )
        print(
            f"[shots={shots}] ASR={summary.asr:.4f} FTR={summary.ftr:.4f} "
            f"TCP={summary.tcp:.4f} shot_std_mean={rows[-1]['strongq_shot_std_mean']:.6f}"
        )

    out = pd.DataFrame(rows).sort_values("shots").reset_index(drop=True)
    base_cfg = load_config(cfg_path)
    results_dir = Path(base_cfg.get("project", {}).get("results_dir", "results"))
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "strongq_shot_stability.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved shot stability table: {out_path}")

    try:
        import matplotlib.pyplot as plt

        figures_dir.mkdir(parents=True, exist_ok=True)
        fig_path = figures_dir / "strongq_shot_stability.png"
        plt.figure(figsize=(7.2, 4.0))
        plt.plot(out["shots"], out["strongq_shot_std_mean"], marker="o", label="mean shot std")
        plt.plot(out["shots"], out["strongq_witness_std"], marker="o", label="witness std")
        plt.xscale("log", base=2)
        plt.xlabel("Shots (log2)")
        plt.ylabel("Dispersion")
        plt.title("StrongQ Stability vs Measurement Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"Saved shot stability figure: {fig_path}")
    except Exception as exc:  # pragma: no cover
        print(f"Skip shot-stability figure (matplotlib unavailable): {exc}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StrongQ shot-count stability sweep.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to configuration yaml.")
    parser.add_argument("--scenario", default="S3", help="Scenario label (default: S3).")
    parser.add_argument("--attack-id", default="A4", help="Attack id (default: A4).")
    parser.add_argument(
        "--shots",
        default="512,1024,2048,4096,8192",
        help="Comma-separated shot counts (default: 512,1024,2048,4096,8192).",
    )
    parser.add_argument("--max-windows", type=int, default=None, help="Run only first N windows.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    args = parser.parse_args()

    run_shot_sweep(
        cfg_path=args.config,
        scenario=str(args.scenario).upper(),
        attack_id=str(args.attack_id).upper(),
        shots_list=_parse_shots(args.shots),
        max_windows=args.max_windows,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
