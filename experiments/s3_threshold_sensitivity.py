"""
S3 threshold sensitivity sweep with exactly 3 profile points (no full grid).

Profiles are built around config defaults:
  - loose   : epsilon-0.05, theta-0.10
  - default : epsilon,      theta
  - strict  : epsilon+0.05, theta+0.10

Outputs:
  - results/tables/s3_threshold_sensitivity.csv
  - results/figures/s3_threshold_sensitivity.png
  - results/figures/s3_threshold_sensitivity_asr.png
  - results/figures/s3_threshold_sensitivity_ftr.png
  - results/figures/s3_threshold_sensitivity_dropped_by_verification.png
  - results/tables/meta_s3_threshold_<profile>.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from qbm.train import apply_overrides, load_config, run_simulation


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _build_profiles(base_epsilon: float, base_theta: float) -> List[Dict[str, float]]:
    return [
        {
            "profile": "loose",
            "epsilon": _clamp01(base_epsilon - 0.05),
            "theta": _clamp01(base_theta - 0.10),
            "profile_order": 0,
        },
        {
            "profile": "default",
            "epsilon": _clamp01(base_epsilon),
            "theta": _clamp01(base_theta),
            "profile_order": 1,
        },
        {
            "profile": "strict",
            "epsilon": _clamp01(base_epsilon + 0.05),
            "theta": _clamp01(base_theta + 0.10),
            "profile_order": 2,
        },
    ]


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


def _save_metric_bar(
    df: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels = [f"{r.profile}\n(eps={r.epsilon:.2f}, th={r.theta:.2f})" for r in df.itertuples(index=False)]
    plt.figure(figsize=(6.8, 4.0))
    plt.bar(labels, df[metric])
    if metric in {"asr", "ftr"}:
        plt.ylim(0.0, 1.0)
    plt.xlabel("Threshold Profile")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved figure: {out_path}")


def run_sensitivity(
    *,
    cfg_path: str,
    attack_id: str,
    max_windows: int | None,
    seed: int | None,
) -> pd.DataFrame:
    base_cfg = load_config(cfg_path)
    base_ver = base_cfg.get("verification", {})
    base_eps = float(base_ver.get("corr_threshold", 0.78))
    base_theta = float(base_ver.get("explanation_threshold", 0.50))
    profiles = _build_profiles(base_eps, base_theta)

    results_dir = Path(base_cfg.get("project", {}).get("results_dir", "results"))
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for profile in profiles:
        cfg = load_config(cfg_path)
        cfg = apply_overrides(
            cfg,
            scenario="S3",
            attack_id=attack_id,
            verifier_impl="s3_mev",
            seed=seed,
        )
        cfg["experiments"]["enable_injection"] = str(attack_id).upper() != "A0"
        cfg["experiments"]["attack_id"] = str(attack_id).upper()
        cfg["verification"]["corr_threshold"] = float(profile["epsilon"])
        cfg["verification"]["explanation_threshold"] = float(profile["theta"])
        _adjust_injection_window(cfg, max_windows)

        sim_df, summary = run_simulation(cfg, max_windows=max_windows)
        node_count = int(cfg.get("blockchain_net", {}).get("validators", 0))
        row = {
            "profile": str(profile["profile"]),
            "profile_order": int(profile["profile_order"]),
            "scenario": summary.scenario,
            "attack_id": summary.attack_id,
            "verifier_name": summary.verifier_name,
            "epsilon": float(profile["epsilon"]),
            "theta": float(profile["theta"]),
            "node_count": node_count,
            **asdict(summary),
        }
        rows.append(row)

        meta_path = tables_dir / f"meta_s3_threshold_{profile['profile']}.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "profile": profile["profile"],
                    "scenario": summary.scenario,
                    "attack_id": summary.attack_id,
                    "verifier": summary.verifier_name,
                    "reproducibility": {
                        "epsilon_corr_threshold": float(profile["epsilon"]),
                        "theta_explanation_threshold": float(profile["theta"]),
                        "node_count": node_count,
                    },
                    "summary": asdict(summary),
                    "max_windows": max_windows,
                    "seed": seed,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"[{profile['profile']}] eps={profile['epsilon']:.2f} theta={profile['theta']:.2f} "
            f"ASR={summary.asr:.4f} FTR={summary.ftr:.4f} drop_v={summary.dropped_by_verification_sum:.2f}"
        )

    out = pd.DataFrame(rows).sort_values("profile_order").reset_index(drop=True)
    out_path = tables_dir / "s3_threshold_sensitivity.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved sensitivity CSV: {out_path}")

    # Requested one figure containing ASR/FTR/dropped_by_verification.
    try:
        import matplotlib.pyplot as plt

        labels = [f"{r.profile}\n(eps={r.epsilon:.2f}, th={r.theta:.2f})" for r in out.itertuples(index=False)]
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
        axes[0].bar(labels, out["asr"])
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_title("ASR")
        axes[0].set_xlabel("Threshold Profile")

        axes[1].bar(labels, out["ftr"])
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("FTR")
        axes[1].set_xlabel("Threshold Profile")

        axes[2].bar(labels, out["dropped_by_verification_sum"])
        axes[2].set_title("Dropped by Verification")
        axes[2].set_xlabel("Threshold Profile")

        fig.suptitle(f"S3 Threshold Sensitivity ({attack_id.upper()})")
        fig.tight_layout()
        fig_path = figures_dir / "s3_threshold_sensitivity.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"Saved figure: {fig_path}")

        # Additional per-metric figures for paper workflow.
        _save_metric_bar(
            out,
            metric="asr",
            ylabel="ASR",
            title=f"S3 Threshold Sensitivity ASR ({attack_id.upper()})",
            out_path=figures_dir / "s3_threshold_sensitivity_asr.png",
        )
        _save_metric_bar(
            out,
            metric="ftr",
            ylabel="FTR",
            title=f"S3 Threshold Sensitivity FTR ({attack_id.upper()})",
            out_path=figures_dir / "s3_threshold_sensitivity_ftr.png",
        )
        _save_metric_bar(
            out,
            metric="dropped_by_verification_sum",
            ylabel="Dropped by Verification",
            title=f"S3 Threshold Sensitivity Dropped-by-Verification ({attack_id.upper()})",
            out_path=figures_dir / "s3_threshold_sensitivity_dropped_by_verification.png",
        )
    except Exception as exc:  # pragma: no cover
        print(f"Skip sensitivity figures (matplotlib unavailable): {exc}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3-point S3 threshold sensitivity sweep.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to configuration yaml.")
    parser.add_argument("--attack-id", default="A5", help="Attack id for sensitivity (default: A5).")
    parser.add_argument("--max-windows", type=int, default=None, help="Run only first N windows.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    args = parser.parse_args()

    run_sensitivity(
        cfg_path=args.config,
        attack_id=str(args.attack_id).upper(),
        max_windows=args.max_windows,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
