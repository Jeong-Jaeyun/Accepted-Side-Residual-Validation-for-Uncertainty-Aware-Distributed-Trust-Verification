"""
Generate compact tau-sensitivity and soft-score distribution reports.

Outputs:
  - results/tables/tau_sensitivity_<attack>.csv
  - results/figures/tau_sensitivity_<attack>_asr_vs_latency.png
  - results/tables/soft_score_distribution_by_attack.csv
  - results/figures/soft_score_distribution_a0_a5.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from qbm.train import apply_overrides, load_config, run_simulation


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


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


def _run_once(
    *,
    cfg_path: str,
    scenario: str,
    attack_id: str,
    verifier_impl: str,
    seed: int | None,
    max_windows: int | None,
    verification_patch: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, Any]:
    cfg = load_config(cfg_path)
    cfg = apply_overrides(
        cfg,
        scenario=scenario,
        attack_id=attack_id,
        verifier_impl=verifier_impl,
        seed=seed,
    )
    cfg["experiments"]["enable_injection"] = str(attack_id).upper() != "A0"
    cfg["experiments"]["attack_id"] = str(attack_id).upper()
    if verification_patch:
        cfg["verification"].update(dict(verification_patch))
    _adjust_injection_window(cfg, max_windows)
    sim_df, summary = run_simulation(cfg, max_windows=max_windows)
    return sim_df, summary


def _resolve_base_tau(cfg_path: str, seed: int | None, max_windows: int | None) -> float:
    sim_df, _summary = _run_once(
        cfg_path=cfg_path,
        scenario="S3",
        attack_id="A0",
        verifier_impl="s3_mev",
        seed=seed,
        max_windows=max_windows,
        verification_patch=None,
    )
    runtime = dict(sim_df.attrs.get("verifier_runtime", {}))
    tau = runtime.get("tau_precalibrated_value")
    if tau is None:
        tau = runtime.get("tau_final")
    if tau is None or not np.isfinite(float(tau)):
        series = pd.to_numeric(sim_df.get("tau", pd.Series([], dtype=float)), errors="coerce").dropna()
        if len(series) > 0:
            tau = float(series.iloc[-1])
    if tau is None or not np.isfinite(float(tau)):
        tau = 0.72
    return _clamp01(float(tau))


def _plot_tau_sensitivity(df: pd.DataFrame, out_path: Path, tau_base: float, attack_id: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for verifier, color in [("s3_mev", "#d95f02"), ("strongq_verifier", "#1b9e77")]:
        sub = df[df["verifier"] == verifier].sort_values("tau")
        label = "S3-MEV" if verifier == "s3_mev" else "StrongQ"
        ax.plot(
            sub["latency_ms_mean"],
            sub["asr"],
            marker="o",
            linewidth=1.8,
            color=color,
            label=label,
        )
        for r in sub.itertuples(index=False):
            ax.annotate(f"{r.tau:.3f}", (r.latency_ms_mean, r.asr), fontsize=8, xytext=(4, 3), textcoords="offset points")

    ax.set_xlabel("Latency Mean (ms)")
    ax.set_ylabel("ASR")
    ax.set_ylim(0.0, min(1.0, max(0.05, float(df["asr"].max()) * 1.20)))
    ax.grid(alpha=0.25)
    ax.set_title(f"{attack_id} Tau Sensitivity (tau0={tau_base:.3f}, range=tau0±0.03)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_soft_score_distribution(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    attacks = ["A0", "A1", "A2", "A3", "A4", "A5"]
    colors = {
        "A0": "#1f77b4",
        "A1": "#ff7f0e",
        "A2": "#2ca02c",
        "A3": "#d62728",
        "A4": "#9467bd",
        "A5": "#8c564b",
    }

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for attack in attacks:
        vals = pd.to_numeric(df.loc[df["attack_id"] == attack, "soft_score"], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        ax.hist(
            vals.to_numpy(),
            bins=64,
            density=True,
            histtype="step",
            linewidth=1.7,
            color=colors[attack],
            label=attack,
        )

    ax.set_xlabel("soft_score")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.set_title("soft_score Distribution by Attack (S3-MEV)")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_report(
    *,
    cfg_path: str,
    max_windows: int | None,
    seed: int | None,
    tau_attack_id: str,
) -> None:
    base_cfg = load_config(cfg_path)
    results_dir = Path(base_cfg.get("project", {}).get("results_dir", "results"))
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tau_base = _resolve_base_tau(cfg_path, seed, max_windows)
    tau_offsets = [-0.03, -0.015, 0.0, 0.015, 0.03]
    tau_rows: List[Dict[str, Any]] = []

    target_attack = str(tau_attack_id).upper()
    for verifier in ("s3_mev", "strongq"):
        for offset in tau_offsets:
            tau = _clamp01(tau_base + float(offset))
            sim_df, summary = _run_once(
                cfg_path=cfg_path,
                scenario="S3",
                attack_id=target_attack,
                verifier_impl=verifier,
                seed=seed,
                max_windows=max_windows,
                verification_patch={
                    "s3_auto_tau_from_a0": False,
                    "s3_soft_tau": tau,
                },
            )
            runtime = dict(sim_df.attrs.get("verifier_runtime", {}))
            gray_rate = float(
                (pd.to_numeric(sim_df.get("gray_zone_flag", pd.Series([], dtype=float)), errors="coerce").fillna(0.0) > 0).mean()
            ) if len(sim_df) > 0 else float("nan")
            strongq_called_rate = float(
                (pd.to_numeric(sim_df.get("strongq_called", pd.Series([], dtype=float)), errors="coerce").fillna(0.0) > 0).mean()
            ) if len(sim_df) > 0 else float("nan")
            n_flip_reject = int(
                (pd.to_numeric(sim_df.get("flip_mev_to_reject", pd.Series([], dtype=float)), errors="coerce").fillna(0.0) > 0).sum()
            )
            tau_rows.append(
                {
                    "attack_id": target_attack,
                    "verifier": summary.verifier_name,
                    "tau_base": tau_base,
                    "tau_offset": float(offset),
                    "tau": float(tau),
                    "asr": float(summary.asr),
                    "ftr": float(summary.ftr),
                    "latency_ms_mean": float(summary.latency_ms_mean),
                    "processed_tps_mean": float(summary.processed_tps_mean),
                    "gray_rate": gray_rate,
                    "strongq_called_rate": strongq_called_rate,
                    "n_flip_mev_to_reject": n_flip_reject,
                    "tau_runtime_reported": float(runtime.get("tau_final", tau)),
                }
            )
            print(
                f"[tau={tau:.4f}] verifier={summary.verifier_name} "
                f"ASR={summary.asr:.4f} latency={summary.latency_ms_mean:.1f}"
            )

    tau_df = pd.DataFrame(tau_rows)
    tau_df = tau_df.sort_values(["verifier", "tau"]).reset_index(drop=True)
    tau_csv = tables_dir / f"tau_sensitivity_{target_attack.lower()}.csv"
    tau_df.to_csv(tau_csv, index=False)
    _plot_tau_sensitivity(
        tau_df.rename(columns={"verifier": "verifier"}),
        out_path=figures_dir / f"tau_sensitivity_{target_attack.lower()}_asr_vs_latency.png",
        tau_base=tau_base,
        attack_id=target_attack,
    )
    print(f"Saved tau sensitivity CSV: {tau_csv}")
    print(f"Saved tau sensitivity figure: {figures_dir / f'tau_sensitivity_{target_attack.lower()}_asr_vs_latency.png'}")

    soft_rows: List[pd.DataFrame] = []
    for attack_id in ("A0", "A1", "A2", "A3", "A4", "A5"):
        sim_df, _summary = _run_once(
            cfg_path=cfg_path,
            scenario="S3",
            attack_id=attack_id,
            verifier_impl="s3_mev",
            seed=seed,
            max_windows=max_windows,
            verification_patch=None,
        )
        col = pd.to_numeric(sim_df.get("soft_score", pd.Series([], dtype=float)), errors="coerce").dropna()
        soft_rows.append(pd.DataFrame({"attack_id": attack_id, "soft_score": col.to_numpy(dtype=float)}))
        print(f"[soft_score] attack={attack_id} n={len(col)}")

    soft_df = pd.concat(soft_rows, ignore_index=True)
    soft_csv = tables_dir / "soft_score_distribution_by_attack.csv"
    soft_df.to_csv(soft_csv, index=False)
    _plot_soft_score_distribution(
        soft_df,
        out_path=figures_dir / "soft_score_distribution_a0_a5.png",
    )
    print(f"Saved soft_score CSV: {soft_csv}")
    print(f"Saved soft_score figure: {figures_dir / 'soft_score_distribution_a0_a5.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tau sensitivity and soft-score distribution report.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config yaml.")
    parser.add_argument("--max-windows", type=int, default=900, help="Use first N windows.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--tau-attack-id", default="A4P", help="Attack id for tau sensitivity curve (default: A4P).")
    args = parser.parse_args()

    run_report(
        cfg_path=args.config,
        max_windows=args.max_windows,
        seed=args.seed,
        tau_attack_id=str(args.tau_attack_id).upper(),
    )


if __name__ == "__main__":
    main()
