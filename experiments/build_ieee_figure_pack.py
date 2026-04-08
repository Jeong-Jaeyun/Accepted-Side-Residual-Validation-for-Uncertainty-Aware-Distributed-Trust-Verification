from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qbm.train import apply_overrides, load_config, run_simulation, save_outputs


ROOT = Path(__file__).resolve().parent.parent
IEEE_FIG_DIR = ROOT / "IEEE" / "fig"
DEFAULT_MIRROR_DIR = ROOT / "results" / "figures" / "ieee_tdsc_pack"
PORT_CONFIGS = {
    "Antwerp": "configs/datasets/antwerp.yaml",
    "Cape Town": "configs/datasets/cape_town.yaml",
    "Los Angeles": "configs/datasets/los_angeles.yaml",
    "Singapore": "configs/datasets/singapore.yaml",
}

MIRROR_PDF_PNG_BASES = [
    "soft_trust_score_distribution_by_attack",
    "soft_score_distribution_a0_a5",
    "threshold_sensitivity_analysis_a4",
    "tau_sensitivity_a4_asr_vs_latency",
    "tau_sensitivity_asr_vs_latency",
    "strongq_a4_a5_ftr",
    "strongq_a4_a5_ftr_asr",
    "strongq_a4p_a5_ftr_asr",
    "fig1_cross_port_summary_panel",
    "fig2_a5_ftr_focus",
]
MIRROR_CSV_FILES = [
    "soft_score_distribution_by_attack.csv",
    "soft_trust_score_distribution_by_attack.csv",
    "tau_sensitivity_a4.csv",
    "strongq_a4p_a5_ftr_asr.csv",
    "cross_port_a5_source_data.csv",
]
COLORS = {
    "baseline": "#6C8AE4",
    "strongq": "#F1A340",
    "a0": "#1F77B4",
    "a4": "#FF7F0E",
    "a4p": "#D62728",
    "a5": "#2CA02C",
    "grid": "#CBD5E1",
    "edge": "#1F2937",
    "dark": "#111827",
    "gray": "#6B7280",
    "delta_good": "#2F6B3B",
    "delta_neutral": "#4B5563",
    "a0_calls": "#E5E7EB",
    "a5_calls": "#6B7280",
}


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["left"].set_color(COLORS["edge"])
    ax.spines["bottom"].set_color(COLORS["edge"])
    ax.grid(axis=grid_axis, linestyle=(0, (3, 3)), linewidth=0.7, color=COLORS["grid"], alpha=0.8)


def _save_fig(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _copy_alias(src_base: Path, dst_base: Path) -> None:
    for suffix in (".png", ".pdf"):
        src = src_base.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, dst_base.with_suffix(suffix))


def _mirror_ieee_outputs(src_dir: Path, dst_dir: Path) -> None:
    figures_dir = dst_dir / "figures"
    source_dir = dst_dir / "source_data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    for stem in MIRROR_PDF_PNG_BASES:
        for suffix in (".png", ".pdf"):
            src = src_dir / f"{stem}{suffix}"
            if src.exists():
                shutil.copy2(src, figures_dir / src.name)
    for name in MIRROR_CSV_FILES:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, source_dir / name)


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


def _prepare_cfg(
    cfg_path: str,
    *,
    attack_id: str,
    pipeline_mode: str,
    seed: int | None,
    max_windows: int | None,
    verification_patch: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    cfg = apply_overrides(cfg, scenario="S3", attack_id=attack_id, verifier_impl="s3_mev", seed=seed)
    cfg.setdefault("verification", {})
    cfg["verification"]["pipeline_mode"] = str(pipeline_mode).strip().lower()
    if verification_patch:
        cfg["verification"].update(dict(verification_patch))
    cfg.setdefault("experiments", {})
    cfg["experiments"]["enable_injection"] = str(attack_id).upper() != "A0"
    cfg["experiments"]["attack_id"] = str(attack_id).upper()
    _adjust_injection_window(cfg, max_windows)
    return cfg


def _run_once(
    cfg_path: str,
    *,
    attack_id: str,
    pipeline_mode: str,
    seed: int | None,
    max_windows: int | None,
    verification_patch: Mapping[str, Any] | None = None,
    save_artifacts: bool = False,
) -> Tuple[pd.DataFrame, Any, Dict[str, Any], Dict[str, str]]:
    cfg = _prepare_cfg(
        cfg_path,
        attack_id=attack_id,
        pipeline_mode=pipeline_mode,
        seed=seed,
        max_windows=max_windows,
        verification_patch=verification_patch,
    )
    sim_df, summary = run_simulation(cfg, max_windows=max_windows)
    runtime = dict(sim_df.attrs.get("verifier_runtime", {}))
    paths = save_outputs(cfg, sim_df, summary) if save_artifacts else {}
    return sim_df, summary, runtime, paths


def _resolve_tau_and_margin(cfg_path: str, *, seed: int | None, max_windows: int | None) -> Tuple[float, float]:
    sim_df, _summary, runtime, _paths = _run_once(
        cfg_path,
        attack_id="A0",
        pipeline_mode="s3_only",
        seed=seed,
        max_windows=max_windows,
        save_artifacts=True,
    )
    tau = runtime.get("tau_precalibrated_value", runtime.get("tau_final"))
    gray = runtime.get("gray_margin_precalibrated_value", runtime.get("gray_margin_final"))
    if tau is None or not np.isfinite(float(tau)):
        tau_series = pd.to_numeric(sim_df.get("tau", pd.Series(dtype=float)), errors="coerce").dropna()
        tau = float(tau_series.iloc[-1]) if not tau_series.empty else 0.72
    if gray is None or not np.isfinite(float(gray)):
        gray_series = pd.to_numeric(sim_df.get("gray_margin", pd.Series(dtype=float)), errors="coerce").dropna()
        gray = float(gray_series.iloc[-1]) if not gray_series.empty else 0.05
    return float(tau), float(gray)


def build_soft_trust_distribution(
    *,
    cfg_path: str,
    max_windows: int,
    seed: int | None,
    out_dir: Path,
) -> None:
    tau, gray = _resolve_tau_and_margin(cfg_path, seed=seed, max_windows=max_windows)
    lower = max(0.0, tau - gray)
    upper = min(1.0, tau + gray)
    attacks = ["A0", "A4", "A4P", "A5"]
    rows: List[pd.DataFrame] = []
    for attack_id in attacks:
        sim_df, _summary, _runtime, _paths = _run_once(
            cfg_path,
            attack_id=attack_id,
            pipeline_mode="s3_only",
            seed=seed,
            max_windows=max_windows,
            save_artifacts=True,
        )
        scores = pd.to_numeric(sim_df.get("soft_score", pd.Series(dtype=float)), errors="coerce").dropna()
        rows.append(pd.DataFrame({"attack_id": attack_id, "soft_score": scores.to_numpy(dtype=float)}))
        print(f"soft-score attack={attack_id} n={len(scores)}")

    soft_df = pd.concat(rows, ignore_index=True)
    soft_df.to_csv(out_dir / "soft_score_distribution_by_attack.csv", index=False)
    soft_df.to_csv(out_dir / "soft_trust_score_distribution_by_attack.csv", index=False)

    _paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    bins = np.linspace(
        float(max(0.85, soft_df["soft_score"].min() - 0.002)),
        float(min(0.99, soft_df["soft_score"].max() + 0.002)),
        34,
    )
    color_map = {"A0": COLORS["a0"], "A4": COLORS["a4"], "A4P": COLORS["a4p"], "A5": COLORS["a5"]}

    for ax, density, ylabel in (
        (axes[0], True, "Density"),
        (axes[1], False, "Count"),
    ):
        ax.axvspan(lower, upper, color="#D1D5DB", alpha=0.35, label="Gray-zone" if ax is axes[0] else None)
        ax.axvline(tau, color=COLORS["dark"], linestyle="--", linewidth=1.6, label=r"Threshold $\tau$" if ax is axes[0] else None)
        for attack_id in attacks:
            vals = soft_df.loc[soft_df["attack_id"] == attack_id, "soft_score"].to_numpy(dtype=float)
            ax.hist(
                vals,
                bins=bins,
                density=density,
                histtype="step",
                linewidth=2.0,
                color=color_map[attack_id],
                label=attack_id if ax is axes[0] else None,
            )
        ax.set_xlabel("Soft score")
        ax.set_ylabel(ylabel)
        _style_axes(ax)

    fig.suptitle(
        "Soft Trust Score Distribution by Attack\n"
        + rf"Threshold $\tau$={tau:.4f}, gray-zone=[{lower:.4f}, {upper:.4f}]",
        y=1.04,
        fontsize=12.5,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    _save_fig(fig, out_dir / "soft_trust_score_distribution_by_attack")
    _copy_alias(out_dir / "soft_trust_score_distribution_by_attack", out_dir / "soft_score_distribution_a0_a5")


def build_threshold_sensitivity(
    *,
    cfg_path: str,
    max_windows: int,
    seed: int | None,
    out_dir: Path,
) -> None:
    tau_base, _gray = _resolve_tau_and_margin(cfg_path, seed=seed, max_windows=max_windows)
    tau_values = [max(0.0, min(1.0, tau_base + offset)) for offset in (-0.03, -0.015, 0.0, 0.015, 0.03)]
    rows: List[Dict[str, Any]] = []
    for pipeline_mode, label in (("s3_only", "S3-MEV"), ("s3_strongq", "StrongQ")):
        for tau in tau_values:
            _sim_df, summary, _runtime, _paths = _run_once(
                cfg_path,
                attack_id="A4",
                pipeline_mode=pipeline_mode,
                seed=seed,
                max_windows=max_windows,
                verification_patch={"s3_auto_tau_from_a0": False, "s3_soft_tau": float(tau)},
                save_artifacts=True,
            )
            rows.append(
                {
                    "attack_id": "A4",
                    "pipeline_mode": pipeline_mode,
                    "label": label,
                    "tau_base": float(tau_base),
                    "tau": float(tau),
                    "asr": float(summary.asr),
                    "ftr": float(summary.ftr),
                    "latency_ms_mean": float(summary.latency_ms_mean),
                    "processed_tps_mean": float(summary.processed_tps_mean),
                }
            )
            print(f"threshold attack=A4 pipeline={pipeline_mode} tau={tau:.4f} ASR={summary.asr:.4f} latency={summary.latency_ms_mean:.1f}")

    df = pd.DataFrame(rows).sort_values(["pipeline_mode", "tau"]).reset_index(drop=True)
    df.to_csv(out_dir / "tau_sensitivity_a4.csv", index=False)

    _paper_style()
    fig, ax1 = plt.subplots(figsize=(7.6, 4.0))
    ax2 = ax1.twinx()
    series_cfg = {
        "S3-MEV": {"color": "#C75C0C", "marker": "o"},
        "StrongQ": {"color": "#1F8A78", "marker": "o"},
    }
    for label in ("S3-MEV", "StrongQ"):
        sub = df[df["label"] == label].sort_values("tau")
        color = series_cfg[label]["color"]
        marker = series_cfg[label]["marker"]
        ax1.plot(
            sub["tau"],
            sub["asr"],
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=7,
            label=f"{label} ASR",
        )
        ax2.plot(
            sub["tau"],
            sub["latency_ms_mean"],
            color=color,
            marker="s",
            linewidth=1.9,
            markersize=6.5,
            linestyle=(0, (5, 4)),
            markerfacecolor="white",
            label=f"{label} Latency",
        )
    ax1.axvline(float(tau_base), color=COLORS["gray"], linestyle=":", linewidth=1.3, label=rf"Baseline $\tau_0$")
    ax1.text(float(tau_base) + 0.001, max(df["asr"].max(), 0.05) * 1.06, rf"$\tau_0$={tau_base:.3f}", color=COLORS["gray"], fontsize=8.8)
    ax1.set_xlabel(r"Threshold $\tau$")
    ax1.set_ylabel("ASR")
    ax2.set_ylabel("Latency (ms)")
    ax1.set_ylim(0.0, min(1.0, max(0.68, float(df["asr"].max()) * 1.15)))
    ax2.set_ylim(0.0, max(2500.0, float(df["latency_ms_mean"].max()) * 1.1))
    ax1.set_title("Threshold Sensitivity Analysis\nA4 replay-evasion scenario", pad=10)
    _style_axes(ax1)
    ax2.spines["right"].set_color(COLORS["edge"])
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    _save_fig(fig, out_dir / "threshold_sensitivity_analysis_a4")
    _copy_alias(out_dir / "threshold_sensitivity_analysis_a4", out_dir / "tau_sensitivity_a4_asr_vs_latency")
    _copy_alias(out_dir / "threshold_sensitivity_analysis_a4", out_dir / "tau_sensitivity_asr_vs_latency")


def build_strongq_ftr_figure(
    *,
    cfg_path: str,
    max_windows: int,
    seed: int | None,
    out_dir: Path,
) -> None:
    attacks = [("A4P", "A4P\nNear-replay"), ("A5", "A5\nCompromise")]
    rows: List[Dict[str, Any]] = []
    for attack_id, _label in attacks:
        for pipeline_mode, method_label in (("s3_only", "S3-MEV"), ("s3_strongq", "StrongQ")):
            _sim_df, summary, _runtime, _paths = _run_once(
                cfg_path,
                attack_id=attack_id,
                pipeline_mode=pipeline_mode,
                seed=seed,
                max_windows=max_windows,
                save_artifacts=True,
            )
            rows.append(
                {
                    "attack_id": attack_id,
                    "method": method_label,
                    "pipeline_mode": pipeline_mode,
                    "ftr": float(summary.ftr),
                    "asr": float(summary.asr),
                    "tcp": float(summary.tcp),
                }
            )
            print(f"strongq figure attack={attack_id} pipeline={pipeline_mode} FTR={summary.ftr:.4f} ASR={summary.asr:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "strongq_a4p_a5_ftr_asr.csv", index=False)

    _paper_style()
    fig, ax = plt.subplots(figsize=(6.1, 4.2))
    x = np.arange(len(attacks))
    width = 0.28
    mev = [float(df[(df["attack_id"] == attack_id) & (df["method"] == "S3-MEV")]["ftr"].iloc[0]) * 100.0 for attack_id, _ in attacks]
    strongq = [float(df[(df["attack_id"] == attack_id) & (df["method"] == "StrongQ")]["ftr"].iloc[0]) * 100.0 for attack_id, _ in attacks]
    bars1 = ax.bar(x - width / 2.0, mev, width=width, color="#D96B10", edgecolor=COLORS["edge"], linewidth=0.8, hatch="//", label="S3-MEV")
    bars2 = ax.bar(x + width / 2.0, strongq, width=width, color="#198675", edgecolor=COLORS["edge"], linewidth=0.8, hatch=".", label="StrongQ")
    ymax = max(max(mev), max(strongq)) * 1.22
    ax.set_ylim(0.0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _attack, label in attacks])
    ax.set_ylabel("Rate")
    ax.set_title("FTR")
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2)
    for bars, values in ((bars1, mev), (bars2, strongq)):
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + ymax * 0.018, f"{value/100.0:.3f}", ha="center", va="bottom", fontsize=8.2, color=COLORS["edge"])
    for idx, (mev_val, sq_val) in enumerate(zip(mev, strongq)):
        delta = ((sq_val / mev_val) - 1.0) * 100.0 if mev_val > 1.0e-12 else 0.0
        top = max(mev_val, sq_val) + ymax * 0.06
        ax.plot([idx - width / 2.0, idx - width / 2.0, idx + width / 2.0, idx + width / 2.0], [top, top + ymax * 0.02, top + ymax * 0.02, top], color=COLORS["gray"], linewidth=1.0)
        color = COLORS["delta_good"] if delta < 0 else COLORS["delta_neutral"]
        ax.text(idx, top + ymax * 0.035, f"{abs(delta):.1f}% lower" if delta < 0 else f"{delta:+.1f}% change", ha="center", va="bottom", fontsize=8.2, color=color)
    _save_fig(fig, out_dir / "strongq_a4p_a5_ftr_asr")
    _copy_alias(out_dir / "strongq_a4p_a5_ftr_asr", out_dir / "strongq_a4_a5_ftr")
    _copy_alias(out_dir / "strongq_a4p_a5_ftr_asr", out_dir / "strongq_a4_a5_ftr_asr")


def _cross_port_rows(
    *,
    max_windows: int,
    seed: int | None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for port_name, cfg_path in PORT_CONFIGS.items():
        for attack_id in ("A0", "A5"):
            for pipeline_mode, method_label in (("s3_only", "S3-MEV"), ("s3_strongq", "StrongQ")):
                sim_df, summary, _runtime, _paths = _run_once(
                    cfg_path,
                    attack_id=attack_id,
                    pipeline_mode=pipeline_mode,
                    seed=seed,
                    max_windows=max_windows,
                    save_artifacts=True,
                )
                rows.append(
                    {
                        "port": port_name,
                        "attack_id": attack_id,
                        "method": method_label,
                        "pipeline_mode": pipeline_mode,
                        "ftr": float(summary.ftr),
                        "asr": float(summary.asr),
                        "tcp": float(summary.tcp),
                        "latency_ms_mean": float(summary.latency_ms_mean),
                        "processed_tps_mean": float(summary.processed_tps_mean),
                        "strongq_calls": int(summary.n_strongq_called),
                        "windows": int(len(sim_df)),
                        "strongq_call_rate_pct": float(summary.n_strongq_called / max(len(sim_df), 1) * 100.0),
                    }
                )
                print(f"cross-port port={port_name} attack={attack_id} pipeline={pipeline_mode} FTR={summary.ftr:.4f} calls={summary.n_strongq_called}")
    return pd.DataFrame(rows)


def build_cross_port_figures(
    *,
    max_windows: int,
    seed: int | None,
    out_dir: Path,
) -> None:
    frame = _cross_port_rows(max_windows=max_windows, seed=seed)
    frame.to_csv(out_dir / "cross_port_a5_source_data.csv", index=False)

    ports = list(PORT_CONFIGS.keys())
    a5 = frame[frame["attack_id"] == "A5"].copy()
    a0 = frame[frame["attack_id"] == "A0"].copy()

    ftr_mev = [float(a5[(a5["port"] == port) & (a5["method"] == "S3-MEV")]["ftr"].iloc[0]) * 100.0 for port in ports]
    ftr_sq = [float(a5[(a5["port"] == port) & (a5["method"] == "StrongQ")]["ftr"].iloc[0]) * 100.0 for port in ports]
    calls_a0 = [int(a0[(a0["port"] == port) & (a0["method"] == "StrongQ")]["strongq_calls"].iloc[0]) for port in ports]
    calls_a5 = [int(a5[(a5["port"] == port) & (a5["method"] == "StrongQ")]["strongq_calls"].iloc[0]) for port in ports]
    delta_pp = [sq - mev for sq, mev in zip(ftr_sq, ftr_mev)]

    _paper_style()
    fig, axes = plt.subplot_mosaic([["top", "top"], ["left", "right"]], figsize=(7.4, 4.9), constrained_layout=True)
    x = np.arange(len(ports))
    width = 0.32

    ax = axes["top"]
    bars1 = ax.bar(x - width / 2.0, ftr_mev, width=width, color=COLORS["baseline"], edgecolor=COLORS["edge"], linewidth=0.8, hatch="///", label="S3-MEV")
    bars2 = ax.bar(x + width / 2.0, ftr_sq, width=width, color=COLORS["strongq"], edgecolor=COLORS["edge"], linewidth=0.8, hatch="xx", label="StrongQ")
    ax.set_xticks(x)
    ax.set_xticklabels(ports)
    ax.set_ylabel("FTR (%)")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    _style_axes(ax)
    ymax = max(max(ftr_mev), max(ftr_sq)) * 1.23
    ax.set_ylim(0.0, ymax)
    for bars, values in ((bars1, ftr_mev), (bars2, ftr_sq)):
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + ymax * 0.015, f"{value:.1f}", ha="center", va="bottom", fontsize=8.0)
    ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold", color=COLORS["edge"])

    ax = axes["left"]
    bars1 = ax.bar(x - width / 2.0, calls_a0, width=width, color=COLORS["a0_calls"], edgecolor=COLORS["edge"], linewidth=0.8, hatch="///", label="A0")
    bars2 = ax.bar(x + width / 2.0, calls_a5, width=width, color=COLORS["a5_calls"], edgecolor=COLORS["edge"], linewidth=0.8, hatch="xx", label="A5")
    ax.set_xticks(x)
    ax.set_xticklabels(["Antwerp", "Cape\nTown", "Los\nAngeles", "Singapore"])
    ax.set_ylabel("Calls (#)")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    _style_axes(ax)
    ymax = max(max(calls_a0), max(calls_a5)) * 1.22
    ax.set_ylim(0.0, ymax)
    for bars, values in ((bars1, calls_a0), (bars2, calls_a5)):
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + ymax * 0.02, f"{int(value)}", ha="center", va="bottom", fontsize=7.9)
    ax.text(-0.18, 1.05, "(b)", transform=ax.transAxes, fontsize=10, fontweight="bold", color=COLORS["edge"])

    ax = axes["right"]
    y = np.arange(len(ports))
    ax.axvline(0.0, color=COLORS["edge"], linewidth=1.0)
    for yi, val in zip(y, delta_pp):
        color = COLORS["delta_good"] if val < 0 else COLORS["delta_neutral"]
        ax.hlines(yi, xmin=min(0.0, val), xmax=max(0.0, val), color=color, linewidth=2.0)
        ax.plot(val, yi, "o", color=color, markersize=6.5, markeredgecolor=COLORS["edge"], markeredgewidth=0.5)
        ax.text(val - 0.12 if val < 0 else val + 0.12, yi, f"{val:+.2f}", va="center", ha="right" if val < 0 else "left", fontsize=8.2, color=COLORS["edge"])
    ax.set_yticks(y)
    ax.set_yticklabels(["Antwerp", "Cape\nTown", "Los\nAngeles", "Singapore"])
    ax.set_xlabel(r"$\Delta$FTR (pp)")
    _style_axes(ax, grid_axis="x")
    ax.invert_yaxis()
    span = max(abs(float(np.min(delta_pp))), abs(float(np.max(delta_pp))), 0.4)
    ax.set_xlim(min(-0.4, -span * 1.35), max(0.4, span * 1.35))
    ax.text(-0.16, 1.05, "(c)", transform=ax.transAxes, fontsize=10, fontweight="bold", color=COLORS["edge"])

    _save_fig(fig, out_dir / "fig1_cross_port_summary_panel")

    _paper_style()
    fig2, ax2 = plt.subplots(figsize=(7.0, 3.7))
    bars1 = ax2.bar(x - width / 2.0, ftr_mev, width=width, color=COLORS["baseline"], edgecolor=COLORS["edge"], linewidth=0.8, hatch="///", label="S3-MEV")
    bars2 = ax2.bar(x + width / 2.0, ftr_sq, width=width, color=COLORS["strongq"], edgecolor=COLORS["edge"], linewidth=0.8, hatch="xx", label="StrongQ")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ports)
    ax2.set_ylabel("False trust rate (%)")
    ax2.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    _style_axes(ax2)
    ymax = max(max(ftr_mev), max(ftr_sq)) * 1.27
    ax2.set_ylim(0.0, ymax)
    for bars, values in ((bars1, ftr_mev), (bars2, ftr_sq)):
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2.0, value + ymax * 0.015, f"{value:.1f}", ha="center", va="bottom", fontsize=8.0)
    for idx, delta in enumerate(delta_pp):
        top = max(ftr_mev[idx], ftr_sq[idx]) + ymax * 0.08
        ax2.plot([idx - width / 2.0, idx - width / 2.0, idx + width / 2.0, idx + width / 2.0], [top, top + ymax * 0.025, top + ymax * 0.025, top], color=COLORS["delta_good"] if delta < 0 else COLORS["delta_neutral"], linewidth=1.0)
        ax2.text(idx, top + ymax * 0.04, rf"$\Delta$ {delta:+.2f} pp", ha="center", va="bottom", fontsize=8.3, color=COLORS["delta_good"] if delta < 0 else COLORS["delta_neutral"])
    _save_fig(fig2, out_dir / "fig2_a5_ftr_focus")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the IEEE figure pack directly into IEEE/fig.")
    parser.add_argument("--config", default="configs/default.yaml", help="Base config for Busan figures.")
    parser.add_argument("--max-windows", type=int, default=300, help="Window count for Busan figures.")
    parser.add_argument("--cross-port-windows", type=int, default=300, help="Window count for cross-port figures.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--skip-cross-port", action="store_true", help="Skip cross-port figure generation.")
    parser.add_argument("--output-dir", default=str(IEEE_FIG_DIR), help="Directory to write the IEEE paper figures.")
    parser.add_argument(
        "--mirror-dir",
        default=str(DEFAULT_MIRROR_DIR),
        help="Optional mirror directory inside results/figures for organized copies of the IEEE figure pack.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_soft_trust_distribution(
        cfg_path=args.config,
        max_windows=max(1, int(args.max_windows)),
        seed=args.seed,
        out_dir=out_dir,
    )
    build_threshold_sensitivity(
        cfg_path=args.config,
        max_windows=max(1, int(args.max_windows)),
        seed=args.seed,
        out_dir=out_dir,
    )
    build_strongq_ftr_figure(
        cfg_path=args.config,
        max_windows=max(1, int(args.max_windows)),
        seed=args.seed,
        out_dir=out_dir,
    )
    if not bool(args.skip_cross_port):
        build_cross_port_figures(
            max_windows=max(1, int(args.cross_port_windows)),
            seed=args.seed,
            out_dir=out_dir,
        )

    mirror_dir = Path(args.mirror_dir) if str(args.mirror_dir).strip() else None
    if mirror_dir is not None:
        _mirror_ieee_outputs(out_dir, mirror_dir)

    print(f"saved figure pack to: {out_dir}")
    if mirror_dir is not None:
        print(f"mirrored figure pack to: {mirror_dir}")


if __name__ == "__main__":
    main()
