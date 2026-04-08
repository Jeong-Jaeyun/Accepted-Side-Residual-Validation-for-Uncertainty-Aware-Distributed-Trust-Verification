from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PORTS = [
    ("antwerp", "Antwerp"),
    ("cape_town", "Cape Town"),
    ("los_angeles", "Los Angeles"),
    ("singapore", "Singapore"),
]
PORT_LABELS_COMPACT = {
    "Antwerp": "Antwerp",
    "Cape Town": "Cape\nTown",
    "Los Angeles": "Los\nAngeles",
    "Singapore": "Singapore",
}
MATRIX_PRIORITY = {"primary": 0, "s3_ablation": 1}
COLORS = {
    "mev": "#6C8AE4",
    "strongq": "#E7A34B",
    "a0": "#D9DEE7",
    "a5": "#6B7280",
    "improve": "#2F6B3B",
    "neutral": "#4B5563",
}
EDGECOLOR = "#1F2937"
GRIDCOLOR = "#94A3B8"
IEEE_2COL_WIDTH = 6.9


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_results_dir(config_path: str) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    return Path(cfg.get("project", {}).get("results_dir", "results"))


def _dedup_runs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_matrix_priority"] = out["matrix"].astype(str).map(MATRIX_PRIORITY).fillna(99)
    out = out.sort_values(
        ["scenario", "attack_id", "verifier_name", "_matrix_priority", "run_idx"],
        kind="mergesort",
    )
    out = out.drop_duplicates(subset=["scenario", "attack_id", "verifier_name"], keep="first")
    return out.drop(columns=["_matrix_priority"])


def _resolve_path(path_str: str) -> Path:
    path = Path(str(path_str).replace("\\", "/"))
    if path.is_absolute():
        return path
    return _repo_root() / path


def _sim_row_count(sim_csv: str) -> int:
    path = _resolve_path(sim_csv)
    if not path.exists():
        return 0
    return int(len(pd.read_csv(path)))


def build_cross_port_frame(results_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for dataset_key, port_name in PORTS:
        bench_path = results_dir / dataset_key / "tables" / "benchmark_matrix.csv"
        if not bench_path.exists():
            raise FileNotFoundError(bench_path)
        bench = _dedup_runs(pd.read_csv(bench_path))

        mev_a5 = bench[
            (bench["scenario"] == "S3")
            & (bench["attack_id"] == "A5")
            & (bench["verifier_name"] == "s3_mev")
        ].iloc[0]
        strongq_a5 = bench[
            (bench["scenario"] == "S3")
            & (bench["attack_id"] == "A5")
            & (bench["verifier_name"] == "strongq_verifier")
        ].iloc[0]
        strongq_a0 = bench[
            (bench["scenario"] == "S3")
            & (bench["attack_id"] == "A0")
            & (bench["verifier_name"] == "strongq_verifier")
        ].iloc[0]

        a0_windows = max(_sim_row_count(str(strongq_a0["sim_csv"])), 1)
        a5_windows = max(_sim_row_count(str(strongq_a5["sim_csv"])), 1)

        rows.append(
            {
                "dataset_key": dataset_key,
                "port": port_name,
                "ftr_mev": float(mev_a5["ftr"]),
                "ftr_strongq": float(strongq_a5["ftr"]),
                "delta_ftr": float(strongq_a5["ftr"] - mev_a5["ftr"]),
                "delta_ftr_pp": float((strongq_a5["ftr"] - mev_a5["ftr"]) * 100.0),
                "asr_mev": float(mev_a5["asr"]),
                "asr_strongq": float(strongq_a5["asr"]),
                "latency_mev": float(mev_a5["latency_ms_mean"]),
                "latency_strongq": float(strongq_a5["latency_ms_mean"]),
                "processed_tps_mev": float(mev_a5["processed_tps_mean"]),
                "processed_tps_strongq": float(strongq_a5["processed_tps_mean"]),
                "calls_a0": int(strongq_a0["n_strongq_called"]),
                "calls_a5": int(strongq_a5["n_strongq_called"]),
                "calls_rate_a0": float(int(strongq_a0["n_strongq_called"]) / a0_windows),
                "calls_rate_a5": float(int(strongq_a5["n_strongq_called"]) / a5_windows),
                "windows_a0": int(a0_windows),
                "windows_a5": int(a5_windows),
            }
        )

    frame = pd.DataFrame(rows)
    frame["port"] = pd.Categorical(frame["port"], categories=[p for _, p in PORTS], ordered=True)
    frame = frame.sort_values("port").reset_index(drop=True)
    return frame


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9,
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


def _save(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["left"].set_color(EDGECOLOR)
    ax.spines["bottom"].set_color(EDGECOLOR)
    ax.grid(axis=grid_axis, linestyle=(0, (3, 3)), linewidth=0.6, color=GRIDCOLOR, alpha=0.35)


def _bar_label(
    ax: plt.Axes,
    bars,
    values: np.ndarray,
    *,
    fmt: str = "{:.1f}",
    dy: float = 0.12,
    fs: int = 8,
) -> None:
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(bar.get_height()) + dy,
            fmt.format(float(val)),
            ha="center",
            va="bottom",
            fontsize=fs,
            color=EDGECOLOR,
        )


def _bracket(ax: plt.Axes, x1: float, x2: float, y: float, text: str, *, h: float = 0.18, color: str = EDGECOLOR) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color, linewidth=0.8, clip_on=False)
    ax.text((x1 + x2) / 2.0, y + h + 0.06, text, ha="center", va="bottom", fontsize=8.5, color=color)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=EDGECOLOR,
    )


def _set_xticklabels(ax: plt.Axes, labels: List[str], *, compact: bool = False) -> None:
    if compact:
        labels = [PORT_LABELS_COMPACT.get(label, label) for label in labels]
    ax.set_xticklabels(labels)


def plot_a5_ftr(frame: pd.DataFrame, out_dir: Path) -> None:
    _paper_style()
    x = np.arange(len(frame))
    width = 0.32
    ftr_mev = frame["ftr_mev"].to_numpy(dtype=float) * 100.0
    ftr_strongq = frame["ftr_strongq"].to_numpy(dtype=float) * 100.0
    delta_pp = frame["delta_ftr_pp"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(IEEE_2COL_WIDTH, 3.05))
    bars_mev = ax.bar(
        x - width / 2.0,
        ftr_mev,
        width=width,
        color=COLORS["mev"],
        edgecolor=EDGECOLOR,
        linewidth=0.8,
        hatch="///",
        label="S3-MEV",
    )
    bars_sq = ax.bar(
        x + width / 2.0,
        ftr_strongq,
        width=width,
        color=COLORS["strongq"],
        edgecolor=EDGECOLOR,
        linewidth=0.8,
        hatch="xx",
        label="StrongQ",
    )

    ax.set_xticks(x)
    _set_xticklabels(ax, frame["port"].astype(str).tolist())
    ax.set_ylabel("False trust rate (%)")
    _style_axes(ax, grid_axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))

    ymax = max(float(np.nanmax(ftr_mev)), float(np.nanmax(ftr_strongq))) * 1.24
    ax.set_ylim(0.0, ymax if ymax > 0 else 1.0)
    _bar_label(ax, bars_mev, ftr_mev, fmt="{:.1f}", dy=ymax * 0.012, fs=7.8)
    _bar_label(ax, bars_sq, ftr_strongq, fmt="{:.1f}", dy=ymax * 0.012, fs=7.8)

    for i, delta in enumerate(delta_pp):
        top = max(ftr_mev[i], ftr_strongq[i]) + ymax * 0.05
        _bracket(
            ax,
            x[i] - width / 2.0,
            x[i] + width / 2.0,
            top,
            rf"$\Delta$ {delta:+.2f} pp",
            h=ymax * 0.018,
            color=COLORS["improve"] if delta < 0 else COLORS["neutral"],
        )

    _save(fig, out_dir / "figure_a5_ftr_cross_port")


def plot_strongq_calls(frame: pd.DataFrame, out_dir: Path) -> None:
    _paper_style()
    x = np.arange(len(frame))
    width = 0.32
    calls_a0 = frame["calls_a0"].to_numpy(dtype=float)
    calls_a5 = frame["calls_a5"].to_numpy(dtype=float)
    rate_a0 = frame["calls_rate_a0"].to_numpy(dtype=float) * 100.0
    rate_a5 = frame["calls_rate_a5"].to_numpy(dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(IEEE_2COL_WIDTH, 3.15))
    bars_a0 = ax.bar(
        x - width / 2.0,
        calls_a0,
        width=width,
        color=COLORS["a0"],
        edgecolor=EDGECOLOR,
        linewidth=0.8,
        hatch="///",
        label="A0 (benign)",
    )
    bars_a5 = ax.bar(
        x + width / 2.0,
        calls_a5,
        width=width,
        color=COLORS["a5"],
        edgecolor=EDGECOLOR,
        linewidth=0.8,
        hatch="xx",
        label="A5 (node compromise)",
    )

    ax.set_xticks(x)
    _set_xticklabels(ax, frame["port"].astype(str).tolist())
    ax.set_ylabel("StrongQ calls")
    _style_axes(ax, grid_axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))

    ymax = max(float(np.nanmax(calls_a0)), float(np.nanmax(calls_a5))) * 1.32
    ax.set_ylim(0.0, ymax if ymax > 0 else 1.0)

    for i, bar in enumerate(bars_a0):
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + ymax * 0.016,
            f"{int(round(height))}\n({rate_a0[i]:.1f}%)",
            ha="center",
            va="bottom",
            linespacing=0.95,
            fontsize=7.7,
            color=EDGECOLOR,
        )

    for i, bar in enumerate(bars_a5):
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + ymax * 0.016,
            f"{int(round(height))}\n({rate_a5[i]:.1f}%)",
            ha="center",
            va="bottom",
            linespacing=0.95,
            fontsize=7.7,
            color=EDGECOLOR,
        )

    _save(fig, out_dir / "figure_strongq_invocations")


def plot_delta_ftr(frame: pd.DataFrame, out_dir: Path) -> None:
    _paper_style()
    delta_pp = frame["delta_ftr_pp"].to_numpy(dtype=float)
    y = np.arange(len(frame))
    colors = [COLORS["improve"] if v < 0 else COLORS["neutral"] for v in delta_pp]

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.axvline(0.0, color=EDGECOLOR, linewidth=0.9)
    for yi, val, color in zip(y, delta_pp, colors):
        ax.hlines(yi, xmin=min(0.0, val), xmax=max(0.0, val), color=color, linewidth=2.0)
        ax.plot(val, yi, "o", color=color, markersize=6.5, markeredgecolor=EDGECOLOR, markeredgewidth=0.5)
        if abs(val) < 1.0e-12:
            ax.plot(0.0, yi, "o", color="white", markersize=7.2, markeredgecolor=EDGECOLOR, markeredgewidth=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels(frame["port"].astype(str).tolist())
    ax.set_xlabel(r"$\Delta$FTR (StrongQ - S3-MEV, pp)")
    _style_axes(ax, grid_axis="x")
    ax.invert_yaxis()

    span = max(float(np.max(np.abs(delta_pp))), 0.25)
    ax.set_xlim(min(-0.2, -span * 1.35), max(0.2, span * 1.35))

    for yi, val in zip(y, delta_pp):
        xpos = val - 0.08 if val < 0 else val + 0.08
        ax.text(
            xpos,
            yi,
            f"{val:+.2f} pp",
            va="center",
            ha="right" if val < 0 else "left",
            fontsize=8.5,
            fontweight="bold",
            color=EDGECOLOR,
        )

    ax.text(
        0.01,
        1.03,
        "More negative is better",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#64748B",
    )

    _save(fig, out_dir / "figure_delta_ftr_a5")


def plot_summary_panel(frame: pd.DataFrame, out_dir: Path) -> None:
    _paper_style()
    fig, axes = plt.subplot_mosaic(
        [["ftr", "ftr"], ["calls", "delta"]],
        figsize=(IEEE_2COL_WIDTH, 4.85),
        constrained_layout=True,
    )

    x = np.arange(len(frame))
    width = 0.32
    ports = frame["port"].astype(str).tolist()
    ftr_mev = frame["ftr_mev"].to_numpy(dtype=float) * 100.0
    ftr_sq = frame["ftr_strongq"].to_numpy(dtype=float) * 100.0
    delta_pp = frame["delta_ftr_pp"].to_numpy(dtype=float)
    calls_a0 = frame["calls_a0"].to_numpy(dtype=float)
    calls_a5 = frame["calls_a5"].to_numpy(dtype=float)

    ax = axes["ftr"]
    bars_mev = ax.bar(
        x - width / 2.0,
        ftr_mev,
        width=width,
        color=COLORS["mev"],
        edgecolor=EDGECOLOR,
        linewidth=0.8,
        hatch="///",
        label="S3-MEV",
    )
    bars_sq = ax.bar(
        x + width / 2.0,
        ftr_sq,
        width=width,
        color=COLORS["strongq"],
        edgecolor=EDGECOLOR,
        linewidth=0.8,
        hatch="xx",
        label="StrongQ",
    )
    ax.set_xticks(x)
    _set_xticklabels(ax, ports)
    ax.set_ylabel("FTR (%)")
    _style_axes(ax, grid_axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    _panel_label(ax, "(a)")
    ymax = max(float(np.nanmax(ftr_mev)), float(np.nanmax(ftr_sq))) * 1.20
    ax.set_ylim(0.0, ymax)
    _bar_label(ax, bars_mev, ftr_mev, fmt="{:.1f}", dy=ymax * 0.012, fs=7.3)
    _bar_label(ax, bars_sq, ftr_sq, fmt="{:.1f}", dy=ymax * 0.012, fs=7.3)

    ax = axes["calls"]
    bars_a0 = ax.bar(
        x - width / 2.0,
        calls_a0,
        width=width,
        color=COLORS["a0"],
        edgecolor=EDGECOLOR,
        linewidth=0.8,
        hatch="///",
        label="A0",
    )
    bars_a5 = ax.bar(
        x + width / 2.0,
        calls_a5,
        width=width,
        color=COLORS["a5"],
        edgecolor=EDGECOLOR,
        linewidth=0.8,
        hatch="xx",
        label="A5",
    )
    ax.set_xticks(x)
    _set_xticklabels(ax, ports, compact=True)
    ax.set_ylabel("Calls (#)")
    _style_axes(ax, grid_axis="y")
    _panel_label(ax, "(b)")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    ymax = max(float(np.nanmax(calls_a0)), float(np.nanmax(calls_a5))) * 1.20
    ax.set_ylim(0.0, ymax)
    _bar_label(ax, bars_a0, calls_a0, fmt="{:.0f}", dy=ymax * 0.012, fs=7.5)
    _bar_label(ax, bars_a5, calls_a5, fmt="{:.0f}", dy=ymax * 0.012, fs=7.5)

    ax = axes["delta"]
    colors = [COLORS["improve"] if v < 0 else COLORS["neutral"] for v in delta_pp]
    ax.axvline(0.0, color=EDGECOLOR, linewidth=0.9)
    for yi, val, color in zip(np.arange(len(frame)), delta_pp, colors):
        ax.hlines(yi, xmin=min(0.0, val), xmax=max(0.0, val), color=color, linewidth=2.0)
        ax.plot(val, yi, "o", color=color, markersize=5.5, markeredgecolor=EDGECOLOR, markeredgewidth=0.5)
    ax.set_yticks(np.arange(len(frame)))
    ax.set_yticklabels([PORT_LABELS_COMPACT.get(port, port) for port in ports])
    ax.set_xlabel(r"$\Delta$FTR (pp)")
    _style_axes(ax, grid_axis="x")
    _panel_label(ax, "(c)")
    ax.invert_yaxis()
    span = max(float(np.max(np.abs(delta_pp))), 0.25)
    ax.set_xlim(min(-0.2, -span * 1.35), max(0.2, span * 1.35))
    for yi, val in zip(np.arange(len(frame)), delta_pp):
        xpos = val - 0.08 if val < 0 else val + 0.08
        ax.text(
            xpos,
            yi,
            f"{val:+.2f}",
            va="center",
            ha="right" if val < 0 else "left",
            fontsize=7.8,
            color=EDGECOLOR,
        )

    _save(fig, out_dir / "figure_cross_port_summary_panel")


def write_submission_package(frame: pd.DataFrame, out_dir: Path) -> None:
    alias_pairs = [
        ("figure_cross_port_summary_panel", "fig1_cross_port_summary_panel"),
        ("figure_a5_ftr_cross_port", "fig2_a5_ftr_focus"),
    ]
    for src_base, dst_base in alias_pairs:
        for suffix in (".png", ".pdf"):
            src = out_dir / f"{src_base}{suffix}"
            dst = out_dir / f"{dst_base}{suffix}"
            if src.exists():
                shutil.copy2(src, dst)

    best_port_row = frame.loc[frame["delta_ftr_pp"].idxmin()]
    ports_improved = int((frame["delta_ftr_pp"] < 0).sum())
    a0_min = float(frame["calls_rate_a0"].min() * 100.0)
    a0_max = float(frame["calls_rate_a0"].max() * 100.0)
    a5_min = float(frame["calls_rate_a5"].min() * 100.0)
    a5_max = float(frame["calls_rate_a5"].max() * 100.0)

    caption_md = f"""# IEEE TDSC Figure Package

## Recommended Main Figure

File:
- `fig1_cross_port_summary_panel.png`
- `fig1_cross_port_summary_panel.pdf`

Suggested caption:

> **Fig. 1. Cross-port robustness and selective escalation behavior of the proposed verification framework.** (a) False trust rate (FTR) under the A5 node-compromise scenario across Antwerp, Cape Town, Los Angeles, and Singapore. StrongQ reduces FTR in {ports_improved} of {len(frame)} ports, with the largest reduction observed in {best_port_row['port']} ({best_port_row['ftr_mev']*100.0:.1f}% to {best_port_row['ftr_strongq']*100.0:.1f}%, {best_port_row['delta_ftr_pp']:+.2f} percentage points). (b) Number of StrongQ invocations under benign A0 and adversarial A5 conditions. Under A0, StrongQ is triggered only in {a0_min:.1f}% to {a0_max:.1f}% of windows, whereas under A5 it rises to {a5_min:.1f}% to {a5_max:.1f}%, indicating selective escalation rather than indiscriminate forensic activation. (c) Per-port FTR difference between StrongQ and S3-MEV under A5, where more negative values indicate stronger improvement.

In-text use:

> Fig. 1 shows that the proposed framework preserves its core verification behavior across heterogeneous ports, while StrongQ is activated selectively and yields its clearest benefit under the A5 node-compromise scenario.

## Recommended Detail Figure

File:
- `fig2_a5_ftr_focus.png`
- `fig2_a5_ftr_focus.pdf`

Suggested caption:

> **Fig. 2. Cross-port comparison of FTR under the A5 node-compromise scenario.** StrongQ consistently matches or improves upon S3-MEV across all evaluated ports and provides the largest gain in Cape Town, while Los Angeles shows near-identical behavior between the two verifiers. This pattern supports the claim that StrongQ is most beneficial in ambiguous high-risk cases rather than uniformly across all traffic environments.

In-text use:

> As shown in Fig. 2, the added value of StrongQ is not uniform across ports; instead, its gains concentrate in ports where the A5 scenario produces more ambiguous verification outcomes.
"""
    (out_dir / "captions_tdsc.md").write_text(caption_md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build journal-facing cross-port figures.")
    parser.add_argument("--config", default="configs/default.yaml", help="Used only to resolve the base results directory.")
    parser.add_argument(
        "--output-dir",
        default="results/journal_figures",
        help="Directory to write source CSV and figure files.",
    )
    args = parser.parse_args()

    results_dir = _load_results_dir(args.config)
    out_dir = _resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = build_cross_port_frame(results_dir)
    frame.to_csv(out_dir / "cross_port_a5_source_data.csv", index=False)

    plot_a5_ftr(frame, out_dir)
    plot_strongq_calls(frame, out_dir)
    plot_delta_ftr(frame, out_dir)
    plot_summary_panel(frame, out_dir)
    write_submission_package(frame, out_dir)

    print(f"Saved source CSV: {out_dir / 'cross_port_a5_source_data.csv'}")
    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
