from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "baseline": "#6C8AE4",
    "pipeline": "#E7A34B",
    "benign": "#C44E52",
    "malicious": "#2F6B3B",
    "tcp": "#4B5563",
    "a4p": "#0F766E",
    "a5": "#A16207",
    "highlight": "#B91C1C",
}
EDGECOLOR = "#1F2937"
GRIDCOLOR = "#94A3B8"
IEEE_2COL_WIDTH = 6.9
METHOD_LABELS = {
    "s3_only": "S3 Only",
    "s3_qbm_strongq": "S3 + QBM + StrongQ",
    "s3_strongq": "S3 + StrongQ",
    "s3_qbm": "S3 + QBM",
}
ATTACK_LABELS = {
    "A0": "A0 TCP",
    "A4P": "A4P FTR",
    "A5": "A5 FTR",
}


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


def _style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["left"].set_color(EDGECOLOR)
    ax.spines["bottom"].set_color(EDGECOLOR)
    ax.grid(axis=grid_axis, linestyle=(0, (3, 3)), linewidth=0.6, color=GRIDCOLOR, alpha=0.35)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=EDGECOLOR,
    )


def _save(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _fmt_pct(value: float) -> str:
    return f"{float(value) * 100.0:.1f}%"


def _load_pipeline_eval(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["backend_mode"] == "exact_state"].copy()


def _load_tradeoff(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_main_overview(
    pipeline_df: pd.DataFrame,
    *,
    methods: List[str],
    out_base: Path,
) -> None:
    _paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_2COL_WIDTH, 3.2), gridspec_kw={"width_ratios": [1.35, 1.0]})

    method_rows = []
    for method in methods:
        rows = pipeline_df[pipeline_df["pipeline_mode"] == method]
        if rows.empty:
            continue
        method_rows.append(
            {
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "A0": float(rows.loc[rows["attack_id"] == "A0", "tcp"].iloc[0]),
                "A4P": float(rows.loc[rows["attack_id"] == "A4P", "ftr"].iloc[0]),
                "A5": float(rows.loc[rows["attack_id"] == "A5", "ftr"].iloc[0]),
            }
        )
    if len(method_rows) < 2:
        raise ValueError("Main overview requires at least two methods in pipeline evaluation results.")

    ax = axes[0]
    categories = ["A0", "A4P", "A5"]
    x = np.arange(len(categories))
    width = 0.34
    offsets = np.linspace(-width / 2.0, width / 2.0, num=len(method_rows))
    colors = [COLORS["baseline"], COLORS["pipeline"], "#8B5CF6", "#0EA5E9"]

    for idx, row in enumerate(method_rows):
        values = np.array([row["A0"], row["A4P"], row["A5"]], dtype=float) * 100.0
        bars = ax.bar(
            x + offsets[idx],
            values,
            width=width if len(method_rows) == 2 else width * 0.82,
            color=colors[idx % len(colors)],
            edgecolor=EDGECOLOR,
            linewidth=0.8,
            label=row["label"],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.8,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=EDGECOLOR,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_LABELS[item] for item in categories])
    ax.set_ylabel("Rate (%)")
    ax.set_title("Main Pipeline Result")
    ax.legend(frameon=False, ncol=1, loc="upper right")
    ax.set_ylim(0.0, max(25.0, float(ax.get_ylim()[1])))
    _style_axes(ax)
    _panel_label(ax, "A")

    ax2 = axes[1]
    veto_rows = pipeline_df[pipeline_df["pipeline_mode"] == "s3_qbm_strongq"].copy()
    veto_rows = veto_rows[veto_rows["attack_id"].isin(["A0", "A4P", "A5"])]
    veto_rows["attack_order"] = veto_rows["attack_id"].map({"A0": 0, "A4P": 1, "A5": 2}).fillna(99)
    veto_rows = veto_rows.sort_values("attack_order")
    x2 = np.arange(len(veto_rows))
    benign = veto_rows["qbm_benign_veto_count"].to_numpy(dtype=float)
    malicious = veto_rows["qbm_malicious_veto_count"].to_numpy(dtype=float)
    bars_b = ax2.bar(
        x2,
        benign,
        width=0.56,
        color=COLORS["benign"],
        edgecolor=EDGECOLOR,
        linewidth=0.8,
        label="Benign veto",
    )
    bars_m = ax2.bar(
        x2,
        malicious,
        width=0.56,
        bottom=benign,
        color=COLORS["malicious"],
        edgecolor=EDGECOLOR,
        linewidth=0.8,
        label="Malicious veto",
    )
    totals = benign + malicious
    for xpos, total in zip(x2, totals):
        ax2.text(xpos, total + 0.35, f"{int(total)}", ha="center", va="bottom", fontsize=7.5, color=EDGECOLOR)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(veto_rows["attack_id"].tolist())
    ax2.set_ylabel("Veto Count")
    ax2.set_title("QBM Veto Composition")
    ax2.legend(frameon=False, loc="upper right")
    ax2.set_ylim(0.0, max(20.0, float(np.nanmax(totals)) * 1.25 if len(totals) > 0 else 1.0))
    _style_axes(ax2)
    _panel_label(ax2, "B")

    fig.tight_layout(w_pad=1.4)
    _save(fig, out_base)


def plot_tradeoff(
    tradeoff_df: pd.DataFrame,
    *,
    selected_quantile: float,
    out_base: Path,
) -> None:
    _paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_2COL_WIDTH, 3.15), gridspec_kw={"width_ratios": [1.2, 1.0]})

    trade = tradeoff_df.copy()
    trade = trade.sort_values("Quantile").reset_index(drop=True)
    quantiles = trade["Quantile"].to_numpy(dtype=float)

    ax = axes[0]
    ax.plot(
        quantiles,
        trade["A0 TCP"].to_numpy(dtype=float) * 100.0,
        marker="o",
        color=COLORS["tcp"],
        linewidth=1.8,
        markersize=5.5,
        label="A0 TCP",
    )
    ax.plot(
        quantiles,
        trade["A4P FTR"].to_numpy(dtype=float) * 100.0,
        marker="s",
        color=COLORS["a4p"],
        linewidth=1.8,
        markersize=5.5,
        label="A4P FTR",
    )
    ax.plot(
        quantiles,
        trade["A5 FTR"].to_numpy(dtype=float) * 100.0,
        marker="^",
        color=COLORS["a5"],
        linewidth=1.8,
        markersize=5.8,
        label="A5 FTR",
    )
    ax.axvline(float(selected_quantile), color=COLORS["highlight"], linestyle="--", linewidth=1.1, label=f"Selected q={selected_quantile:.2f}")
    ax.set_xlabel("Benign calibration quantile")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Operating-Point Trade-off")
    ax.legend(frameon=False, loc="upper left")
    _style_axes(ax)
    _panel_label(ax, "A")

    ax2 = axes[1]
    ax2.plot(
        quantiles,
        trade["Benign veto count"].to_numpy(dtype=float),
        marker="o",
        color=COLORS["benign"],
        linewidth=1.8,
        markersize=5.5,
        label="Benign veto count",
    )
    ax2.plot(
        quantiles,
        trade["Malicious veto count"].to_numpy(dtype=float),
        marker="s",
        color=COLORS["malicious"],
        linewidth=1.8,
        markersize=5.5,
        label="Malicious veto count",
    )
    ax2.axvline(float(selected_quantile), color=COLORS["highlight"], linestyle="--", linewidth=1.1)
    chosen = trade.loc[(trade["Quantile"] - float(selected_quantile)).abs() < 1.0e-12]
    if not chosen.empty:
        row = chosen.iloc[0]
        ax2.scatter([selected_quantile], [row["Benign veto count"]], color=COLORS["highlight"], s=20, zorder=4)
        ax2.scatter([selected_quantile], [row["Malicious veto count"]], color=COLORS["highlight"], s=20, zorder=4)
    ax2.set_xlabel("Benign calibration quantile")
    ax2.set_ylabel("Count")
    ax2.set_title("Veto Composition vs Quantile")
    ax2.legend(frameon=False, loc="upper left")
    _style_axes(ax2)
    _panel_label(ax2, "B")

    fig.tight_layout(w_pad=1.4)
    _save(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-facing QBM figures from the locked 300-window results.")
    parser.add_argument("--pipeline-eval", default="results/tables/qiskit_qbm_pipeline_eval.csv", help="Pipeline eval CSV path.")
    parser.add_argument("--tradeoff", default="results/tables/qiskit_qbm_paper_tradeoff_table_300w.csv", help="Tradeoff CSV path.")
    parser.add_argument("--selected-quantile", type=float, default=0.01, help="Selected operating-point quantile.")
    parser.add_argument("--label", default="300w", help="Output suffix label.")
    parser.add_argument(
        "--output-dir",
        default="results/figures/qiskit_qbm/main",
        help="Directory to write the paper-facing QBM figures.",
    )
    parser.add_argument(
        "--methods",
        default="s3_only,s3_qbm_strongq",
        help="Comma-separated methods for the main overview figure.",
    )
    args = parser.parse_args()

    pipeline_df = _load_pipeline_eval(Path(args.pipeline_eval))
    tradeoff_df = _load_tradeoff(Path(args.tradeoff))
    methods = [item.strip() for item in str(args.methods).split(",") if item.strip()]

    figures_dir = Path(args.output_dir)
    label = str(args.label).strip()
    suffix = f"_{label}" if label else ""

    plot_main_overview(
        pipeline_df,
        methods=methods,
        out_base=figures_dir / f"qiskit_qbm_main_overview{suffix}",
    )
    plot_tradeoff(
        tradeoff_df,
        selected_quantile=float(args.selected_quantile),
        out_base=figures_dir / f"qiskit_qbm_operating_point_tradeoff{suffix}",
    )

    print(f"saved: {figures_dir / f'qiskit_qbm_main_overview{suffix}.png'}")
    print(f"saved: {figures_dir / f'qiskit_qbm_main_overview{suffix}.pdf'}")
    print(f"saved: {figures_dir / f'qiskit_qbm_operating_point_tradeoff{suffix}.png'}")
    print(f"saved: {figures_dir / f'qiskit_qbm_operating_point_tradeoff{suffix}.pdf'}")


if __name__ == "__main__":
    main()
