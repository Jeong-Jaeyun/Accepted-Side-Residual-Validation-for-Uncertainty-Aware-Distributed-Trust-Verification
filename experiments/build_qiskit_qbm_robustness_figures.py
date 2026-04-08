from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "exact": "#2563EB",
    "shot": "#D97706",
    "benign": "#DC2626",
    "malicious": "#15803D",
    "a0": "#374151",
    "a4p": "#0F766E",
    "a5": "#7C3AED",
    "highlight": "#B91C1C",
}
EDGECOLOR = "#1F2937"
GRIDCOLOR = "#94A3B8"
IEEE_2COL_WIDTH = 6.9


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.4,
            "axes.titlesize": 9,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
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
        -0.15,
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


def plot_shot_sensitivity(df: pd.DataFrame, out_base: Path) -> None:
    _paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_2COL_WIDTH, 3.1), gridspec_kw={"width_ratios": [1.1, 1.0]})
    attack_colors = {"A0": COLORS["a0"], "A4P": COLORS["a4p"]}

    ax = axes[0]
    for attack_id, group in df.groupby("attack_id", sort=False):
        if attack_id not in attack_colors:
            continue
        ordered = group.sort_values("shots")
        ax.plot(
            ordered["shots"],
            ordered["qbm_shot_std_mean"],
            marker="o",
            linewidth=1.8,
            markersize=5.0,
            color=attack_colors[attack_id],
            label=f"{attack_id} mean q-score std",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Shots (log2)")
    ax.set_ylabel("Estimated q-score std")
    ax.set_title("Shot Noise Sensitivity")
    ax.legend(frameon=False, loc="upper right")
    _style_axes(ax)
    _panel_label(ax, "A")

    ax2 = axes[1]
    a0 = df[df["attack_id"] == "A0"].sort_values("shots")
    a4p = df[df["attack_id"] == "A4P"].sort_values("shots")
    ax2.plot(
        a0["shots"],
        a0["tcp"] * 100.0,
        marker="o",
        linewidth=1.8,
        markersize=5.0,
        color=COLORS["a0"],
        label="A0 TCP",
    )
    ax2.plot(
        a4p["shots"],
        a4p["ftr"] * 100.0,
        marker="s",
        linewidth=1.8,
        markersize=5.0,
        color=COLORS["a4p"],
        label="A4P FTR",
    )
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Shots (log2)")
    ax2.set_ylabel("Rate (%)")
    ax2.set_title("Pipeline Sensitivity vs Shots")
    ax2.legend(frameon=False, loc="upper right")
    _style_axes(ax2)
    _panel_label(ax2, "B")

    fig.tight_layout(w_pad=1.4)
    _save(fig, out_base)


def plot_noise_robustness(df: pd.DataFrame, out_base: Path) -> None:
    _paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_2COL_WIDTH, 3.0))

    a0 = df[df["attack_id"] == "A0"].sort_values("noise_level")
    a4p = df[df["attack_id"] == "A4P"].sort_values("noise_level")

    ax = axes[0]
    ax.plot(
        a0["noise_level"],
        a0["tcp"] * 100.0,
        marker="o",
        linewidth=1.8,
        markersize=5.0,
        color=COLORS["a0"],
    )
    ax.set_xlabel("Uniform noise level p")
    ax.set_ylabel("A0 TCP (%)")
    ax.set_title("Benign Stability Under Injected Noise")
    _style_axes(ax)
    _panel_label(ax, "A")

    ax2 = axes[1]
    ax2.plot(
        a4p["noise_level"],
        a4p["ftr"] * 100.0,
        marker="s",
        linewidth=1.8,
        markersize=5.0,
        color=COLORS["a4p"],
    )
    ax2.set_xlabel("Uniform noise level p")
    ax2.set_ylabel("A4P FTR (%)")
    ax2.set_title("Attack Robustness Under Injected Noise")
    _style_axes(ax2)
    _panel_label(ax2, "B")

    fig.tight_layout(w_pad=1.4)
    _save(fig, out_base)


def _boxplot_positions() -> tuple[list[float], list[str]]:
    positions = [1.0, 1.75, 3.0, 3.75, 5.0, 5.75]
    labels = ["A0", "", "A4P", "", "A5", ""]
    return positions, labels


def plot_backend_compare(compare_df: pd.DataFrame, windows_df: pd.DataFrame, out_base: Path) -> None:
    _paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 3.0), gridspec_kw={"width_ratios": [1.25, 1.0, 1.05]})

    ax = axes[0]
    positions, labels = _boxplot_positions()
    plotted = []
    colors = []
    for attack_id in ["A0", "A4P", "A5"]:
        for backend_mode in ["exact_state", "aer_shot"]:
            subset = windows_df[
                (windows_df["attack_id"] == attack_id)
                & (windows_df["backend_mode"] == backend_mode)
            ]["q_score_shadow_minus_threshold"].dropna()
            plotted.append(subset.to_numpy(dtype=float))
            colors.append(COLORS["exact"] if backend_mode == "exact_state" else COLORS["shot"])
    bp = ax.boxplot(
        plotted,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": EDGECOLOR, "linewidth": 0.9},
        whiskerprops={"color": EDGECOLOR, "linewidth": 0.8},
        capprops={"color": EDGECOLOR, "linewidth": 0.8},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(EDGECOLOR)
        patch.set_linewidth(0.8)
        patch.set_alpha(0.80)
    ax.axhline(0.0, color=COLORS["highlight"], linestyle="--", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("q-score minus threshold")
    ax.set_title("Backend Score Distribution")
    _style_axes(ax)
    _panel_label(ax, "A")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=COLORS["exact"], lw=6, label="Exact-state"),
            plt.Line2D([0], [0], color=COLORS["shot"], lw=6, label="Aer shot"),
        ],
        frameon=False,
        loc="lower left",
    )

    ax2 = axes[1]
    categories = ["A0", "A4P", "A5"]
    x = np.arange(len(categories))
    width = 0.34
    exact_values = []
    shot_values = []
    for attack_id in categories:
        exact_row = compare_df[(compare_df["backend_mode"] == "exact_state") & (compare_df["attack_id"] == attack_id)].iloc[0]
        shot_row = compare_df[(compare_df["backend_mode"] == "aer_shot") & (compare_df["attack_id"] == attack_id)].iloc[0]
        metric_name = "tcp" if attack_id == "A0" else "ftr"
        exact_values.append(float(exact_row[metric_name]) * 100.0)
        shot_values.append(float(shot_row[metric_name]) * 100.0)
    bars1 = ax2.bar(x - width / 2.0, exact_values, width=width, color=COLORS["exact"], edgecolor=EDGECOLOR, linewidth=0.8, label="Exact-state")
    bars2 = ax2.bar(x + width / 2.0, shot_values, width=width, color=COLORS["shot"], edgecolor=EDGECOLOR, linewidth=0.8, label="Aer shot")
    for bars in (bars1, bars2):
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.6, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["A0 TCP", "A4P FTR", "A5 FTR"])
    ax2.set_ylabel("Rate (%)")
    ax2.set_title("Pipeline Outcome by Backend")
    ax2.legend(frameon=False, loc="upper right")
    _style_axes(ax2)
    _panel_label(ax2, "B")

    ax3 = axes[2]
    benign_exact = []
    malicious_exact = []
    benign_shot = []
    malicious_shot = []
    for attack_id in categories:
        exact_row = compare_df[(compare_df["backend_mode"] == "exact_state") & (compare_df["attack_id"] == attack_id)].iloc[0]
        shot_row = compare_df[(compare_df["backend_mode"] == "aer_shot") & (compare_df["attack_id"] == attack_id)].iloc[0]
        benign_exact.append(float(exact_row["qbm_benign_veto_count"]))
        malicious_exact.append(float(exact_row["qbm_malicious_veto_count"]))
        benign_shot.append(float(shot_row["qbm_benign_veto_count"]))
        malicious_shot.append(float(shot_row["qbm_malicious_veto_count"]))
    ax3.bar(x - width / 2.0, benign_exact, width=width, color=COLORS["benign"], edgecolor=EDGECOLOR, linewidth=0.8, label="Benign veto")
    ax3.bar(x - width / 2.0, malicious_exact, width=width, bottom=benign_exact, color=COLORS["malicious"], edgecolor=EDGECOLOR, linewidth=0.8, label="Malicious veto")
    ax3.bar(x + width / 2.0, benign_shot, width=width, color=COLORS["benign"], edgecolor=EDGECOLOR, linewidth=0.8, alpha=0.45)
    ax3.bar(x + width / 2.0, malicious_shot, width=width, bottom=benign_shot, color=COLORS["malicious"], edgecolor=EDGECOLOR, linewidth=0.8, alpha=0.45)
    for xpos, total_exact, total_shot in zip(x, np.array(benign_exact) + np.array(malicious_exact), np.array(benign_shot) + np.array(malicious_shot)):
        ax3.text(xpos - width / 2.0, total_exact + 0.3, f"{int(total_exact)}", ha="center", va="bottom", fontsize=7.0)
        ax3.text(xpos + width / 2.0, total_shot + 0.3, f"{int(total_shot)}", ha="center", va="bottom", fontsize=7.0)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.set_ylabel("Veto count")
    ax3.set_title("Backend Veto Composition")
    _style_axes(ax3)
    _panel_label(ax3, "C")

    fig.tight_layout(w_pad=1.2)
    _save(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build robustness appendix figures for the Qiskit forensic gate.")
    parser.add_argument("--shot", default="results/tables/qiskit_qbm_robustness_shot_sensitivity.csv", help="Shot sensitivity CSV.")
    parser.add_argument("--noise", default="results/tables/qiskit_qbm_robustness_noise_sweep.csv", help="Noise sweep CSV.")
    parser.add_argument("--backend", default="results/tables/qiskit_qbm_robustness_backend_compare.csv", help="Backend compare CSV.")
    parser.add_argument("--backend-windows", default="results/tables/qiskit_qbm_robustness_backend_windows.csv", help="Backend per-window CSV.")
    parser.add_argument("--label", default="robustness", help="Output suffix label.")
    parser.add_argument(
        "--output-dir",
        default="results/figures/qiskit_qbm/robustness",
        help="Directory to write the robustness appendix figures.",
    )
    args = parser.parse_args()

    figures_dir = Path(args.output_dir)
    suffix = f"_{str(args.label).strip()}" if str(args.label).strip() else ""

    shot_df = pd.read_csv(args.shot)
    noise_df = pd.read_csv(args.noise)
    backend_df = pd.read_csv(args.backend)
    backend_windows_df = pd.read_csv(args.backend_windows)

    plot_shot_sensitivity(shot_df, figures_dir / f"qiskit_qbm_shot_sensitivity{suffix}")
    plot_noise_robustness(noise_df, figures_dir / f"qiskit_qbm_noise_robustness{suffix}")
    plot_backend_compare(backend_df, backend_windows_df, figures_dir / f"qiskit_qbm_backend_compare{suffix}")

    print(f"saved: {figures_dir / f'qiskit_qbm_shot_sensitivity{suffix}.png'}")
    print(f"saved: {figures_dir / f'qiskit_qbm_noise_robustness{suffix}.png'}")
    print(f"saved: {figures_dir / f'qiskit_qbm_backend_compare{suffix}.png'}")


if __name__ == "__main__":
    main()
