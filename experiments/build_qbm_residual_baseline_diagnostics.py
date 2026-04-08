from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


ATTACKS = ("A0", "A4P", "A5")
MODEL_ORDER = ("QBM", "Logistic Regression", "One-class SVM", "Isolation Forest")
MODEL_COLORS = {
    "QBM": "#0F766E",
    "Logistic Regression": "#A16207",
    "One-class SVM": "#6D28D9",
    "Isolation Forest": "#2563EB",
}
EDGE_COLOR = "#1F2937"
GRID_COLOR = "#94A3B8"
IEEE_2COL_WIDTH = 6.9


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "legend.fontsize": 7.7,
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
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)
    ax.grid(axis=grid_axis, linestyle=(0, (3, 3)), linewidth=0.6, color=GRID_COLOR, alpha=0.35)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.03,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=EDGE_COLOR,
    )


def _save(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _render_markdown_table(headers: List[str], rows: Iterable[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _load_frames(paths: Dict[str, Path]) -> pd.DataFrame:
    frames = []
    for attack_id, path in paths.items():
        df = pd.read_csv(path)
        df = df[df["qbm_stage2_eligible"].fillna(0).astype(float) > 0].copy()
        df["attack_id"] = attack_id
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _feature_columns(df: pd.DataFrame) -> List[str]:
    cols = sorted(col for col in df.columns if col.startswith("qbm_feat_"))
    if not cols:
        raise ValueError("No residual feature columns starting with 'qbm_feat_' were found.")
    return cols


def _fit_models(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    benign_a0 = df[(df["attack_id"] == "A0") & (df["malicious_visible"].fillna(0).astype(int) == 0)].copy()
    if benign_a0.empty:
        raise ValueError("No benign A0 accepted windows were found for calibration.")

    scaler = StandardScaler().fit(df[feature_cols].to_numpy())

    lr_train = df[
        ((df["attack_id"] == "A0") & (df["malicious_visible"].fillna(0).astype(int) == 0))
        | (df["malicious_visible"].fillna(0).astype(int) == 1)
    ].copy()
    lr_y = (lr_train["malicious_visible"].fillna(0).astype(int) == 0).astype(int).to_numpy()
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    lr.fit(scaler.transform(lr_train[feature_cols].to_numpy()), lr_y)

    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(scaler.transform(benign_a0[feature_cols].to_numpy()))

    isolation_forest = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42,
    )
    isolation_forest.fit(benign_a0[feature_cols].to_numpy())

    score_map = {
        "QBM": pd.to_numeric(df["q_score_shadow"], errors="coerce").to_numpy(),
        "Logistic Regression": lr.predict_proba(scaler.transform(df[feature_cols].to_numpy()))[:, 1],
        "One-class SVM": ocsvm.decision_function(scaler.transform(df[feature_cols].to_numpy())),
        "Isolation Forest": isolation_forest.score_samples(df[feature_cols].to_numpy()),
    }

    thresholds = {
        "QBM": float(pd.to_numeric(benign_a0["qbm_threshold"], errors="coerce").dropna().iloc[0]),
        "Logistic Regression": float(np.quantile(score_map["Logistic Regression"][benign_a0.index.to_numpy()], 0.01)),
        "One-class SVM": float(np.quantile(score_map["One-class SVM"][benign_a0.index.to_numpy()], 0.01)),
        "Isolation Forest": float(np.quantile(score_map["Isolation Forest"][benign_a0.index.to_numpy()], 0.01)),
    }
    return score_map, thresholds


def _build_summary(df: pd.DataFrame, score_map: Dict[str, np.ndarray], thresholds: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        threshold = thresholds[model_name]
        model_scores = score_map[model_name]
        for attack_id in ATTACKS:
            subset = df[df["attack_id"] == attack_id].copy()
            idx = subset.index.to_numpy()
            scores = model_scores[idx]
            benign_mask = subset["malicious_visible"].fillna(0).astype(int).to_numpy() == 0
            malicious_mask = ~benign_mask
            benign_scores = scores[benign_mask]
            malicious_scores = scores[malicious_mask]
            rows.append(
                {
                    "model": model_name,
                    "attack_id": attack_id,
                    "threshold": threshold,
                    "n_windows": int(len(subset)),
                    "n_benign": int(benign_mask.sum()),
                    "n_malicious": int(malicious_mask.sum()),
                    "benign_mean_score": float(np.mean(benign_scores)) if benign_scores.size else float("nan"),
                    "malicious_mean_score": float(np.mean(malicious_scores)) if malicious_scores.size else float("nan"),
                    "score_gap": float(np.mean(benign_scores) - np.mean(malicious_scores))
                    if benign_scores.size and malicious_scores.size
                    else float("nan"),
                    "benign_veto_rate": float(np.mean(benign_scores < threshold)) if benign_scores.size else float("nan"),
                    "malicious_veto_rate": float(np.mean(malicious_scores < threshold))
                    if malicious_scores.size
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _build_threshold_sweep(
    df: pd.DataFrame,
    score_map: Dict[str, np.ndarray],
    *,
    quantiles: List[float],
) -> pd.DataFrame:
    benign_a0 = df[(df["attack_id"] == "A0") & (df["malicious_visible"].fillna(0).astype(int) == 0)].copy()
    rows = []
    for model_name in MODEL_ORDER:
        scores_all = score_map[model_name]
        benign_ref = scores_all[benign_a0.index.to_numpy()]
        for quantile in quantiles:
            threshold = float(np.quantile(benign_ref, quantile))
            for attack_id in ATTACKS:
                subset = df[df["attack_id"] == attack_id].copy()
                idx = subset.index.to_numpy()
                scores = scores_all[idx]
                benign_mask = subset["malicious_visible"].fillna(0).astype(int).to_numpy() == 0
                malicious_mask = ~benign_mask
                benign_scores = scores[benign_mask]
                malicious_scores = scores[malicious_mask]
                rows.append(
                    {
                        "model": model_name,
                        "attack_id": attack_id,
                        "quantile": quantile,
                        "threshold": threshold,
                        "benign_veto_rate": float(np.mean(benign_scores < threshold)) if benign_scores.size else float("nan"),
                        "malicious_veto_rate": float(np.mean(malicious_scores < threshold))
                        if malicious_scores.size
                        else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _write_outputs(
    summary_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    *,
    output_dir: Path,
    label: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{label}" if label else ""

    summary_csv = output_dir / f"qbm_classical_baseline_summary{suffix}.csv"
    sweep_csv = output_dir / f"qbm_classical_baseline_sweep{suffix}.csv"
    summary_md = output_dir / f"qbm_classical_baseline_summary{suffix}.md"

    summary_df.to_csv(summary_csv, index=False)
    sweep_df.to_csv(sweep_csv, index=False)

    main_rows = summary_df[summary_df["attack_id"].isin(["A4P", "A5"])].copy()
    main_rows["attack_order"] = main_rows["attack_id"].map({"A4P": 0, "A5": 1}).fillna(99)
    main_rows = main_rows.sort_values(["model", "attack_order"]).reset_index(drop=True)

    summary_md.write_text(
        _render_markdown_table(
            ["Model", "Attack", "Benign veto", "Malicious veto", "Score gap"],
            [
                [
                    str(row["model"]),
                    str(row["attack_id"]),
                    f"{float(row['benign_veto_rate']):.4f}" if pd.notna(row["benign_veto_rate"]) else "",
                    f"{float(row['malicious_veto_rate']):.4f}" if pd.notna(row["malicious_veto_rate"]) else "",
                    f"{float(row['score_gap']):.4f}" if pd.notna(row["score_gap"]) else "",
                ]
                for _, row in main_rows.iterrows()
            ],
        ),
        encoding="utf-8",
    )

    print(f"saved: {summary_csv}")
    print(f"saved: {sweep_csv}")
    print(f"saved: {summary_md}")


def _plot(summary_df: pd.DataFrame, *, out_base: Path) -> None:
    _paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_2COL_WIDTH, 3.1), sharey=True)
    attack_titles = {
        "A4P": "Accepted-Side Diagnostic on A4P",
        "A5": "Accepted-Side Diagnostic on A5",
    }

    for panel_idx, attack_id in enumerate(("A4P", "A5")):
        ax = axes[panel_idx]
        panel = summary_df[summary_df["attack_id"] == attack_id].copy()
        panel["model_order"] = panel["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)}).fillna(99)
        panel = panel.sort_values("model_order")

        x = np.arange(len(panel))
        width = 0.34
        benign = panel["benign_veto_rate"].to_numpy(dtype=float) * 100.0
        malicious = panel["malicious_veto_rate"].to_numpy(dtype=float) * 100.0

        bars_m = ax.bar(
            x - width / 2.0,
            malicious,
            width=width,
            color=[MODEL_COLORS[name] for name in panel["model"]],
            edgecolor=EDGE_COLOR,
            linewidth=0.8,
            label="Malicious veto",
        )
        bars_b = ax.bar(
            x + width / 2.0,
            benign,
            width=width,
            color="white",
            edgecolor=EDGE_COLOR,
            linewidth=0.8,
            hatch="///",
            label="Benign veto",
        )

        for bars in (bars_m, bars_b):
            for bar in bars:
                val = bar.get_height()
                if np.isfinite(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        val + 1.1,
                        f"{val:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.2,
                        color=EDGE_COLOR,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(["QBM", "LR", "OC-SVM", "IF"])
        ax.set_ylim(0.0, 105.0)
        ax.set_ylabel("Veto rate (%)" if panel_idx == 0 else "")
        ax.set_title(attack_titles[attack_id])
        _style_axes(ax)
        _panel_label(ax, "A" if panel_idx == 0 else "B")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#64748B", edgecolor=EDGE_COLOR, linewidth=0.8),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=EDGE_COLOR, linewidth=0.8, hatch="///"),
    ]
    fig.legend(handles, ["Malicious veto", "Benign veto"], frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(w_pad=1.2)
    _save(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build accepted-side classical residual baseline diagnostics against the locked QBM operating point."
    )
    parser.add_argument("--a0", default="results/tables/sim_S3_A0_s3_qbm_strongq_quantile_0p0100.csv", help="Accepted-side A0 CSV.")
    parser.add_argument("--a4p", default="results/tables/sim_S3_A4P_s3_qbm_strongq_quantile_0p0100.csv", help="Accepted-side A4P CSV.")
    parser.add_argument("--a5", default="results/tables/sim_S3_A5_s3_qbm_strongq_quantile_0p0100.csv", help="Accepted-side A5 CSV.")
    parser.add_argument("--output-dir", default="results/tables", help="Directory for CSV/MD outputs.")
    parser.add_argument(
        "--figure-dir",
        default="results/figures/qiskit_qbm/main",
        help="Directory for the diagnostic figure.",
    )
    parser.add_argument("--label", default="300w", help="Output suffix label.")
    args = parser.parse_args()

    paths = {
        "A0": Path(args.a0),
        "A4P": Path(args.a4p),
        "A5": Path(args.a5),
    }
    df = _load_frames(paths)
    feature_cols = _feature_columns(df)
    score_map, thresholds = _fit_models(df, feature_cols)
    summary_df = _build_summary(df, score_map, thresholds)
    sweep_df = _build_threshold_sweep(df, score_map, quantiles=[0.01, 0.02, 0.03, 0.05, 0.07])

    _write_outputs(
        summary_df,
        sweep_df,
        output_dir=Path(args.output_dir),
        label=str(args.label).strip(),
    )
    out_base = Path(args.figure_dir) / f"qbm_classical_baseline_diagnostics_{str(args.label).strip()}"
    _plot(summary_df, out_base=out_base)
    print(f"saved: {out_base.with_suffix('.png')}")
    print(f"saved: {out_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
