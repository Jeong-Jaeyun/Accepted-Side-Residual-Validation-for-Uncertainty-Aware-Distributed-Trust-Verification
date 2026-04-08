from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
import pandas as pd
import yaml


DATASETS = {
    "antwerp": "Antwerp",
    "cape_town": "Cape Town",
    "los_angeles": "Los Angeles",
    "singapore": "Singapore",
}

METRICS = [
    "processed_tps_mean",
    "latency_ms_mean",
    "backlog_max",
    "dropped_sum",
    "dropped_by_verification_sum",
    "dropped_by_network_sum",
    "dropped_by_overflow_sum",
    "policy_fired_ratio",
    "asr",
    "ftr",
    "tcp",
    "ttd_windows",
    "auroc",
    "auprc",
    "mcc",
    "precision",
    "recall",
    "f1_score",
    "fpr_at_tpr95",
    "detection_delay_windows",
    "cost_sensitive_score",
    "n_gray_zone",
    "n_strongq_called",
    "n_flip_mev_to_reject",
    "n_flip_mev_to_accept",
    "n_strongq_agree",
    "n_strongq_disagree",
]

SCENARIO_ORDER = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}
ATTACK_ORDER = {"A0": 0, "A1": 1, "A2": 2, "A3": 3, "A4": 4, "A4P": 5, "A5": 6}
VERIFIER_ORDER = {
    "none": 0,
    "s2_strict": 1,
    "s3_mev": 2,
    "qbm_verifier": 3,
    "strongq_verifier": 4,
}
MATRIX_PRIORITY = {"primary": 0, "s3_ablation": 1}


def _load_results_dir(config_path: str) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    return Path(cfg.get("project", {}).get("results_dir", "results"))


def _load_one_dataset(base_results_dir: Path, dataset: str, pretty_name: str) -> pd.DataFrame:
    path = base_results_dir / dataset / "tables" / "benchmark_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["dataset"] = pretty_name
    df["_matrix_priority"] = df["matrix"].astype(str).map(MATRIX_PRIORITY).fillna(99)
    df = df.sort_values(
        ["scenario", "attack_id", "verifier_name", "_matrix_priority", "run_idx"],
        kind="mergesort",
    )
    df = df.drop_duplicates(subset=["scenario", "attack_id", "verifier_name"], keep="first").copy()
    df["source_matrix"] = df["matrix"].astype(str)
    return df


def _sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_scenario_order"] = out["scenario"].astype(str).map(SCENARIO_ORDER).fillna(999)
    out["_attack_order"] = out["attack_id"].astype(str).map(ATTACK_ORDER).fillna(999)
    out["_verifier_order"] = out["verifier_name"].astype(str).map(VERIFIER_ORDER).fillna(999)
    out = out.sort_values(
        ["_scenario_order", "_attack_order", "_verifier_order", "scenario", "attack_id", "verifier_name"],
        kind="mergesort",
    )
    return out.drop(columns=["_scenario_order", "_attack_order", "_verifier_order"])


def build_metric_tables(base_results_dir: Path, out_dir: Path) -> Dict[str, pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for dataset, pretty_name in DATASETS.items():
        frames.append(_load_one_dataset(base_results_dir, dataset, pretty_name))
    merged = pd.concat(frames, ignore_index=True)

    row_keys = ["scenario", "attack_id", "verifier_name", "source_matrix"]
    master_rows = _sort_rows(merged[row_keys].drop_duplicates().reset_index(drop=True))
    tables: Dict[str, pd.DataFrame] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        pivot = merged[row_keys + ["dataset", metric]].pivot(
            index=row_keys,
            columns="dataset",
            values=metric,
        ).reset_index()
        pivot.columns.name = None
        for pretty_name in DATASETS.values():
            if pretty_name not in pivot.columns:
                pivot[pretty_name] = pd.NA
        pivot = master_rows.merge(pivot, on=row_keys, how="left")
        pivot = pivot[row_keys + list(DATASETS.values())]
        pivot.to_csv(out_dir / f"{metric}_by_dataset.csv", index=False, na_rep="NA")
        tables[metric] = pivot

    merged_out = _sort_rows(
        merged[
            [
                "dataset",
                "scenario",
                "attack_id",
                "verifier_name",
                "source_matrix",
                *METRICS,
            ]
        ].rename(columns={"dataset": "dataset_region"})
    )
    merged_out.to_csv(out_dir / "all_metrics_long.csv", index=False, na_rep="NA")
    return tables


def _format_html_table(df: pd.DataFrame) -> str:
    styled = df.copy()
    numeric_cols = [c for c in styled.columns if c not in {"scenario", "attack_id", "verifier_name", "source_matrix"}]
    for col in numeric_cols:
        styled[col] = styled[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}" if isinstance(x, (int, float)) else x)
    return styled.to_html(index=False, border=0, classes="metric-table")


def _format_value(metric: str, value: object) -> str:
    if pd.isna(value):
        return "NA"
    if metric.startswith("n_"):
        return str(int(round(float(value))))
    if metric in {"ttd_windows", "detection_delay_windows"}:
        fval = float(value)
        return str(int(fval)) if fval.is_integer() else f"{fval:.2f}"
    fval = float(value)
    return f"{fval:.4f}"


def _format_display_table(metric: str, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dataset_cols = [c for c in out.columns if c in DATASETS.values()]
    for col in dataset_cols:
        out[col] = out[col].map(lambda v: _format_value(metric, v))
    return out


def write_png_tables(tables: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        if metric not in tables:
            continue
        display_df = _format_display_table(metric, tables[metric])
        n_rows, n_cols = display_df.shape
        fig_width = max(16.0, 2.25 * n_cols)
        fig_height = max(4.0, 0.36 * (n_rows + 3))
        font_size = 8 if n_rows >= 30 else 9

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")
        ax.set_title(f"{metric} by dataset region", fontsize=14, fontweight="bold", pad=16)

        table = ax.table(
            cellText=display_df.values.tolist(),
            colLabels=display_df.columns.tolist(),
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1.0, 1.2)

        try:
            table.auto_set_column_width(col=list(range(n_cols)))
        except Exception:
            pass

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#d1d5db")
            cell.set_linewidth(0.6)
            if row == 0:
                cell.set_facecolor("#e5eef9")
                cell.set_text_props(weight="bold", color="#0f172a")
                if col < 4:
                    cell.get_text().set_ha("left")
            else:
                cell.set_facecolor("#f8fafc" if row % 2 == 0 else "#ffffff")
                if col < 4:
                    cell.get_text().set_ha("left")

        fig.tight_layout()
        fig.savefig(png_dir / f"{metric}_by_dataset.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def _row_labels(df: pd.DataFrame) -> List[str]:
    labels = []
    for row in df.itertuples(index=False):
        labels.append(f"{row.scenario} | {row.attack_id} | {row.verifier_name}")
    return labels


def _pick_cmap_and_norm(metric: str, values: np.ndarray):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return plt.get_cmap("viridis").copy(), None

    if metric == "mcc":
        bound = max(abs(float(finite.min())), abs(float(finite.max())), 1.0e-9)
        cmap = plt.get_cmap("coolwarm").copy()
        norm = mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
        return cmap, norm

    cmap = plt.get_cmap("viridis").copy()
    vmin = float(finite.min())
    vmax = float(finite.max())
    if abs(vmax - vmin) < 1.0e-12:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm


def write_heatmap_plots(tables: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    heatmap_dir = out_dir / "heatmap"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    dataset_cols = list(DATASETS.values())
    for metric in METRICS:
        if metric not in tables:
            continue

        df = tables[metric].copy()
        values_df = df[dataset_cols].apply(pd.to_numeric, errors="coerce")
        values = values_df.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(values)
        cmap, norm = _pick_cmap_and_norm(metric, values)
        cmap.set_bad(color="#e5e7eb")

        n_rows = len(df)
        fig_width = 8.0
        fig_height = max(5.0, 0.34 * n_rows + 1.6)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)

        ax.set_xticks(np.arange(len(dataset_cols)))
        ax.set_xticklabels(dataset_cols, fontsize=10)
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(_row_labels(df), fontsize=8)
        ax.set_title(f"{metric} heatmap", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Dataset Region", fontsize=11)
        ax.set_ylabel("Scenario | Attack | Verifier", fontsize=11)

        ax.set_xticks(np.arange(-0.5, len(dataset_cols), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        finite = values[np.isfinite(values)]
        text_color_dark = "#111827"
        text_color_light = "#f9fafb"
        annotate = n_rows <= 40
        if annotate and finite.size > 0:
            value_span = float(finite.max() - finite.min())
            threshold = float(finite.min() + value_span * 0.5)
            for i in range(n_rows):
                for j in range(len(dataset_cols)):
                    val = values[i, j]
                    if not np.isfinite(val):
                        ax.text(j, i, "NA", ha="center", va="center", fontsize=7, color=text_color_dark)
                        continue
                    text = _format_value(metric, val)
                    color = text_color_light if val >= threshold and metric != "mcc" else text_color_dark
                    if metric == "mcc" and abs(val) >= 0.5 * max(abs(float(finite.min())), abs(float(finite.max())), 1.0e-9):
                        color = text_color_light
                    ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.set_ylabel(metric, rotation=270, labelpad=14)

        fig.tight_layout()
        fig.savefig(heatmap_dir / f"{metric}_heatmap.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def write_html_report(tables: Dict[str, pd.DataFrame], out_path: Path) -> None:
    nav = "\n".join(
        f'<li><a href="#{html.escape(metric)}">{html.escape(metric)}</a></li>'
        for metric in METRICS
        if metric in tables
    )
    body = []
    for metric in METRICS:
        if metric not in tables:
            continue
        body.append(f"<section id=\"{html.escape(metric)}\">")
        body.append(f"<h2>{html.escape(metric)}</h2>")
        body.append(_format_html_table(tables[metric]))
        body.append("</section>")

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Dataset Comparison Tables</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #1f2937;
    }}
    h1, h2 {{
      margin-bottom: 12px;
    }}
    .nav {{
      columns: 3;
      margin-bottom: 24px;
      padding-left: 20px;
    }}
    .metric-table {{
      border-collapse: collapse;
      width: 100%;
      margin-bottom: 36px;
      font-size: 13px;
    }}
    .metric-table th,
    .metric-table td {{
      border: 1px solid #d1d5db;
      padding: 6px 8px;
      text-align: right;
      white-space: nowrap;
    }}
    .metric-table th:nth-child(-n+4),
    .metric-table td:nth-child(-n+4) {{
      text-align: left;
      position: sticky;
      background: #ffffff;
    }}
    .metric-table th {{
      background: #f3f4f6;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    section {{
      margin-bottom: 24px;
    }}
  </style>
</head>
<body>
  <h1>Dataset Comparison Tables</h1>
  <p>Rows are unique scenario / attack / verifier combinations. Dataset regions are shown as columns.</p>
  <ul class="nav">
    {nav}
  </ul>
  {''.join(body)}
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build metric-wise dataset comparison tables.")
    parser.add_argument("--config", default="configs/default.yaml", help="Used only to resolve the base results directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write comparison tables. Defaults to <results>/comparison.",
    )
    args = parser.parse_args()

    base_results_dir = _load_results_dir(args.config)
    out_dir = Path(args.output_dir) if args.output_dir else (base_results_dir / "comparison")
    tables = build_metric_tables(base_results_dir=base_results_dir, out_dir=out_dir)
    write_png_tables(tables, out_dir)
    write_heatmap_plots(tables, out_dir)
    write_html_report(tables, out_dir / "dataset_comparison_tables.html")

    print(f"Saved comparison tables to: {out_dir}")
    print(f"Saved PNG tables to: {out_dir / 'png'}")
    print(f"Saved heatmap plots to: {out_dir / 'heatmap'}")
    print(f"Saved HTML report to: {out_dir / 'dataset_comparison_tables.html'}")


if __name__ == "__main__":
    main()
