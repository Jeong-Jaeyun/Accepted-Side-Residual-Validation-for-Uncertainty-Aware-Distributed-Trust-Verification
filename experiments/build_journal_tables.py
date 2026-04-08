from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import yaml


PORTS = [
    ("antwerp", "Antwerp"),
    ("cape_town", "Cape Town"),
    ("los_angeles", "Los Angeles"),
    ("singapore", "Singapore"),
]
MATRIX_PRIORITY = {"primary": 0, "s3_ablation": 1}
ATTACK_ORDER = {"A0": 0, "A4": 1, "A5": 2}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_path(path_str: str) -> Path:
    path = Path(str(path_str).replace("\\", "/"))
    if path.is_absolute():
        return path
    return _repo_root() / path


def _load_results_dir(config_path: str) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    return Path(cfg.get("project", {}).get("results_dir", "results"))


def _sim_row_count(sim_csv: str) -> int:
    path = _resolve_path(sim_csv)
    if not path.exists():
        return 0
    return int(len(pd.read_csv(path)))


def _dedup_runs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_matrix_priority"] = out["matrix"].astype(str).map(MATRIX_PRIORITY).fillna(99)
    out = out.sort_values(
        ["scenario", "attack_id", "verifier_name", "_matrix_priority", "run_idx"],
        kind="mergesort",
    )
    out = out.drop_duplicates(subset=["scenario", "attack_id", "verifier_name"], keep="first")
    return out.drop(columns=["_matrix_priority"])


def _fetch_row(bench: pd.DataFrame, attack_id: str, verifier_name: str) -> pd.Series:
    rows = bench[
        (bench["scenario"] == "S3")
        & (bench["attack_id"] == attack_id)
        & (bench["verifier_name"] == verifier_name)
    ]
    if rows.empty:
        raise KeyError(f"Missing row for attack={attack_id}, verifier={verifier_name}")
    return rows.iloc[0]


def build_cross_port_table_frame(results_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for dataset_key, port_name in PORTS:
        bench_path = results_dir / dataset_key / "tables" / "benchmark_matrix.csv"
        if not bench_path.exists():
            raise FileNotFoundError(bench_path)
        bench = _dedup_runs(pd.read_csv(bench_path))

        for attack_id in ("A0", "A4", "A5"):
            mev = _fetch_row(bench, attack_id, "s3_mev")
            strongq = _fetch_row(bench, attack_id, "strongq_verifier")
            windows = max(_sim_row_count(str(strongq["sim_csv"])), 1)

            latency_mev = float(mev["latency_ms_mean"])
            latency_strongq = float(strongq["latency_ms_mean"])
            calls = int(strongq["n_strongq_called"])

            rows.append(
                {
                    "dataset_key": dataset_key,
                    "port": port_name,
                    "attack": attack_id,
                    "ftr_mev": float(mev["ftr"]),
                    "ftr_strongq": float(strongq["ftr"]),
                    "delta_ftr": float(strongq["ftr"] - mev["ftr"]),
                    "asr_mev": float(mev["asr"]),
                    "asr_strongq": float(strongq["asr"]),
                    "strongq_calls": calls,
                    "latency_mev_ms": latency_mev,
                    "latency_strongq_ms": latency_strongq,
                    "overhead_pct": float(((latency_strongq - latency_mev) / latency_mev) * 100.0)
                    if abs(latency_mev) > 1.0e-12
                    else 0.0,
                    "calls_ratio_pct": float(calls / windows * 100.0),
                    "windows": windows,
                }
            )

    frame = pd.DataFrame(rows)
    frame["port"] = pd.Categorical(frame["port"], categories=[p for _, p in PORTS], ordered=True)
    frame["attack"] = pd.Categorical(frame["attack"], categories=["A0", "A4", "A5"], ordered=True)
    frame = frame.sort_values(["port", "attack"]).reset_index(drop=True)
    return frame


def build_table1(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[frame["attack"].isin(["A4", "A5"])].copy()
    return out[
        [
            "port",
            "attack",
            "ftr_mev",
            "ftr_strongq",
            "delta_ftr",
            "asr_mev",
            "asr_strongq",
            "strongq_calls",
        ]
    ].reset_index(drop=True)


def build_table2(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[frame["attack"].isin(["A0", "A5"])].copy()
    return out[
        [
            "port",
            "attack",
            "latency_mev_ms",
            "latency_strongq_ms",
            "overhead_pct",
            "calls_ratio_pct",
        ]
    ].reset_index(drop=True)


def _fmt_prob(value: float) -> str:
    return f"{value:.4f}"


def _fmt_delta(value: float, *, mode: str) -> str:
    text = f"{value:+.4f}"
    if value < -1.0e-12:
        if mode == "md":
            return f"**{text}**"
        if mode == "tex":
            return rf"\textbf{{{text}}}"
    return text


def _fmt_ms(value: float) -> str:
    return f"{value:.1f}"


def _fmt_pct(value: float, *, signed: bool) -> str:
    if signed:
        return f"{value:+.1f}"
    return f"{value:.1f}"


def _render_markdown_table(headers: List[str], rows: Iterable[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def render_table1_markdown(df: pd.DataFrame) -> str:
    headers = [
        "Port",
        "Attack",
        "S3-MEV FTR",
        "StrongQ FTR",
        "Delta FTR",
        "S3-MEV ASR",
        "StrongQ ASR",
        "StrongQ Calls",
    ]
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                str(row["port"]),
                str(row["attack"]),
                _fmt_prob(float(row["ftr_mev"])),
                _fmt_prob(float(row["ftr_strongq"])),
                _fmt_delta(float(row["delta_ftr"]), mode="md"),
                _fmt_prob(float(row["asr_mev"])),
                _fmt_prob(float(row["asr_strongq"])),
                str(int(row["strongq_calls"])),
            ]
        )
    return _render_markdown_table(headers, rows)


def render_table2_markdown(df: pd.DataFrame) -> str:
    headers = [
        "Port",
        "Attack",
        "Latency (MEV, ms)",
        "Latency (StrongQ, ms)",
        "Overhead (%)",
        "Calls Ratio (%)",
    ]
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                str(row["port"]),
                str(row["attack"]),
                _fmt_ms(float(row["latency_mev_ms"])),
                _fmt_ms(float(row["latency_strongq_ms"])),
                _fmt_pct(float(row["overhead_pct"]), signed=True),
                _fmt_pct(float(row["calls_ratio_pct"]), signed=False),
            ]
        )
    return _render_markdown_table(headers, rows)


def render_table1_latex(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cross-port verification performance under replay (A4) and node-compromise (A5) attacks. Negative $\Delta$FTR indicates an improvement over S3-MEV.}",
        r"\label{tab:cross_port_main}",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Port & Attack & S3-MEV FTR & StrongQ FTR & $\Delta$FTR & S3-MEV ASR & StrongQ ASR & StrongQ Calls \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            " & ".join(
                [
                    str(row["port"]),
                    str(row["attack"]),
                    _fmt_prob(float(row["ftr_mev"])),
                    _fmt_prob(float(row["ftr_strongq"])),
                    _fmt_delta(float(row["delta_ftr"]), mode="tex"),
                    _fmt_prob(float(row["asr_mev"])),
                    _fmt_prob(float(row["asr_strongq"])),
                    str(int(row["strongq_calls"])),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_table2_latex(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cross-port latency overhead and StrongQ activation ratio under benign (A0) and node-compromise (A5) conditions. Overhead is computed as $(L_{\mathrm{StrongQ}} - L_{\mathrm{MEV}}) / L_{\mathrm{MEV}} \times 100$.}",
        r"\label{tab:cross_port_efficiency}",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Port & Attack & Latency (MEV, ms) & Latency (StrongQ, ms) & Overhead (\%) & Calls Ratio (\%) \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            " & ".join(
                [
                    str(row["port"]),
                    str(row["attack"]),
                    _fmt_ms(float(row["latency_mev_ms"])),
                    _fmt_ms(float(row["latency_strongq_ms"])),
                    _fmt_pct(float(row["overhead_pct"]), signed=True),
                    _fmt_pct(float(row["calls_ratio_pct"]), signed=False),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_caption_package(frame: pd.DataFrame) -> str:
    table1 = build_table1(frame)
    table2 = build_table2(frame)

    best_row = table1.loc[table1["delta_ftr"].idxmin()]
    improved_ports = int((table1[table1["attack"] == "A5"]["delta_ftr"] < 0).sum())
    a0_min = float(table2[table2["attack"] == "A0"]["calls_ratio_pct"].min())
    a0_max = float(table2[table2["attack"] == "A0"]["calls_ratio_pct"].max())
    a5_min = float(table2[table2["attack"] == "A5"]["calls_ratio_pct"].min())
    a5_max = float(table2[table2["attack"] == "A5"]["calls_ratio_pct"].max())
    oh_abs_max = float(table2["overhead_pct"].abs().max())

    return f"""# IEEE TDSC Table Package

## Table 1

Files:
- `table1_cross_port_main.csv`
- `table1_cross_port_main.md`
- `table1_cross_port_main.tex`

Suggested caption:

> **Table 1. Cross-port verification performance under replay (A4) and node-compromise (A5) attacks.** StrongQ produces little change under A4, where the baseline MEV layer already performs strongly, but reduces FTR under A5 in {improved_ports} of 4 ports. The largest improvement appears in {best_row['port']} ({best_row['delta_ftr']:+.4f} in absolute FTR units). Negative $\\Delta$FTR indicates improvement over S3-MEV.

In-text use:

> Table 1 confirms that the added value of StrongQ is concentrated in the more ambiguous A5 node-compromise setting, whereas its effect is marginal under A4.

## Table 2

Files:
- `table2_cross_port_efficiency.csv`
- `table2_cross_port_efficiency.md`
- `table2_cross_port_efficiency.tex`

Suggested caption:

> **Table 2. Cross-port latency overhead and StrongQ activation ratio under benign (A0) and node-compromise (A5) conditions.** Across ports, StrongQ activation remains low under benign traffic ({a0_min:.1f}% to {a0_max:.1f}% of windows) and increases under A5 ({a5_min:.1f}% to {a5_max:.1f}% of windows), while the absolute latency overhead remains limited within {oh_abs_max:.1f}\\% across the reported settings.

In-text use:

> Table 2 shows that StrongQ is triggered selectively rather than continuously, and that the resulting latency overhead remains modest across heterogeneous ports.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build journal-facing cross-port tables.")
    parser.add_argument("--config", default="configs/default.yaml", help="Used only to resolve the base results directory.")
    parser.add_argument(
        "--output-dir",
        default="results/journal_tables",
        help="Directory to write table source CSV, Markdown, and LaTeX files.",
    )
    args = parser.parse_args()

    results_dir = _load_results_dir(args.config)
    out_dir = _resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = build_cross_port_table_frame(results_dir)
    table1 = build_table1(frame)
    table2 = build_table2(frame)

    table1.to_csv(out_dir / "table1_cross_port_main.csv", index=False)
    table2.to_csv(out_dir / "table2_cross_port_efficiency.csv", index=False)
    (out_dir / "table1_cross_port_main.md").write_text(render_table1_markdown(table1), encoding="utf-8")
    (out_dir / "table2_cross_port_efficiency.md").write_text(render_table2_markdown(table2), encoding="utf-8")
    (out_dir / "table1_cross_port_main.tex").write_text(render_table1_latex(table1), encoding="utf-8")
    (out_dir / "table2_cross_port_efficiency.tex").write_text(render_table2_latex(table2), encoding="utf-8")
    (out_dir / "captions_tdsc_tables.md").write_text(render_caption_package(frame), encoding="utf-8")

    print(f"Saved tables to: {out_dir}")


if __name__ == "__main__":
    main()
