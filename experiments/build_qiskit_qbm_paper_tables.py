from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


DEFAULT_METHODS = ["s3_only", "s3_qbm_strongq"]
ATTACK_ORDER = {"A0": 0, "A4P": 1, "A5": 2}
METHOD_LABELS = {
    "s3_only": "S3 Only",
    "s3_qbm_strongq": "S3 + QBM + StrongQ",
    "s3_strongq": "S3 + StrongQ",
    "s3_qbm": "S3 + QBM",
}


def _results_dir() -> Path:
    return Path("results") / "tables"


def _render_markdown_table(headers: List[str], rows: Iterable[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _fmt_prob(value: float) -> str:
    return f"{float(value):.4f}"


def _fmt_int(value: float) -> str:
    return str(int(round(float(value))))


def _fetch_metric(df: pd.DataFrame, *, method: str, attack_id: str, column: str) -> float:
    rows = df[(df["pipeline_mode"] == method) & (df["attack_id"] == attack_id)]
    if rows.empty:
        raise KeyError(f"Missing pipeline row for method={method}, attack={attack_id}")
    return float(rows.iloc[0][column])


def build_main_table(pipeline_df: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
        method_df = pipeline_df[pipeline_df["pipeline_mode"] == method]
        if method_df.empty:
            continue
        rows.append(
            {
                "Method": METHOD_LABELS.get(method, method),
                "method_id": method,
                "A0 TCP": _fetch_metric(pipeline_df, method=method, attack_id="A0", column="tcp"),
                "A4P FTR": _fetch_metric(pipeline_df, method=method, attack_id="A4P", column="ftr"),
                "A5 FTR": _fetch_metric(pipeline_df, method=method, attack_id="A5", column="ftr"),
                "QBM veto count": int(method_df["qbm_veto_count"].fillna(0.0).sum()),
                "Benign veto count": int(method_df["qbm_benign_veto_count"].fillna(0.0).sum()),
                "Malicious veto count": int(method_df["qbm_malicious_veto_count"].fillna(0.0).sum()),
                "QBM threshold": float(method_df["qbm_threshold_final"].dropna().iloc[0]) if method_df["qbm_threshold_final"].notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_tradeoff_table(tradeoff_df: pd.DataFrame) -> pd.DataFrame:
    exact = tradeoff_df[
        (tradeoff_df["backend_mode"] == "exact_state")
        & (tradeoff_df["pipeline_mode"] == "s3_qbm_strongq")
        & (tradeoff_df["sweep_mode"] == "quantile")
    ].copy()
    if exact.empty:
        return pd.DataFrame()

    rows = []
    for sweep_value, group in exact.groupby("sweep_value", sort=True):
        def _metric(attack_id: str, column: str) -> float:
            attack_rows = group[group["attack_id"] == attack_id]
            if attack_rows.empty:
                raise KeyError(f"Missing tradeoff row for quantile={sweep_value}, attack={attack_id}")
            return float(attack_rows.iloc[0][column])

        rows.append(
            {
                "Quantile": float(sweep_value),
                "Threshold": _metric("A0", "qbm_threshold_final"),
                "A0 TCP": _metric("A0", "tcp"),
                "A4P FTR": _metric("A4P", "ftr"),
                "A5 FTR": _metric("A5", "ftr"),
                "Benign veto count": int(group["benign_veto_count"].fillna(0.0).sum()),
                "Malicious veto count": int(group["malicious_veto_count"].fillna(0.0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("Quantile").reset_index(drop=True)


def build_operating_point_note(tradeoff_table: pd.DataFrame, *, selected_quantile: float, windows: int) -> str:
    selected = tradeoff_table.loc[(tradeoff_table["Quantile"] - float(selected_quantile)).abs() < 1.0e-12]
    if selected.empty:
        raise KeyError(f"Selected quantile {selected_quantile} not found in tradeoff table.")
    row = selected.iloc[0]
    benign_veto = int(row["Benign veto count"])
    a0_tcp = float(row["A0 TCP"])
    a4p_ftr = float(row["A4P FTR"])
    a5_ftr = float(row["A5 FTR"])
    best_a0_tcp = float(tradeoff_table["A0 TCP"].min())
    best_a4p_ftr = float(tradeoff_table["A4P FTR"].min())
    a0_phrase = (
        f"it gave the lowest `A0 TCP={a0_tcp:.4f}` among the evaluated quantiles"
        if abs(a0_tcp - best_a0_tcp) < 1.0e-12
        else f"`A0 TCP={a0_tcp:.4f}`"
    )
    a4p_phrase = (
        f"while matching the best observed `A4P FTR={a4p_ftr:.4f}`"
        if abs(a4p_ftr - best_a4p_ftr) < 1.0e-12
        else f"with `A4P FTR={a4p_ftr:.4f}`"
    )
    return "\n".join(
        [
            "# QBM Operating Point",
            "",
            "Veto rule: `q_score < threshold`.",
            "Threshold is defined as the benign calibration quantile over accepted-only A0 windows.",
            "Lower quantile corresponds to stricter veto activation.",
            (
                f"`q={selected_quantile:.2f}` was selected as the main operating point because {a0_phrase} "
                f"{a4p_phrase}; `A5 FTR={a5_ftr:.4f}` remained marginally improved, with "
                f"`benign_veto_count={benign_veto}` and "
                f"`malicious_veto_count={int(row['Malicious veto count'])}` in the evaluated `{windows}`-window setting."
            ),
            "",
            "Appendix positioning: `aer_shot` remains a limitation case; the main paper tables use `exact_state` only.",
        ]
    ) + "\n"


def write_outputs(
    *,
    main_table: pd.DataFrame,
    tradeoff_table: pd.DataFrame,
    note_text: str,
    label: str,
) -> None:
    out_dir = _results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{label}" if label else ""
    main_csv = out_dir / f"qiskit_qbm_paper_main_table{suffix}.csv"
    tradeoff_csv = out_dir / f"qiskit_qbm_paper_tradeoff_table{suffix}.csv"
    main_md = out_dir / f"qiskit_qbm_paper_main_table{suffix}.md"
    tradeoff_md = out_dir / f"qiskit_qbm_paper_tradeoff_table{suffix}.md"
    note_md = out_dir / f"qiskit_qbm_operating_point_note{suffix}.md"

    main_table.to_csv(main_csv, index=False)
    tradeoff_table.to_csv(tradeoff_csv, index=False)

    main_md.write_text(
        _render_markdown_table(
            ["Method", "A0 TCP", "A4P FTR", "A5 FTR", "QBM veto count", "Benign veto count", "Malicious veto count"],
            [
                [
                    str(row["Method"]),
                    _fmt_prob(row["A0 TCP"]),
                    _fmt_prob(row["A4P FTR"]),
                    _fmt_prob(row["A5 FTR"]),
                    _fmt_int(row["QBM veto count"]),
                    _fmt_int(row["Benign veto count"]),
                    _fmt_int(row["Malicious veto count"]),
                ]
                for _, row in main_table.iterrows()
            ],
        ),
        encoding="utf-8",
    )
    tradeoff_md.write_text(
        _render_markdown_table(
            ["Quantile", "Threshold", "A0 TCP", "A4P FTR", "A5 FTR", "Benign veto count", "Malicious veto count"],
            [
                [
                    f"{float(row['Quantile']):.2f}",
                    _fmt_prob(row["Threshold"]),
                    _fmt_prob(row["A0 TCP"]),
                    _fmt_prob(row["A4P FTR"]),
                    _fmt_prob(row["A5 FTR"]),
                    _fmt_int(row["Benign veto count"]),
                    _fmt_int(row["Malicious veto count"]),
                ]
                for _, row in tradeoff_table.iterrows()
            ],
        ),
        encoding="utf-8",
    )
    note_md.write_text(note_text, encoding="utf-8")

    print(f"saved: {main_csv}")
    print(f"saved: {tradeoff_csv}")
    print(f"saved: {main_md}")
    print(f"saved: {tradeoff_md}")
    print(f"saved: {note_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-facing QBM main/tradeoff tables and locked operating-point text.")
    parser.add_argument("--pipeline-eval", default="results/tables/qiskit_qbm_pipeline_eval.csv", help="Pipeline eval CSV path.")
    parser.add_argument("--tradeoff", default="results/tables/qiskit_qbm_exact_tradeoff_table.csv", help="Tradeoff CSV path.")
    parser.add_argument("--selected-quantile", type=float, default=0.01, help="Selected main operating-point quantile.")
    parser.add_argument("--windows", type=int, default=300, help="Window count used for the locked paper run.")
    parser.add_argument("--label", default="300w", help="Output suffix label.")
    parser.add_argument(
        "--methods",
        default="s3_only,s3_qbm_strongq",
        help="Comma-separated methods to keep in the main paper table.",
    )
    args = parser.parse_args()

    pipeline_df = pd.read_csv(args.pipeline_eval)
    pipeline_df = pipeline_df[pipeline_df["backend_mode"] == "exact_state"].copy()
    tradeoff_df = pd.read_csv(args.tradeoff)
    methods = [item.strip() for item in str(args.methods).split(",") if item.strip()]

    main_table = build_main_table(pipeline_df, methods)
    tradeoff_table = build_tradeoff_table(tradeoff_df)
    note_text = build_operating_point_note(
        tradeoff_table,
        selected_quantile=float(args.selected_quantile),
        windows=int(args.windows),
    )
    write_outputs(
        main_table=main_table,
        tradeoff_table=tradeoff_table,
        note_text=note_text,
        label=str(args.label).strip(),
    )


if __name__ == "__main__":
    main()
