from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


METHOD_ORDER = {
    "s3_only": 0,
    "s3_strongq": 1,
    "s3_qbm_strongq": 2,
}

METHOD_LABELS = {
    "s3_only": "S3",
    "s3_strongq": "S3 + StrongQ",
    "s3_qbm_strongq": "S3 + QBM + StrongQ",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _fmt_prob(value: float) -> str:
    return f"{float(value):.4f}"


def _fmt_threshold(value: float) -> str:
    return f"{float(value):.6f}"


def _render_markdown_table(headers: List[str], rows: Iterable[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_main_table(pipeline_eval: pd.DataFrame, *, backend_mode: str) -> pd.DataFrame:
    df = pipeline_eval[pipeline_eval["backend_mode"].astype(str) == str(backend_mode)].copy()
    df = df[df["pipeline_mode"].astype(str).isin(["s3_only", "s3_strongq", "s3_qbm_strongq"])]
    if df.empty:
        raise ValueError(f"No rows found for backend_mode={backend_mode!r}")

    rows = []
    for pipeline_mode, group in df.groupby("pipeline_mode", sort=False):
        attack_map = {str(row["attack_id"]): row for _, row in group.iterrows()}
        rows.append(
            {
                "Method": METHOD_LABELS.get(str(pipeline_mode), str(pipeline_mode)),
                "A0 TCP": float(attack_map["A0"]["tcp"]) if "A0" in attack_map else float("nan"),
                "A4P FTR": float(attack_map["A4P"]["ftr"]) if "A4P" in attack_map else float("nan"),
                "A5 FTR": float(attack_map["A5"]["ftr"]) if "A5" in attack_map else float("nan"),
                "QBM veto count": int(group["qbm_veto_count"].fillna(0).sum()),
                "Benign veto count": int(group["qbm_benign_veto_count"].fillna(0).sum()),
                "Malicious veto count": int(group["qbm_malicious_veto_count"].fillna(0).sum()),
                "_method_order": METHOD_ORDER.get(str(pipeline_mode), 99),
            }
        )
    out = pd.DataFrame(rows).sort_values("_method_order", kind="mergesort").drop(columns=["_method_order"])
    return out.reset_index(drop=True)


def build_tradeoff_table(tradeoff_long: pd.DataFrame, *, backend_mode: str, sweep_mode: str) -> pd.DataFrame:
    df = tradeoff_long[
        (tradeoff_long["backend_mode"].astype(str) == str(backend_mode))
        & (tradeoff_long["sweep_mode"].astype(str) == str(sweep_mode))
    ].copy()
    df = df[df["attack_id"].astype(str).isin(["A0", "A4P", "A5"])]
    if df.empty:
        raise ValueError(f"No trade-off rows found for backend_mode={backend_mode!r}, sweep_mode={sweep_mode!r}")

    rows = []
    for sweep_value, group in df.groupby("sweep_value", sort=True):
        attack_map = {str(row["attack_id"]): row for _, row in group.iterrows()}
        rows.append(
            {
                "Quantile": float(sweep_value),
                "Threshold": float(group["qbm_threshold_final"].iloc[0]),
                "A0 TCP": float(attack_map["A0"]["tcp"]) if "A0" in attack_map else float("nan"),
                "A4P FTR": float(attack_map["A4P"]["ftr"]) if "A4P" in attack_map else float("nan"),
                "A5 FTR": float(attack_map["A5"]["ftr"]) if "A5" in attack_map else float("nan"),
                "Benign veto count": int(group["benign_veto_count"].fillna(0).sum()),
                "Malicious veto count": int(group["malicious_veto_count"].fillna(0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("Quantile", kind="mergesort").reset_index(drop=True)


def render_main_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                str(row["Method"]),
                _fmt_prob(row["A0 TCP"]),
                _fmt_prob(row["A4P FTR"]),
                _fmt_prob(row["A5 FTR"]),
                str(int(row["QBM veto count"])),
                str(int(row["Benign veto count"])),
                str(int(row["Malicious veto count"])),
            ]
        )
    return _render_markdown_table(headers, rows)


def render_tradeoff_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                f"{float(row['Quantile']):.2f}",
                _fmt_threshold(row["Threshold"]),
                _fmt_prob(row["A0 TCP"]),
                _fmt_prob(row["A4P FTR"]),
                _fmt_prob(row["A5 FTR"]),
                str(int(row["Benign veto count"])),
                str(int(row["Malicious veto count"])),
            ]
        )
    return _render_markdown_table(headers, rows)


def render_operating_point_note(tradeoff_df: pd.DataFrame, *, chosen_quantile: float) -> str:
    row = tradeoff_df.loc[(tradeoff_df["Quantile"] - float(chosen_quantile)).abs() < 1.0e-12]
    if row.empty:
        raise KeyError(f"Chosen quantile {chosen_quantile} not found in trade-off table.")
    selected = row.iloc[0]
    return (
        "The veto rule is `q_score < threshold`.\n\n"
        "The threshold is defined as a benign calibration quantile.\n\n"
        "lower quantile corresponds to stricter veto activation.\n\n"
        f"`q={float(chosen_quantile):.2f}` was selected as the main operating point because it preserved "
        f"`A0 TCP={float(selected['A0 TCP']):.4f}` while reducing "
        f"`A4P FTR={float(selected['A4P FTR']):.4f}` and `A5 FTR={float(selected['A5 FTR']):.4f}` "
        f"with `zero benign veto` in the evaluated set. "
        f"The corresponding threshold was `{float(selected['Threshold']):.6f}`, and the evaluated-set veto totals "
        f"were `benign={int(selected['Benign veto count'])}`, `malicious={int(selected['Malicious veto count'])}`.\n"
    )


def run(
    *,
    pipeline_eval_csv: Path,
    tradeoff_csv: Path,
    output_dir: Path,
    backend_mode: str,
    chosen_quantile: float,
    sweep_mode: str,
) -> None:
    pipeline_eval = _read_csv(pipeline_eval_csv)
    tradeoff_long = _read_csv(tradeoff_csv)

    main_table = build_main_table(pipeline_eval, backend_mode=backend_mode)
    tradeoff_table = build_tradeoff_table(tradeoff_long, backend_mode=backend_mode, sweep_mode=sweep_mode)
    note_text = render_operating_point_note(tradeoff_table, chosen_quantile=chosen_quantile)

    output_dir.mkdir(parents=True, exist_ok=True)
    main_csv = output_dir / "qiskit_qbm_main_paper_table.csv"
    main_md = output_dir / "qiskit_qbm_main_paper_table.md"
    tradeoff_csv_out = output_dir / "qiskit_qbm_tradeoff_paper_table.csv"
    tradeoff_md = output_dir / "qiskit_qbm_tradeoff_paper_table.md"
    note_md = output_dir / "qiskit_qbm_operating_point_note.md"

    main_table.to_csv(main_csv, index=False)
    main_md.write_text(render_main_markdown(main_table), encoding="utf-8")
    tradeoff_table.to_csv(tradeoff_csv_out, index=False)
    tradeoff_md.write_text(render_tradeoff_markdown(tradeoff_table), encoding="utf-8")
    note_md.write_text(note_text, encoding="utf-8")

    print(f"saved: {main_csv}")
    print(f"saved: {main_md}")
    print(f"saved: {tradeoff_csv_out}")
    print(f"saved: {tradeoff_md}")
    print(f"saved: {note_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-facing exact-state QBM tables and operating-point note.")
    parser.add_argument(
        "--pipeline-eval-csv",
        default="results/tables/qiskit_qbm_pipeline_eval.csv",
        help="Path to qiskit_qbm_pipeline_eval.csv",
    )
    parser.add_argument(
        "--tradeoff-csv",
        default="results/tables/qiskit_qbm_exact_tradeoff_table.csv",
        help="Path to qiskit_qbm_exact_tradeoff_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables",
        help="Directory to store paper-facing tables.",
    )
    parser.add_argument("--backend-mode", default="exact_state", help="Backend mode to keep in the paper tables.")
    parser.add_argument("--chosen-quantile", type=float, default=0.01, help="Chosen operating-point quantile.")
    parser.add_argument("--sweep-mode", default="quantile", help="Sweep mode to keep in trade-off table.")
    args = parser.parse_args()

    run(
        pipeline_eval_csv=_repo_root() / Path(str(args.pipeline_eval_csv)),
        tradeoff_csv=_repo_root() / Path(str(args.tradeoff_csv)),
        output_dir=_repo_root() / Path(str(args.output_dir)),
        backend_mode=str(args.backend_mode),
        chosen_quantile=float(args.chosen_quantile),
        sweep_mode=str(args.sweep_mode),
    )


if __name__ == "__main__":
    main()
