from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(round(float(value)))


def rate_str(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def threshold_str(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def latency_str(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.1f}"


def signed_one_decimal(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.1f}"


def signed_four_decimal(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.4f}"


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    return value


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(name, "") for name in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    *,
    caption: str,
    label: str,
    table_env: str = "table*",
    colspec: str | None = None,
    small: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if colspec is None:
        first = "l"
        rest = "r" * max(0, len(fieldnames) - 1)
        colspec = first + rest
    body_lines = []
    for row in rows:
        cells = [latex_escape(row.get(name, "")) for name in fieldnames]
        body_lines.append(" & ".join(cells) + r" \\")
    lines = [
        rf"\begin{{{table_env}}}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]
    if small:
        lines.append(r"\small")
    lines.extend(
        [
            r"\setlength{\tabcolsep}{4.5pt}",
            rf"\begin{{tabular}}{{{colspec}}}",
            r"\toprule",
            " & ".join(latex_escape(name) for name in fieldnames) + r" \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            rf"\end{{{table_env}}}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(path: Path, main_dir: Path, appendix_dir: Path) -> None:
    text = f"""# Tables

Canonical paper tables are organized into two folders:

- `paper_main`: four tables intended for the main paper body
- `paper_appendix`: three tables intended for appendix/robustness sections

Files are emitted as `csv`, `md`, and `tex`.

Main tables:
- `{main_dir.name}/table1_qiskit_main_operating_point.*`
- `{main_dir.name}/table2_qiskit_operating_tradeoff.*`
- `{main_dir.name}/table3_cross_port_main.*`
- `{main_dir.name}/table4_cross_port_efficiency.*`

Appendix tables:
- `{appendix_dir.name}/tableA1_qiskit_backend_compare.*`
- `{appendix_dir.name}/tableA2_qiskit_noise_robustness.*`
- `{appendix_dir.name}/tableA3_qiskit_shot_sensitivity.*`

Raw experiment outputs remain in `results/tables` and per-port `results/<port>/tables`.
`IEEE/tables` contains the mirrored `tex` files used by the manuscript.
"""
    path.write_text(text, encoding="utf-8")


def build_qiskit_main_table(source: Path) -> tuple[list[str], list[dict[str, str]]]:
    rows = load_rows(source)
    fieldnames = [
        "Method",
        "A0 TCP",
        "A4P FTR",
        "A5 FTR",
        "QBM veto count",
        "Benign veto count",
        "Malicious veto count",
    ]
    out_rows: list[dict[str, str]] = []
    for row in rows:
        out_rows.append(
            {
                "Method": row["Method"],
                "A0 TCP": rate_str(parse_float(row["A0 TCP"])),
                "A4P FTR": rate_str(parse_float(row["A4P FTR"])),
                "A5 FTR": rate_str(parse_float(row["A5 FTR"])),
                "QBM veto count": str(parse_int(row["QBM veto count"]) or 0),
                "Benign veto count": str(parse_int(row["Benign veto count"]) or 0),
                "Malicious veto count": str(parse_int(row["Malicious veto count"]) or 0),
            }
        )
    return fieldnames, out_rows


def build_qiskit_tradeoff_table(source: Path) -> tuple[list[str], list[dict[str, str]]]:
    rows = load_rows(source)
    fieldnames = [
        "Quantile",
        "Threshold",
        "A0 TCP",
        "A4P FTR",
        "A5 FTR",
        "Benign veto count",
        "Malicious veto count",
    ]
    out_rows: list[dict[str, str]] = []
    for row in rows:
        out_rows.append(
            {
                "Quantile": rate_str(parse_float(row["Quantile"])),
                "Threshold": threshold_str(parse_float(row["Threshold"])),
                "A0 TCP": rate_str(parse_float(row["A0 TCP"])),
                "A4P FTR": rate_str(parse_float(row["A4P FTR"])),
                "A5 FTR": rate_str(parse_float(row["A5 FTR"])),
                "Benign veto count": str(parse_int(row["Benign veto count"]) or 0),
                "Malicious veto count": str(parse_int(row["Malicious veto count"]) or 0),
            }
        )
    return fieldnames, out_rows


def port_label(port_key: str) -> str:
    mapping = {
        "antwerp": "Antwerp",
        "cape_town": "Cape Town",
        "los_angeles": "Los Angeles",
        "singapore": "Singapore",
    }
    return mapping.get(port_key, port_key.replace("_", " ").title())


def load_summary_row(summary_rows: list[dict[str, str]], attack_id: str, verifier_name: str) -> dict[str, str]:
    for row in summary_rows:
        if row.get("attack_id") == attack_id and row.get("verifier_name") == verifier_name:
            return row
    raise KeyError(f"Missing row for attack_id={attack_id}, verifier_name={verifier_name}")


def build_cross_port_main_table(results_root: Path, ports: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames = [
        "Port",
        "Attack",
        "S3-MEV FTR",
        "StrongQ FTR",
        "Delta FTR",
        "S3-MEV ASR",
        "StrongQ ASR",
        "StrongQ Calls",
    ]
    rows: list[dict[str, str]] = []
    for port in ports:
        summary_path = results_root / port / "tables" / "summary_end2end_extended.csv"
        summary_rows = load_rows(summary_path)
        attack_verifiers = {
            "A4": ("s3_mev", "strongq_verifier"),
            "A5": ("s3_only", "s3_strongq"),
        }
        for attack_id, (base_verifier, strong_verifier) in attack_verifiers.items():
            mev = load_summary_row(summary_rows, attack_id, base_verifier)
            strong = load_summary_row(summary_rows, attack_id, strong_verifier)
            mev_ftr = parse_float(mev["ftr"])
            strong_ftr = parse_float(strong["ftr"])
            mev_asr = parse_float(mev["asr"])
            strong_asr = parse_float(strong["asr"])
            delta_ftr = None
            if mev_ftr is not None and strong_ftr is not None:
                delta_ftr = strong_ftr - mev_ftr
            rows.append(
                {
                    "Port": port_label(port),
                    "Attack": attack_id,
                    "S3-MEV FTR": rate_str(mev_ftr),
                    "StrongQ FTR": rate_str(strong_ftr),
                    "Delta FTR": signed_four_decimal(delta_ftr),
                    "S3-MEV ASR": rate_str(mev_asr),
                    "StrongQ ASR": rate_str(strong_asr),
                    "StrongQ Calls": str(parse_int(strong.get("n_strongq_called")) or 0),
                }
            )
    return fieldnames, rows


def build_cross_port_efficiency_table(
    results_root: Path,
    ports: list[str],
    *,
    cross_port_windows: int,
) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames = [
        "Port",
        "Attack",
        "Latency (MEV, ms)",
        "Latency (StrongQ, ms)",
        "Overhead (%)",
        "Calls Ratio (%)",
    ]
    rows: list[dict[str, str]] = []
    for port in ports:
        summary_path = results_root / port / "tables" / "summary_end2end_extended.csv"
        summary_rows = load_rows(summary_path)
        attack_verifiers = {
            "A0": ("s3_only", "s3_strongq"),
            "A5": ("s3_only", "s3_strongq"),
        }
        for attack_id, (base_verifier, strong_verifier) in attack_verifiers.items():
            base = load_summary_row(summary_rows, attack_id, base_verifier)
            strong = load_summary_row(summary_rows, attack_id, strong_verifier)
            base_latency = parse_float(base["latency_ms_mean"])
            strong_latency = parse_float(strong["latency_ms_mean"])
            overhead = None
            if base_latency and strong_latency is not None:
                overhead = ((strong_latency - base_latency) / base_latency) * 100.0
            calls_ratio = None
            calls = parse_int(strong.get("n_strongq_called"))
            if calls is not None and cross_port_windows > 0:
                calls_ratio = (calls / cross_port_windows) * 100.0
            rows.append(
                {
                    "Port": port_label(port),
                    "Attack": attack_id,
                    "Latency (MEV, ms)": latency_str(base_latency),
                    "Latency (StrongQ, ms)": latency_str(strong_latency),
                    "Overhead (%)": signed_one_decimal(overhead),
                    "Calls Ratio (%)": latency_str(calls_ratio),
                }
            )
    return fieldnames, rows


def build_backend_compare_table(source: Path) -> tuple[list[str], list[dict[str, str]]]:
    rows = load_rows(source)
    fieldnames = [
        "Backend",
        "Attack",
        "Threshold",
        "TCP",
        "FTR",
        "QBM veto count",
        "Benign veto count",
        "Malicious veto count",
    ]
    out_rows: list[dict[str, str]] = []
    for row in rows:
        out_rows.append(
            {
                "Backend": row["backend_mode"],
                "Attack": row["attack_id"],
                "Threshold": threshold_str(parse_float(row["qbm_threshold_final"])),
                "TCP": rate_str(parse_float(row["tcp"])),
                "FTR": rate_str(parse_float(row["ftr"])),
                "QBM veto count": str(parse_int(row["qbm_veto_count"]) or 0),
                "Benign veto count": str(parse_int(row["qbm_benign_veto_count"]) or 0),
                "Malicious veto count": str(parse_int(row["qbm_malicious_veto_count"]) or 0),
            }
        )
    return fieldnames, out_rows


def build_noise_table(source: Path) -> tuple[list[str], list[dict[str, str]]]:
    rows = load_rows(source)
    fieldnames = [
        "Noise level",
        "Attack",
        "Threshold",
        "TCP",
        "FTR",
        "QBM veto count",
        "Mean q-score",
        "Shot std mean",
    ]
    out_rows: list[dict[str, str]] = []
    for row in rows:
        out_rows.append(
            {
                "Noise level": rate_str(parse_float(row["noise_level"])),
                "Attack": row["attack_id"],
                "Threshold": threshold_str(parse_float(row["qbm_threshold_final"])),
                "TCP": rate_str(parse_float(row["tcp"])),
                "FTR": rate_str(parse_float(row["ftr"])),
                "QBM veto count": str(parse_int(row["qbm_veto_count"]) or 0),
                "Mean q-score": rate_str(parse_float(row["qbm_shadow_mean"])),
                "Shot std mean": rate_str(parse_float(row["qbm_shot_std_mean"])),
            }
        )
    return fieldnames, out_rows


def build_shot_table(source: Path) -> tuple[list[str], list[dict[str, str]]]:
    rows = load_rows(source)
    fieldnames = [
        "Shots",
        "Attack",
        "Threshold",
        "TCP",
        "FTR",
        "QBM veto count",
        "Mean q-score",
        "Shot std mean",
    ]
    out_rows: list[dict[str, str]] = []
    for row in rows:
        out_rows.append(
            {
                "Shots": str(parse_int(row["shots"]) or 0),
                "Attack": row["attack_id"],
                "Threshold": threshold_str(parse_float(row["qbm_threshold_final"])),
                "TCP": rate_str(parse_float(row["tcp"])),
                "FTR": rate_str(parse_float(row["ftr"])),
                "QBM veto count": str(parse_int(row["qbm_veto_count"]) or 0),
                "Mean q-score": rate_str(parse_float(row["qbm_shadow_mean"])),
                "Shot std mean": rate_str(parse_float(row["qbm_shot_std_mean"])),
            }
        )
    return fieldnames, out_rows


def write_table_triplet(
    out_base: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    *,
    caption: str,
    label: str,
    ieee_tex_path: Path | None = None,
) -> None:
    csv_path = out_base.with_suffix(".csv")
    md_path = out_base.with_suffix(".md")
    tex_path = out_base.with_suffix(".tex")
    write_csv(csv_path, fieldnames, rows)
    write_markdown(md_path, fieldnames, rows)
    write_latex(tex_path, fieldnames, rows, caption=caption, label=label)
    if ieee_tex_path is not None:
        ieee_tex_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tex_path, ieee_tex_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build organized paper tables for main text and appendix.")
    parser.add_argument("--tables-root", default="results/tables", help="Raw tables root.")
    parser.add_argument("--results-root", default="results", help="Results root.")
    parser.add_argument("--ieee-tables", default="IEEE/tables", help="IEEE tex output directory.")
    parser.add_argument(
        "--ports",
        default="antwerp,cape_town,los_angeles,singapore",
        help="Comma-separated cross-port result directories.",
    )
    parser.add_argument("--cross-port-windows", type=int, default=300, help="Window count for calls-ratio reporting.")
    args = parser.parse_args()

    tables_root = ROOT / args.tables_root
    results_root = ROOT / args.results_root
    ieee_tables = ROOT / args.ieee_tables
    ports = [part.strip() for part in args.ports.split(",") if part.strip()]

    main_dir = tables_root / "paper_main"
    appendix_dir = tables_root / "paper_appendix"
    main_dir.mkdir(parents=True, exist_ok=True)
    appendix_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = build_qiskit_main_table(tables_root / "qiskit_qbm_paper_main_table_300w.csv")
    write_table_triplet(
        main_dir / "table1_qiskit_main_operating_point",
        fieldnames,
        rows,
        caption="Main exact-state QBM operating-point comparison at 300 windows.",
        label="tab:qiskit_main_operating_point",
        ieee_tex_path=ieee_tables / "table3_qiskit_main_operating_point.tex",
    )

    fieldnames, rows = build_qiskit_tradeoff_table(tables_root / "qiskit_qbm_paper_tradeoff_table_300w.csv")
    write_table_triplet(
        main_dir / "table2_qiskit_operating_tradeoff",
        fieldnames,
        rows,
        caption="Operating-point trade-off across benign calibration quantiles for the exact-state QBM pipeline.",
        label="tab:qiskit_operating_tradeoff",
        ieee_tex_path=ieee_tables / "table4_qiskit_operating_tradeoff.tex",
    )

    fieldnames, rows = build_cross_port_main_table(results_root, ports)
    write_table_triplet(
        main_dir / "table3_cross_port_main",
        fieldnames,
        rows,
        caption="Cross-port verification performance under replay (A4) and node-compromise (A5) attacks. Negative Delta FTR indicates an improvement over S3-MEV.",
        label="tab:cross_port_main",
        ieee_tex_path=ieee_tables / "table1_cross_port_main.tex",
    )

    fieldnames, rows = build_cross_port_efficiency_table(
        results_root,
        ports,
        cross_port_windows=args.cross_port_windows,
    )
    write_table_triplet(
        main_dir / "table4_cross_port_efficiency",
        fieldnames,
        rows,
        caption="Cross-port latency overhead and StrongQ activation ratio under benign (A0) and node-compromise (A5) conditions.",
        label="tab:cross_port_efficiency",
        ieee_tex_path=ieee_tables / "table2_cross_port_efficiency.tex",
    )

    fieldnames, rows = build_backend_compare_table(tables_root / "qiskit_qbm_robustness_backend_compare.csv")
    write_table_triplet(
        appendix_dir / "tableA1_qiskit_backend_compare",
        fieldnames,
        rows,
        caption="Backend comparison for the QBM robustness appendix.",
        label="tab:qiskit_backend_compare",
        ieee_tex_path=ieee_tables / "tableA1_qiskit_backend_compare.tex",
    )

    fieldnames, rows = build_noise_table(tables_root / "qiskit_qbm_robustness_noise_sweep.csv")
    write_table_triplet(
        appendix_dir / "tableA2_qiskit_noise_robustness",
        fieldnames,
        rows,
        caption="Noise robustness summary for the shot-based QBM appendix.",
        label="tab:qiskit_noise_robustness",
        ieee_tex_path=ieee_tables / "tableA2_qiskit_noise_robustness.tex",
    )

    fieldnames, rows = build_shot_table(tables_root / "qiskit_qbm_robustness_shot_sensitivity.csv")
    write_table_triplet(
        appendix_dir / "tableA3_qiskit_shot_sensitivity",
        fieldnames,
        rows,
        caption="Shot sensitivity summary for the shot-based QBM appendix.",
        label="tab:qiskit_shot_sensitivity",
        ieee_tex_path=ieee_tables / "tableA3_qiskit_shot_sensitivity.tex",
    )

    write_readme(tables_root / "README.md", main_dir, appendix_dir)


if __name__ == "__main__":
    main()
