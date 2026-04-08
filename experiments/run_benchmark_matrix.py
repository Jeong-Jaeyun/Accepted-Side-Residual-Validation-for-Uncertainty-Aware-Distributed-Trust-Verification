"""
Run reproducible benchmark matrix for AIS trust-layer experiments.

Primary matrix:
  scenarios (S0,S1,S2,S3) x attacks from config.benchmark.attacks
  (default: A0..A5) using each scenario's default verifier

Optional extra matrix:
  S3 x attacks from config.benchmark.attacks x verifiers (s3_mev,qbm,strongq)

Outputs:
  - per-run sim csv/json via qbm.train.save_outputs
  - consolidated table: results/tables/benchmark_matrix.csv
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from qbm.train import apply_overrides, load_config, run_simulation, save_outputs


def _default_matrix(cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    bench = cfg.get("benchmark", {})
    scenarios = list(bench.get("scenarios", ["S0", "S1", "S2", "S3"]))
    attacks = list(bench.get("attacks", ["A0", "A1", "A2", "A3", "A4", "A5"]))
    verifier_map = dict(
        bench.get(
            "default_verifier",
            {"S0": "none", "S1": "none", "S2": "s2_strict", "S3": "s3_mev"},
        )
    )

    runs: List[Dict[str, str]] = []
    for scenario in scenarios:
        for attack in attacks:
            runs.append(
                {
                    "scenario": str(scenario).upper(),
                    "attack_id": str(attack).upper(),
                    "verifier": str(verifier_map.get(str(scenario).upper(), "none")),
                    "matrix": "primary",
                }
            )
    return runs


def _s3_ablation_runs(cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    bench = cfg.get("benchmark", {})
    attacks = list(bench.get("attacks", ["A0", "A1", "A2", "A3", "A4", "A5"]))
    verifiers = list(bench.get("s3_verifier_ablation", ["s3_mev", "qbm", "strongq"]))
    runs: List[Dict[str, str]] = []
    for verifier in verifiers:
        for attack in attacks:
            runs.append(
                {
                    "scenario": "S3",
                    "attack_id": str(attack).upper(),
                    "verifier": str(verifier),
                    "matrix": "s3_ablation",
                }
            )
    return runs


def _save_s1_replay_artifacts(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> None:
    s1 = df[(df["scenario"] == "S1") & (df["attack_id"].isin(["A3", "A4"]))].copy()
    if s1.empty:
        return

    keep_cols = [
        "scenario",
        "attack_id",
        "verifier_name",
        "asr",
        "ftr",
        "tcp",
        "ttd_windows",
        "dropped_by_verification_sum",
        "dropped_by_network_sum",
        "dropped_by_overflow_sum",
    ]
    s1 = s1[keep_cols].sort_values(["attack_id", "verifier_name"]).reset_index(drop=True)
    s1["s1_definition"] = "logging_only_no_verification"
    s1["replay_vulnerable_flag"] = (s1["ftr"] > 0).astype(int)
    out_path = tables_dir / "s1_replay_vulnerability.csv"
    s1.to_csv(out_path, index=False)
    print(f"Saved S1 replay table: {out_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skip S1 replay figure (matplotlib unavailable): {exc}")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "s1_replay_ftr.png"
    plt.figure(figsize=(6.4, 4.0))
    plt.bar(s1["attack_id"], s1["ftr"])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Attack")
    plt.ylabel("FTR")
    plt.title("S1 Replay Vulnerability (Logging Only, No Verification)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved S1 replay figure: {fig_path}")


def _save_a4_verifier_contrast_artifacts(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> None:
    target = df[(df["scenario"] == "S3") & (df["attack_id"] == "A4")].copy()
    if target.empty:
        return

    target = target[target["verifier_name"].isin(["s3_mev", "qbm_verifier", "strongq_verifier"])]
    if target.empty:
        return

    target = target.sort_values(["verifier_name", "matrix"]).drop_duplicates(["verifier_name"], keep="last")
    keep_cols = [
        "scenario",
        "attack_id",
        "verifier_name",
        "matrix",
        "asr",
        "ftr",
        "tcp",
        "ttd_windows",
        "processed_tps_mean",
        "latency_ms_mean",
        "dropped_by_verification_sum",
        "dropped_by_network_sum",
    ]
    target = target[keep_cols].reset_index(drop=True)

    base_row = target[target["verifier_name"] == "s3_mev"]
    if not base_row.empty:
        base_asr = float(base_row.iloc[0]["asr"])
        base_ftr = float(base_row.iloc[0]["ftr"])
        target["asr_reduction_vs_s3_mev"] = base_asr - target["asr"]
        target["ftr_reduction_vs_s3_mev"] = base_ftr - target["ftr"]
    else:
        target["asr_reduction_vs_s3_mev"] = float("nan")
        target["ftr_reduction_vs_s3_mev"] = float("nan")

    out_path = tables_dir / "a4_verifier_contrast.csv"
    target.to_csv(out_path, index=False)
    print(f"Saved A4 verifier contrast table: {out_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skip A4 contrast figure (matplotlib unavailable): {exc}")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "a4_verifier_ftr_asr.png"
    ordered = target.copy()
    ordered["verifier_name"] = pd.Categorical(
        ordered["verifier_name"],
        categories=["s3_mev", "qbm_verifier", "strongq_verifier"],
        ordered=True,
    )
    ordered = ordered.sort_values("verifier_name")

    plt.figure(figsize=(7.2, 4.0))
    x = range(len(ordered))
    width = 0.36
    x_left = [v - width / 2.0 for v in x]
    x_right = [v + width / 2.0 for v in x]
    plt.bar(x_left, ordered["asr"], width=width, label="ASR")
    plt.bar(x_right, ordered["ftr"], width=width, label="FTR")
    plt.xticks(list(x), ordered["verifier_name"])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Verifier (S3, A4)")
    plt.ylabel("Rate")
    plt.title("A4 Time-Shifted Replay: Classical S3 vs QBM/StrongQ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved A4 verifier contrast figure: {fig_path}")


def _save_analysis_artifacts(df: pd.DataFrame, results_dir: Path) -> None:
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    _save_s1_replay_artifacts(df, tables_dir, figures_dir)
    _save_a4_verifier_contrast_artifacts(df, tables_dir, figures_dir)


def run_matrix(
    cfg_path: str,
    *,
    max_windows: int | None,
    seed: int | None,
    include_s3_ablation: bool,
) -> pd.DataFrame:
    base_cfg = load_config(cfg_path)
    runs = _default_matrix(base_cfg)
    if include_s3_ablation:
        runs.extend(_s3_ablation_runs(base_cfg))

    rows: List[Dict[str, Any]] = []
    total = len(runs)
    for idx, run in enumerate(runs, 1):
        scenario = run["scenario"]
        attack_id = run["attack_id"]
        verifier = run["verifier"]

        cfg = load_config(cfg_path)
        cfg = apply_overrides(
            cfg,
            scenario=scenario,
            attack_id=attack_id,
            verifier_impl=verifier,
            seed=seed,
        )
        cfg["experiments"]["enable_injection"] = attack_id != "A0"
        cfg["experiments"]["attack_id"] = attack_id
        if attack_id != "A0" and max_windows is not None:
            iw = cfg["experiments"].setdefault("injection_window", {})
            start_w = int(iw.get("start_window", 0))
            end_w = int(iw.get("end_window", 0))
            if start_w >= int(max_windows) or end_w <= start_w:
                new_start = max(5, int(max_windows * 0.35))
                new_end = max(new_start + 1, int(max_windows * 0.75))
                iw["start_window"] = new_start
                iw["end_window"] = new_end

        sim_df, summary = run_simulation(cfg, max_windows=max_windows)
        paths = save_outputs(cfg, sim_df, summary)
        rows.append(
            {
                "run_idx": idx,
                "run_total": total,
                "matrix": run["matrix"],
                "scenario_req": scenario,
                "attack_req": attack_id,
                "verifier_req": verifier,
                "s1_logging_only": bool(summary.scenario == "S1" and summary.verifier_name == "none"),
                **asdict(summary),
                **paths,
            }
        )
        print(
            f"[{idx:02d}/{total}] matrix={run['matrix']} scenario={scenario} "
            f"attack={attack_id} verifier={verifier} "
            f"-> ASR={summary.asr:.4f} FTR={summary.ftr:.4f} TCP={summary.tcp:.4f}"
        )

    out = pd.DataFrame(rows)
    results_dir = Path(base_cfg.get("project", {}).get("results_dir", "results"))
    out_path = results_dir / "tables" / "benchmark_matrix.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    _save_analysis_artifacts(out, results_dir)
    print(f"Saved matrix summary: {out_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S0~S3 benchmark matrix over config-defined attacks.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to configuration yaml.")
    parser.add_argument("--max-windows", type=int, default=None, help="Run first N windows for each run.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--include-s3-ablation",
        action="store_true",
        help="Also run S3 x config attacks for verifiers (s3_mev,qbm,strongq).",
    )
    args = parser.parse_args()

    run_matrix(
        cfg_path=args.config,
        max_windows=args.max_windows,
        seed=args.seed,
        include_s3_ablation=bool(args.include_s3_ablation),
    )


if __name__ == "__main__":
    main()
