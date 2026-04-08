from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def summarize_sim_csv(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Simulation file is empty: {path}")

    def _num(col: str, default: float = 0.0) -> pd.Series:
        return pd.to_numeric(df.get(col, pd.Series([default] * len(df))), errors="coerce").fillna(default)

    malicious_series = pd.to_numeric(df.get("malicious_injected", pd.Series([0] * len(df))), errors="coerce").fillna(0.0)
    malicious_committed = pd.to_numeric(df.get("malicious_committed", pd.Series([0] * len(df))), errors="coerce").fillna(0.0)
    false_trust_candidate = pd.to_numeric(df.get("false_trust_candidate", pd.Series([0] * len(df))), errors="coerce").fillna(0.0)
    false_trust_committed = pd.to_numeric(df.get("false_trust_committed", pd.Series([0] * len(df))), errors="coerce").fillna(0.0)
    commit_series = df.get("commit", pd.Series([True] * len(df)))
    if commit_series.dtype == object:
        commit_series = commit_series.astype(str).str.lower().isin({"1", "true", "yes", "y"})
    else:
        commit_series = commit_series.astype(bool)

    policy_series = df.get("policy_fired", pd.Series([False] * len(df)))
    if policy_series.dtype == object:
        policy_series = policy_series.astype(str).str.lower().isin({"1", "true", "yes", "y"})
    else:
        policy_series = policy_series.astype(bool)

    malicious_total = int(malicious_series.sum())
    malicious_accepted = int(malicious_committed.sum())
    false_total = int(false_trust_candidate.sum())
    false_committed = int(false_trust_committed.sum())
    first_mal = df.index[malicious_series > 0]
    first_det = df.index[(malicious_series > 0) & (~commit_series)]

    ttd = float("nan")
    if len(first_mal) > 0 and len(first_det) > 0:
        ttd = float(max(0, int(first_det[0]) - int(first_mal[0])))

    out = {
        "processed_tps_mean": float(_num("processed_tps").mean()),
        "latency_ms_mean": float(_num("latency_ms").mean()),
        "backlog_max": float(_num("backlog").max()),
        "dropped_sum": float(_num("dropped").sum()),
        "dropped_by_verification_sum": float(_num("dropped_by_verification").sum()),
        "dropped_by_network_sum": float(_num("dropped_by_network").sum()),
        "dropped_by_overflow_sum": float(_num("dropped_by_overflow").sum()),
        "policy_fired_ratio": float(policy_series.mean()),
        "asr": float(malicious_accepted / malicious_total) if malicious_total > 0 else 0.0,
        "ftr": float(false_committed / false_total) if false_total > 0 else 0.0,
        "ttd_windows": ttd,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize simulation CSV outputs.")
    parser.add_argument(
        "--sim",
        default=None,
        help="Path to one simulation csv. If omitted, summarize all results/tables/sim_*.csv",
    )
    args = parser.parse_args()

    paths: List[Path]
    if args.sim:
        paths = [Path(args.sim)]
    else:
        paths = sorted(Path("results/tables").glob("sim_*.csv"))

    if not paths:
        raise FileNotFoundError("No simulation csv found. Run qbm.train first.")

    rows = []
    for p in paths:
        stats = summarize_sim_csv(p)
        rows.append({"file": str(p), **stats})

    out = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    print(out.to_string(index=False))

    summary_path = Path("results/tables/score_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
