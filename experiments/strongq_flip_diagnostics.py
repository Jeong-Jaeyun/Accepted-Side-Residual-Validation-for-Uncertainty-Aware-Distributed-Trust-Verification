"""
Generate StrongQ flip diagnostics table from per-window simulation outputs.

This is intentionally a small, deterministic post-processing step so the paper
can cite a single CSV artifact with:
  - gray-zone entry rate
  - StrongQ call rate
  - flip counts (MEV -> veto/override)
  - FTR/ASR/latency (from meta summary)
  - decision_path histogram (from sim csv)

Expected inputs (produced by `python -m qbm.train ...`):
  results/tables/sim_<SCENARIO>_<ATTACK>_<VERIFIER>.csv
  results/tables/meta_<SCENARIO>_<ATTACK>_<VERIFIER>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


_VERIFIER_CANONICAL = {
    # CLI aliases -> output verifier_name used in filenames.
    "s3_mev": "s3_mev",
    "qbm": "qbm_verifier",
    "qbm_verifier": "qbm_verifier",
    "strongq": "strongq_verifier",
    "strongq_verifier": "strongq_verifier",
    "none": "none",
}


def _canon_verifier(name: str) -> str:
    key = str(name).strip()
    return _VERIFIER_CANONICAL.get(key, key)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _decision_hist(sim_df: pd.DataFrame) -> Dict[str, int]:
    if "decision_path" not in sim_df.columns:
        return {}
    series = sim_df["decision_path"].astype(str)
    # Keep it robust even if earlier runs had missing decision_path.
    series = series.fillna("unknown")
    vc = series.value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.to_dict().items()}


def build_rows(
    *,
    tables_dir: Path,
    scenario: str,
    attacks: List[str],
    verifiers: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    scenario_u = str(scenario).upper()
    for attack in attacks:
        attack_u = str(attack).upper()
        for verifier in verifiers:
            verifier_c = _canon_verifier(verifier)
            sim_path = tables_dir / f"sim_{scenario_u}_{attack_u}_{verifier_c}.csv"
            meta_path = tables_dir / f"meta_{scenario_u}_{attack_u}_{verifier_c}.json"
            if not sim_path.exists():
                raise FileNotFoundError(str(sim_path))
            sim_df = pd.read_csv(sim_path)
            meta = _load_json(meta_path) or {}
            summary = dict(meta.get("summary", {}))

            n_windows = int(len(sim_df))
            dec_hist = _decision_hist(sim_df)

            # Aggregate attempts/commits for paper-friendly sanity checks.
            ft_attempted = int(pd.to_numeric(sim_df.get("false_trust_candidate", 0), errors="coerce").fillna(0.0).sum())
            ft_committed = int(pd.to_numeric(sim_df.get("false_trust_committed", 0), errors="coerce").fillna(0.0).sum())
            mal_injected = int(pd.to_numeric(sim_df.get("malicious_injected", 0), errors="coerce").fillna(0.0).sum())
            mal_committed = int(pd.to_numeric(sim_df.get("malicious_committed", 0), errors="coerce").fillna(0.0).sum())

            # Rates: prefer meta counters if available, else compute from sim.
            def _meta_int(k: str) -> int:
                try:
                    return int(summary.get(k, 0))
                except Exception:
                    return 0

            n_gray_zone = _meta_int("n_gray_zone")
            n_strongq_called = _meta_int("n_strongq_called")
            if n_gray_zone <= 0 and "gray_zone_flag" in sim_df.columns:
                n_gray_zone = int((pd.to_numeric(sim_df["gray_zone_flag"], errors="coerce").fillna(0.0) > 0.5).sum())
            if n_strongq_called <= 0 and "strongq_called" in sim_df.columns:
                n_strongq_called = int((pd.to_numeric(sim_df["strongq_called"], errors="coerce").fillna(0.0) > 0.5).sum())

            gray_rate = (n_gray_zone / n_windows) if n_windows > 0 else float("nan")
            called_rate = (n_strongq_called / n_windows) if n_windows > 0 else float("nan")

            row = {
                "scenario": scenario_u,
                "attack_id": attack_u,
                "verifier": str(summary.get("verifier_name", verifier_c)),
                "n_windows": n_windows,
                "gray_rate": float(gray_rate),
                "strongq_called_rate": float(called_rate),
                "n_gray_zone": int(n_gray_zone),
                "n_strongq_called": int(n_strongq_called),
                "n_flip_mev_to_reject": int(_meta_int("n_flip_mev_to_reject")),
                "n_flip_mev_to_accept": int(_meta_int("n_flip_mev_to_accept")),
                "n_strongq_agree": int(_meta_int("n_strongq_agree")),
                "n_strongq_disagree": int(_meta_int("n_strongq_disagree")),
                "false_trust_attempted": int(ft_attempted),
                "false_trust_committed": int(ft_committed),
                "malicious_injected": int(mal_injected),
                "malicious_committed": int(mal_committed),
                "asr": float(summary.get("asr", float("nan"))),
                "ftr": float(summary.get("ftr", float("nan"))),
                "processed_tps_mean": float(summary.get("processed_tps_mean", float("nan"))),
                "latency_ms_mean": float(summary.get("latency_ms_mean", float("nan"))),
                "decision_counts_json": json.dumps(dec_hist, ensure_ascii=False, sort_keys=True),
                "sim_csv": str(sim_path).replace("\\", "/"),
                "meta_json": str(meta_path).replace("\\", "/") if meta_path.exists() else "",
            }

            # Keep a small subset of "reproducibility" knobs visible in the CSV for paper appendix.
            repro = dict(meta.get("reproducibility", {})) if isinstance(meta.get("reproducibility", {}), dict) else {}
            for k in (
                "node_count",
                "epsilon_corr_threshold",
                "theta_explanation_threshold",
                "s3_gray_margin_min",
                "s3_risk_policy_ratio_floor",
                "s3_risk_context_floor",
                "s3_risk_explanation_floor",
                "s3_risk_tau_band",
            ):
                if k in repro:
                    row[k] = repro.get(k)
            rows.append(row)
    return rows


def _pick(df: pd.DataFrame, *, scenario: str, attack_id: str, verifier_name: str) -> Dict[str, Any]:
    sub = df[
        (df["scenario"].astype(str).str.upper() == str(scenario).upper())
        & (df["attack_id"].astype(str).str.upper() == str(attack_id).upper())
        & (df["verifier"].astype(str) == str(verifier_name))
    ].copy()
    if sub.empty:
        raise KeyError(f"Missing row: scenario={scenario} attack={attack_id} verifier={verifier_name}")
    # If duplicates exist, keep the last (most recently appended) row.
    row = sub.iloc[-1].to_dict()
    return row


def to_wide(df_long: pd.DataFrame, *, mev_verifier: str, strongq_verifier: str) -> pd.DataFrame:
    """
    Convert long-format rows (attack_id x verifier) into one-row-per-attack wide table.

    Required columns for the paper:
      - strongq_called_rate
      - n_flip_mev_to_reject
      - FTR_mev
      - FTR_strongq
    """
    scenario = str(df_long["scenario"].iloc[0]) if not df_long.empty else "S3"
    attacks = sorted({str(a).upper() for a in df_long["attack_id"].astype(str).tolist()})

    wide_rows: List[Dict[str, Any]] = []
    for attack_id in attacks:
        mev = _pick(df_long, scenario=scenario, attack_id=attack_id, verifier_name=mev_verifier)
        sq = _pick(df_long, scenario=scenario, attack_id=attack_id, verifier_name=strongq_verifier)

        row: Dict[str, Any] = {
            "scenario": scenario,
            "attack_id": attack_id,
            "n_windows": int(mev.get("n_windows", sq.get("n_windows", 0))),
            # Gray-zone entry is independent of StrongQ decision; keep both for sanity checks.
            "gray_rate_mev": float(mev.get("gray_rate", float("nan"))),
            "gray_rate_strongq": float(sq.get("gray_rate", float("nan"))),
            # StrongQ gating behavior (cost proxy).
            "strongq_called_rate": float(sq.get("strongq_called_rate", float("nan"))),
            "n_strongq_called": int(sq.get("n_strongq_called", 0)),
            "n_flip_mev_to_reject": int(sq.get("n_flip_mev_to_reject", 0)),
            "n_flip_mev_to_accept": int(sq.get("n_flip_mev_to_accept", 0)),
            # Core paper metrics.
            "ASR_mev": float(mev.get("asr", float("nan"))),
            "ASR_strongq": float(sq.get("asr", float("nan"))),
            "FTR_mev": float(mev.get("ftr", float("nan"))),
            "FTR_strongq": float(sq.get("ftr", float("nan"))),
            "latency_ms_mean_mev": float(mev.get("latency_ms_mean", float("nan"))),
            "latency_ms_mean_strongq": float(sq.get("latency_ms_mean", float("nan"))),
            "processed_tps_mean_mev": float(mev.get("processed_tps_mean", float("nan"))),
            "processed_tps_mean_strongq": float(sq.get("processed_tps_mean", float("nan"))),
            "decision_counts_mev_json": str(mev.get("decision_counts_json", "")),
            "decision_counts_strongq_json": str(sq.get("decision_counts_json", "")),
            "sim_csv_mev": str(mev.get("sim_csv", "")),
            "sim_csv_strongq": str(sq.get("sim_csv", "")),
            "meta_json_mev": str(mev.get("meta_json", "")),
            "meta_json_strongq": str(sq.get("meta_json", "")),
        }

        # Carry reproducibility knobs (should match across both runs).
        for k in (
            "node_count",
            "epsilon_corr_threshold",
            "theta_explanation_threshold",
            "s3_gray_margin_min",
            "s3_risk_policy_ratio_floor",
            "s3_risk_context_floor",
            "s3_risk_explanation_floor",
            "s3_risk_tau_band",
            "p0",
            "c0",
            "s0",
            "band",
        ):
            if k in sq:
                row[k] = sq.get(k)
            elif k in mev:
                row[k] = mev.get(k)
        wide_rows.append(row)

    out = pd.DataFrame(wide_rows)
    out = out.sort_values(["attack_id"]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate results/tables/strongq_flip_diagnostics.csv")
    parser.add_argument("--config", default="configs/default.yaml", help="Config yaml (used only to locate results dir).")
    parser.add_argument("--scenario", default="S3", help="Scenario id (default: S3).")
    parser.add_argument("--attacks", default="A0,A4,A4P,A5", help="Comma-separated attack ids.")
    parser.add_argument(
        "--verifiers",
        default="s3_mev,strongq",
        help="Comma-separated verifier impls (kept for backwards compatibility; expects mev,strongq).",
    )
    args = parser.parse_args()

    # Resolve results dir from config, but keep script standalone if yaml changes.
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"PyYAML required: {exc}")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    results_dir = Path(cfg.get("project", {}).get("results_dir", "results"))
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    attacks = [a.strip() for a in str(args.attacks).split(",") if a.strip()]
    verifiers = [v.strip() for v in str(args.verifiers).split(",") if v.strip()]
    if len(verifiers) < 2:
        raise SystemExit("--verifiers must include at least 2 items: <mev>,<strongq>")
    mev_impl, strongq_impl = verifiers[0], verifiers[1]
    mev_name, strongq_name = _canon_verifier(mev_impl), _canon_verifier(strongq_impl)
    rows = build_rows(tables_dir=tables_dir, scenario=args.scenario, attacks=attacks, verifiers=verifiers)

    df_long = pd.DataFrame(rows)
    df_long = df_long.sort_values(["attack_id", "verifier"]).reset_index(drop=True)
    wide = to_wide(df_long, mev_verifier=mev_name, strongq_verifier=strongq_name)
    out_path = tables_dir / "strongq_flip_diagnostics.csv"
    wide.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Keep the long-format table for debugging / appendix-level audits.
    out_long = tables_dir / "strongq_flip_diagnostics_long.csv"
    df_long.to_csv(out_long, index=False)
    print(f"Saved: {out_long}")


if __name__ == "__main__":
    main()
