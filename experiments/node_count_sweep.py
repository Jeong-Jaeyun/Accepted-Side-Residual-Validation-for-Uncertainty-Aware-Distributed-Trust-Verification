"""
Node-count sweep to validate non-3-node special-case behavior.

Default matrix:
  node_count in {3,5}
  scenarios in {S0,S2,S3}
  attacks in {A2,A5}
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from qbm.train import apply_overrides, load_config, run_simulation


VERIFIER_BY_SCENARIO = {
    "S0": "none",
    "S2": "s2_strict",
    "S3": "s3_mev",
}


def _parse_list(spec: str, *, upper: bool = False) -> List[str]:
    out: List[str] = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(token.upper() if upper else token)
    return out


def _parse_int_list(spec: str) -> List[int]:
    out: List[int] = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(max(1, int(token)))
    return out


def _adjust_injection_window(cfg: Dict[str, Any], max_windows: int | None) -> None:
    if max_windows is None:
        return
    attack_id = str(cfg.get("experiments", {}).get("attack_id", "A0")).upper()
    if attack_id == "A0":
        return
    iw = cfg["experiments"].setdefault("injection_window", {})
    start_w = int(iw.get("start_window", 0))
    end_w = int(iw.get("end_window", 0))
    if start_w >= int(max_windows) or end_w <= start_w:
        new_start = max(5, int(max_windows * 0.35))
        new_end = max(new_start + 1, int(max_windows * 0.75))
        iw["start_window"] = new_start
        iw["end_window"] = new_end


def run_node_count_sweep(
    *,
    cfg_path: str,
    node_counts: List[int],
    scenarios: List[str],
    attacks: List[str],
    max_windows: int | None,
    seed: int | None,
) -> pd.DataFrame:
    base_cfg = load_config(cfg_path)
    results_dir = Path(base_cfg.get("project", {}).get("results_dir", "results"))
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    total = len(node_counts) * len(scenarios) * len(attacks)
    idx = 0
    for node_count in node_counts:
        for scenario in scenarios:
            verifier = VERIFIER_BY_SCENARIO.get(scenario, "none")
            for attack_id in attacks:
                idx += 1
                cfg = load_config(cfg_path)
                cfg = apply_overrides(
                    cfg,
                    scenario=scenario,
                    attack_id=attack_id,
                    verifier_impl=verifier,
                    seed=seed,
                )
                cfg["blockchain_net"]["validators"] = int(node_count)
                cfg["experiments"]["enable_injection"] = attack_id != "A0"
                cfg["experiments"]["attack_id"] = attack_id
                _adjust_injection_window(cfg, max_windows)

                sim_df, summary = run_simulation(cfg, max_windows=max_windows)
                epsilon = float(cfg.get("verification", {}).get("corr_threshold", float("nan")))
                theta = float(cfg.get("verification", {}).get("explanation_threshold", float("nan")))
                row = {
                    "run_idx": idx,
                    "run_total": total,
                    "node_count": int(node_count),
                    "scenario": scenario,
                    "attack_id": attack_id,
                    "verifier_req": verifier,
                    "epsilon": epsilon,
                    "theta": theta,
                    **asdict(summary),
                }
                rows.append(row)

                meta_path = tables_dir / f"meta_node_count_{scenario}_{attack_id}_n{node_count}.json"
                with meta_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "scenario": scenario,
                            "attack_id": attack_id,
                            "verifier": verifier,
                            "reproducibility": {
                                "epsilon_corr_threshold": epsilon,
                                "theta_explanation_threshold": theta,
                                "node_count": int(node_count),
                            },
                            "summary": asdict(summary),
                            "max_windows": max_windows,
                            "seed": seed,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

                print(
                    f"[{idx:02d}/{total}] n={node_count} {scenario}-{attack_id} "
                    f"ASR={summary.asr:.4f} FTR={summary.ftr:.4f} TCP={summary.tcp:.4f}"
                )

    out = pd.DataFrame(rows).sort_values(["node_count", "scenario", "attack_id"]).reset_index(drop=True)
    out_path = tables_dir / "node_count_sweep.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved node-count table: {out_path}")

    try:
        import matplotlib.pyplot as plt

        figures_dir.mkdir(parents=True, exist_ok=True)
        pivot_cols = ["scenario", "attack_id"]
        group_keys = out[pivot_cols].drop_duplicates().copy()
        group_order = list(group_keys.apply(tuple, axis=1))
        labels = [f"{s}-{a}" for s, a in group_order]
        node_values = sorted(out["node_count"].unique())
        x = list(range(len(labels)))

        def _metric_values(metric: str, node_count: int) -> List[float]:
            return (
                out[out["node_count"] == node_count]
                .set_index(pivot_cols)[metric]
                .reindex(group_order)
                .to_numpy()
                .tolist()
            )

        metrics = [
            ("asr", "ASR", (0.0, 1.0), "node_count_sweep_asr.png"),
            ("ftr", "FTR", (0.0, 1.0), "node_count_sweep_ftr.png"),
            (
                "dropped_by_verification_sum",
                "Dropped by Verification",
                None,
                "node_count_sweep_dropped_by_verification.png",
            ),
        ]

        for metric, title, ylim, filename in metrics:
            plt.figure(figsize=(8.0, 4.0))
            width = 0.75 / max(len(node_values), 1)
            center = (len(node_values) - 1) / 2.0
            for j, n in enumerate(node_values):
                offset = (j - center) * width
                vals = _metric_values(metric, n)
                plt.bar([v + offset for v in x], vals, width=width, label=f"n={n}")
            plt.xticks(x, labels, rotation=25, ha="right")
            if ylim is not None:
                plt.ylim(*ylim)
            plt.xlabel("Scenario-Attack")
            plt.ylabel(title)
            plt.title(f"Node Count Sweep: {title}")
            if len(node_values) > 1:
                plt.legend()
            plt.tight_layout()
            out_fig = figures_dir / filename
            plt.savefig(out_fig, dpi=200)
            plt.close()
            print(f"Saved node-count figure: {out_fig}")
    except Exception as exc:  # pragma: no cover
        print(f"Skip node-count figure (matplotlib unavailable): {exc}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run node-count sweep for S0/S2/S3 and A2/A5.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to configuration yaml.")
    parser.add_argument("--node-counts", default="5", help="Comma-separated node counts (default: 5).")
    parser.add_argument("--scenarios", default="S0,S2,S3", help="Comma-separated scenarios (default: S0,S2,S3).")
    parser.add_argument("--attacks", default="A2,A5", help="Comma-separated attacks (default: A2,A5).")
    parser.add_argument("--max-windows", type=int, default=None, help="Run only first N windows.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    args = parser.parse_args()

    run_node_count_sweep(
        cfg_path=args.config,
        node_counts=_parse_int_list(args.node_counts) or [3, 5],
        scenarios=_parse_list(args.scenarios, upper=True) or ["S0", "S2", "S3"],
        attacks=_parse_list(args.attacks, upper=True) or ["A2", "A5"],
        max_windows=args.max_windows,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
