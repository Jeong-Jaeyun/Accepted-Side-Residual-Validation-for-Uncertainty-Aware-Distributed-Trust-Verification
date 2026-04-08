from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from qbm.train import (
    apply_overrides,
    build_feature_statistics,
    build_node_ids,
    generate_evidence_batch,
    load_config,
    load_feature_windows,
)


def batch_to_records(batch, scenario: str) -> List[Dict]:
    records: List[Dict] = []
    for e in batch.evidences:
        records.append(
            {
                "scenario": scenario,
                "window_id": batch.window_id,
                "node_id": e.node_id,
                "event_id": e.event_id,
                "decision": e.decision,
                "anomaly_score": e.anomaly_score,
                "confidence": e.confidence,
                "uncertainty": e.uncertainty,
                "explanation_topk": "|".join(e.explanation_topk),
                "explanation_hash": e.explanation_hash,
                "context_hash": e.context_hash,
                "timestamp_ms": e.timestamp_ms,
                "prev_evidence_hash": e.prev_evidence_hash,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate node-local evidence preview.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to yaml config.")
    parser.add_argument("--scenario", default=None, help="Override scenario S0|S1|S2|S3")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--windows", type=int, default=5, help="Number of windows to preview")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, scenario=args.scenario, seed=args.seed)
    scenario = str(cfg.get("experiments", {}).get("scenario", "S3")).upper()
    seed = int(cfg.get("project", {}).get("seed", 42))

    features = load_feature_windows(cfg).head(max(1, int(args.windows))).copy()
    stats = build_feature_statistics(features, float(cfg.get("inference", {}).get("train_fraction", 0.70)))
    node_ids = build_node_ids(cfg)
    prev_hash_by_node: Dict[str, str] = {}
    rng = np.random.default_rng(seed)

    records: List[Dict] = []
    for step, (_, row) in enumerate(features.iterrows()):
        batch = generate_evidence_batch(
            row,
            step_index=step,
            cfg=cfg,
            stats=stats,
            node_ids=node_ids,
            prev_hash_by_node=prev_hash_by_node,
            rng=rng,
        )
        records.extend(batch_to_records(batch, scenario))

    out = pd.DataFrame(records)
    out_path = Path(cfg.get("project", {}).get("results_dir", "results")) / "tables" / "evidence_preview.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.head(20).to_string(index=False))
    print(f"\nSaved evidence preview: {out_path}")


if __name__ == "__main__":
    main()
