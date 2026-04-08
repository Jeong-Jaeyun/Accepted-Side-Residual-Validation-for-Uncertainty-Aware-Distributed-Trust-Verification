                             
from __future__ import annotations
import math
import pandas as pd


def _safe(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def simulate_raft(
    feats_with_policy: pd.DataFrame,
    policy_cfg: dict,
                             
    base_capacity_tps: float = 180.0,
    base_offered_per_window: float = 150.0,
    base_latency_ms: float = 120.0,
) -> pd.DataFrame:
    """
    Per-window discrete simulation (HCIS MVP):
    - offered load is driven by F2 (new IDs) and F3 (burstiness), WITHOUT saturation.
    - consensus overhead is driven by F4 and PQ rotation actions.
    - policy actions affect admission + drop rates.
    - backlog increases latency smoothly.
    """
    effects = policy_cfg.get("action_effects", {})
    backlog = 0.0

    rows = []
    for _, r in feats_with_policy.iterrows():
        f2 = _safe(r.get("F2_new_mmsi_rate", 0.0))
        f3 = _safe(r.get("F3_message_burstiness", 0.0))
        f4 = _safe(r.get("F4_position_jump_rate", 0.0))

                                                  
                                               
        offered = base_offered_per_window * (
            1.0
            + 0.8 * math.log1p(max(0.0, f3))               
            + 1.2 * max(0.0, f2)                           
        )

                                                                   
        overhead_mult = 1.0 + 0.6 * min(1.0, max(0.0, f4))                      

                            
        acts = r.get("policy_actions", []) or []
        admission_mult = 1.0
        drop_share = 0.0
        for a in acts:
            eff = effects.get(a, {})
            admission_mult *= float(eff.get("admission_rate_mult", 1.0))
            overhead_mult *= float(eff.get("consensus_overhead_mult", 1.0))

                                    
            if "drop_new_mmsi_mult" in eff:
                drop_share = max(drop_share, 1.0 - float(eff["drop_new_mmsi_mult"]))
            if "drop_suspicious_mult" in eff:
                drop_share = max(drop_share, 1.0 - float(eff["drop_suspicious_mult"]))

                                     
        capacity = base_capacity_tps / max(1e-6, overhead_mult)

                              
        accepted = offered * admission_mult
        dropped = accepted * drop_share
        admitted = max(0.0, accepted - dropped)

                               
        processed = min(capacity, admitted + backlog)
        backlog = max(0.0, backlog + admitted - processed)

                                                    
        latency = base_latency_ms * (1.0 + 0.45 * math.log1p(backlog / max(1.0, base_offered_per_window)))

        rows.append({
            "window_id": int(r["window_id"]),
            "offered": offered,
            "admitted": admitted,
            "processed_tps": processed,
            "backlog": backlog,
            "latency_ms": latency,
            "dropped": dropped,
            "policy_fired": bool(r.get("policy_fired", False)),
            "overhead_mult": overhead_mult,
        })

    return pd.DataFrame(rows)