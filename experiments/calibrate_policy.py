from __future__ import annotations
import os
import yaml
import pandas as pd

def main():
    s0_dir = "data/processed/busan/s0"
    feats = pd.read_parquet(os.path.join(s0_dir, "features.parquet"))

                                
    f2 = "F2_new_mmsi_rate"
    f3 = "F3_message_burstiness"
    f4 = "F4_position_jump_rate"

                                    
    q_f2 = float(feats[f2].quantile(0.999))
    q_f3 = float(feats[f3].quantile(0.999))

                                                
                                           
    q_f4 = float(feats[f4].quantile(0.999))

    policy = {
        "meta": {
            "source": "auto-calibrated from S0",
            "s0_dir": s0_dir,
            "quantiles": {"F2": 0.995, "F3": 0.995, "F4": 0.999},
        },
        "thresholds": {
            "F2_new_mmsi_rate": q_f2,
            "F3_message_burstiness": q_f3,
            "F4_position_jump_rate": max(q_f4, 0.0),
        },
        "rules": [
            {
                "id": "R_S1_ID_FLOOD",
                "if": {"any": [
                    {"feature": f2, "op": ">", "threshold_key": "F2_new_mmsi_rate"},
                    {"feature": f3, "op": ">", "threshold_key": "F3_message_burstiness"},
                ]},
                "then": ["throttle_admission", "quarantine_new_mmsi"],
                "severity": 2,
                "explain": "S1-like identity/message flood",
            },
            {
                "id": "R_S2_POS_JUMP",
                "if": {"all": [
                    {"feature": f4, "op": ">", "value": 0.0}
                ]},
                "then": ["isolate_suspicious_mmsi", "pq_key_rotation_event"],
                "severity": 3,
                "explain": "S2-like physically implausible movement",
            },
            {
                "id": "R_S3_HYBRID",
                "if": {"all": [
                    {"rule": "R_S1_ID_FLOOD"},
                    {"rule": "R_S2_POS_JUMP"},
                ]},
                "then": ["throttle_admission", "isolate_suspicious_mmsi", "quarantine_new_mmsi"],
                "severity": 4,
                "explain": "Hybrid: flood + spoofing",
                "priority": 100,
            },
        ],
        "action_effects": {
            "throttle_admission": {"admission_rate_mult": 0.6},
            "quarantine_new_mmsi": {"drop_new_mmsi_mult": 0.8},
            "isolate_suspicious_mmsi": {"drop_suspicious_mult": 0.5},
            "pq_key_rotation_event": {"consensus_overhead_mult": 1.1},
        }
    }

    os.makedirs("policy", exist_ok=True)
    with open("policy/policy_table.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(policy, f, sort_keys=False, allow_unicode=True)

    print("Saved policy/policy_table.yaml")
    print("Thresholds:", policy["thresholds"])

if __name__ == "__main__":
    main()
