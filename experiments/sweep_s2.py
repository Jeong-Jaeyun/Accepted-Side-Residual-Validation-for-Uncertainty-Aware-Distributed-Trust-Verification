                         
from __future__ import annotations

import os
import copy
import yaml
import pandas as pd

from experiments.run_pipeline import main as run_pipeline_main


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(cfg: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def collect_f4_stats(processed_dir: str) -> dict:
    feats_path = os.path.join(processed_dir, "features.parquet")
    feats = pd.read_parquet(feats_path)
    f4 = feats["F4_position_jump_rate"]

    total_windows = int(feats["window_id"].nunique())
    f4_pos_windows = int((f4 > 0).sum())
    ratio = f4_pos_windows / total_windows if total_windows > 0 else 0.0

    return {
        "processed_dir": processed_dir,
        "total_windows": total_windows,
        "f4_pos_windows": f4_pos_windows,
        "f4_pos_ratio": ratio,
        "f4_mean": float(f4.mean()),
        "f4_max": float(f4.max()),
    }


def main():
    base_cfg_path = "configs/default.yaml"
    ports_path = "ports/ports.yaml"

    base_cfg = load_yaml(base_cfg_path)

                   
    distances = [10.0, 30.0, 80.0]

    out_rows = []
    for d_km in distances:
        cfg = copy.deepcopy(base_cfg)

                
        cfg["experiments"]["enable_injection"] = True
        cfg["experiments"]["scenario"] = "S2"
        cfg["experiments"]["S2"]["jump_distance_km"] = float(d_km)

                            
        suffix = f"s2_jump_{int(d_km)}km"
        cfg["project"]["processed_dir"] = os.path.join("data", "processed", cfg["project"]["port"], suffix)
        cfg["project"]["results_dir"] = os.path.join("results", suffix)
        cfg["project"]["artifacts_dir"] = os.path.join("artifacts", suffix)

                                    
        tmp_cfg_path = os.path.join("configs", "experiments", f"{suffix}.yaml")
        save_yaml(cfg, tmp_cfg_path)

        print(f"\n=== RUN: {suffix} ===")
        run_pipeline_main(cfg_path=tmp_cfg_path, ports_path=ports_path)

                  
        out_rows.append(collect_f4_stats(cfg["project"]["processed_dir"]))

              
    df = pd.DataFrame(out_rows)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/s2_sweep_summary.csv", index=False)
    print("\nSaved: results/tables/s2_sweep_summary.csv")
    print(df)


if __name__ == "__main__":
    main()
