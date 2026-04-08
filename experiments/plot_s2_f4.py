                           
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_f4(processed_dir: str) -> np.ndarray:
    feats = pd.read_parquet(os.path.join(processed_dir, "features.parquet"))
    return feats["F4_position_jump_rate"].to_numpy()


def ecdf(x: np.ndarray):
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def main():
    base = os.path.join("data", "processed", "busan")
    runs = [
        ("S2-10km", os.path.join(base, "s2_jump_10km")),
        ("S2-30km", os.path.join(base, "s2_jump_30km")),
        ("S2-80km", os.path.join(base, "s2_jump_80km")),
    ]

    os.makedirs("results/figures", exist_ok=True)

                  
    plt.figure()
    for label, pdir in runs:
        f4 = load_f4(pdir)
        plt.hist(f4, bins=50, alpha=0.4, label=label)          
    plt.xlabel("F4_position_jump_rate")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/f4_hist_s2_sweep.png", dpi=200)
    plt.close()

                   
    plt.figure()
    for label, pdir in runs:
        f4 = load_f4(pdir)
        x, y = ecdf(f4)
        plt.plot(x, y, label=label)
    plt.xlabel("F4_position_jump_rate")
    plt.ylabel("ECDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/f4_ecdf_s2_sweep.png", dpi=200)
    plt.close()

                                                  
    labels = []
    ratios = []
    for label, pdir in runs:
        f4 = load_f4(pdir)
        ratio = float((f4 > 0).mean())
        labels.append(label)
        ratios.append(ratio)

    plt.figure()
    plt.bar(labels, ratios)
    plt.xlabel("Scenario")
    plt.ylabel("Fraction(F4>0)")
    plt.tight_layout()
    plt.savefig("results/figures/f4_pos_ratio_s2_sweep.png", dpi=200)
    plt.close()

    print("Saved figures to results/figures/")


if __name__ == "__main__":
    main()
