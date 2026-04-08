"""
End-to-end preprocessing pipeline for maritime AIS data -> (features, discrete, onehot)
- raw -> sanitize -> port filter -> windowing -> (optional injection) -> grid -> features -> discretize -> onehot
- saves processed parquet + artifacts (quantiles, t0)

Run:
  python experiments/run_pipeline.py --config configs/default.yaml --ports ports/ports.yaml
"""

from __future__ import annotations

import os
import json
import argparse
import copy
import yaml
import pandas as pd

from preprocess.load_and_map_schema import load_and_prepare
from preprocess.filter_by_port import filter_by_port
from preprocess.windowing import add_windows
from preprocess.grid_mapping import add_grid_indices
from preprocess.feature_extraction import compute_features
from preprocess.discretize_quantiles import fit_quantiles, apply_quantiles
from preprocess.encode_onehot import onehot_levels

            
from experiments.inject_spoofing import (
    inject_s2_position_jump,
    inject_s1_identity_flood,
    inject_s3_hybrid,
)


def _deep_merge(dst: dict, src: dict) -> dict:
    out = dict(dst)
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict:
    default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    with open(default_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    target_path = os.path.abspath(path)
    if os.path.normcase(target_path) == os.path.normcase(default_path):
        return base_cfg
    with open(target_path, "r", encoding="utf-8") as f:
        override_cfg = yaml.safe_load(f) or {}
    return _deep_merge(base_cfg, override_cfg)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_windows_table(df: pd.DataFrame, dt_minutes: int) -> pd.DataFrame:
    """
    Build window metadata table:
      window_id, ts_start, ts_end (exclusive boundary: ts_start + dt)
    """
    if not {"window_id", "ts"}.issubset(df.columns):
        raise KeyError("df must contain 'window_id' and 'ts'")

    dt = pd.Timedelta(minutes=int(dt_minutes))
    g = df.groupby("window_id", sort=True)["ts"]
    windows = g.agg(ts_start="min").reset_index()
    windows["ts_end"] = windows["ts_start"] + dt
    return windows


def select_normal_features_for_fit(
    feats: pd.DataFrame,
    windows: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Select feature rows used to fit quantiles based on cfg['discretization'].

    Supports:
      - fit_on: "clean" (default)  -> use all feats
      - fit_on: "explicit_range"   -> use time range [start_ts, end_ts] on windows table
    """
    disc_cfg = cfg.get("discretization", {})
    fit_on = disc_cfg.get("fit_on", "clean")

    feats2 = feats.merge(windows, on="window_id", how="left")

    if fit_on == "explicit_range":
        nr = disc_cfg.get("normal_range", {}) or {}
        start_ts = nr.get("start_ts")
        end_ts = nr.get("end_ts")
        if not start_ts or not end_ts:
            raise ValueError("discretization.fit_on='explicit_range' requires normal_range.start_ts and end_ts")

        start = pd.to_datetime(start_ts, utc=True)
        end = pd.to_datetime(end_ts, utc=True)

        feats_fit = feats2[(feats2["ts_start"] >= start) & (feats2["ts_end"] <= end)].copy()
        if feats_fit.empty:
            raise ValueError("No windows selected for explicit normal_range. Check start_ts/end_ts or data coverage.")
        return feats_fit.drop(columns=["ts_start", "ts_end"])

                              
    return feats


def main(cfg_path: str = "configs/default.yaml", ports_path: str = "ports/ports.yaml"):
    cfg = load_config(cfg_path)

                       
    proc_dir = cfg["project"]["processed_dir"]
    art_dir = cfg["project"].get("artifacts_dir", "artifacts")
    ensure_dir(proc_dir)
    ensure_dir(art_dir)

                                 
    df = load_and_prepare(
        csv_path=cfg["project"]["raw_path"],
        mapping_dict=cfg["schema_mapping"],
        tz_hint=cfg["time"].get("timezone", "UTC"),
    )

                         
    df = filter_by_port(
        df,
        port_name=cfg["project"]["port"],
        ports_path=ports_path,
        use_polygon=cfg["port_filter"].get("use_polygon", False),
        bbox_override=cfg["port_filter"].get("bbox_override"),
    )

                                                       
    df, t0_used = add_windows(df, cfg["time"]["dt_minutes"], cfg["time"].get("t0"))

                               
    ports_cfg = load_config(ports_path)
    port_entry = ports_cfg.get(cfg["project"]["port"], {})
    bbox_vals = cfg["port_filter"].get("bbox_override") or port_entry.get("bbox")
    if not bbox_vals or len(bbox_vals) != 4:
        raise ValueError("Port bbox must be provided either in ports.yaml or via port_filter.bbox_override")

                                                    
    exp_cfg = cfg.get("experiments", {})
    if exp_cfg.get("enable_injection", False):
        scenario = exp_cfg.get("scenario", "S0")
        seed = int(cfg["project"].get("seed", 42))

        if scenario == "S1":
            s1 = exp_cfg.get("S1", {})
            iw = exp_cfg.get("injection_window", {"start_window": 0, "end_window": 0})
            df = inject_s1_identity_flood(
                df,
                intensity=float(s1.get("intensity", 0.3)),
                message_multiplier=float(s1.get("message_multiplier", 1.5)),
                start_window=int(iw.get("start_window", 0)),
                end_window=int(iw.get("end_window", 0)),
                bbox_vals=bbox_vals,
                target_zone=str(s1.get("target_zone", "hotspot")),
                seed=seed,
            )

        elif scenario == "S2":
            s2 = exp_cfg.get("S2", {})
                                                                                             
            df = inject_s2_position_jump(
                df,
                intensity=float(s2.get("intensity", 0.2)),
                jump_distance_km=float(s2.get("jump_distance_km", 30.0)),
                seed=seed,
                max_dt_seconds=int(s2.get("max_dt_seconds", 3600)),
            )

        elif scenario == "S3":
            s3 = exp_cfg.get("S3", {})
            iw = exp_cfg.get("injection_window", {"start_window": 0, "end_window": 0})

            df = inject_s3_hybrid(
                df,
                intensity=float(s3.get("intensity", 0.25)),
                message_multiplier=float(s3.get("message_multiplier", 2.0)),
                jump_distance_km=float(s3.get("jump_distance_km", 30.0)),
                affect_existing_fraction=float(s3.get("affect_existing_fraction", 0.1)),
                start_window=int(iw.get("start_window", 0)),
                end_window=int(iw.get("end_window", 0)),
                bbox_vals=bbox_vals,
                target_zone=str(s3.get("target_zone", "hotspot")),
                seed=seed,
                max_dt_seconds=int(s3.get("max_dt_seconds", 3600)),
            )
        else:
            print(f"[Injection] scenario={scenario} -> no injection applied")

                          
    df = add_grid_indices(
        df,
        bbox_vals,
        cfg["grid"]["nx"],
        cfg["grid"]["ny"],
        clamp=True,
    )

                                              
    windows = build_windows_table(df, cfg["time"]["dt_minutes"])

                                
    feats = compute_features(df, cfg)

                                                         
    disc_cfg = cfg.get("discretization", {})
    q_low = float(disc_cfg.get("q_low", 0.33))
    q_high = float(disc_cfg.get("q_high", 0.66))

    feats_fit = select_normal_features_for_fit(feats, windows, cfg)
    thresholds = fit_quantiles(feats_fit, q_low, q_high)
    feats_disc = apply_quantiles(feats, thresholds)

                     
    enc_cfg = cfg.get("encoding", {})
    feature_order = enc_cfg.get("feature_order", [])
    if not feature_order:
        raise ValueError("encoding.feature_order is empty. Provide the discrete feature column order for one-hot.")
    onehot = onehot_levels(feats_disc, feature_order)

                                 
    windows.to_parquet(os.path.join(proc_dir, "windows.parquet"), index=False)
    feats.to_parquet(os.path.join(proc_dir, "features.parquet"), index=False)
    feats_disc.to_parquet(os.path.join(proc_dir, "features_discrete.parquet"), index=False)
    onehot.to_parquet(os.path.join(proc_dir, "onehot.parquet"), index=False)

                         
    disc_dir = os.path.join(art_dir, "discretization")
    ensure_dir(disc_dir)

    with open(os.path.join(disc_dir, "quantiles.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

    with open(os.path.join(disc_dir, "t0.txt"), "w", encoding="utf-8") as f:
        f.write(str(t0_used))

                    
    print("Pipeline complete.")
    print(f"t0 used: {t0_used}")
    print(f"Rows after filter: {len(df)}")
    print(f"Windows: {int(feats['window_id'].nunique())}")
    if "F4_position_jump_rate" in feats.columns:
        nz = int((feats["F4_position_jump_rate"] > 0).sum())
        print(f"F4>0 windows: {nz}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ports", type=str, default="ports/ports.yaml")
    args = parser.parse_args()
    main(cfg_path=args.config, ports_path=args.ports)
