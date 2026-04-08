import numpy as np
import pandas as pd

def inject_s2_position_jump(
    df: pd.DataFrame,
    intensity: float,
    jump_distance_km: float,
    seed: int = 42,
    max_dt_seconds: int = 3600,                         
) -> pd.DataFrame:
    """
    S2: position-jump spoofing
    - 같은 MMSI 내부의 연속쌍(i, i+1) 중 dt가 짧은 쌍을 골라
    - i+1 포인트의 위치를 크게 이동시킨다.
    """
    rng = np.random.default_rng(seed)
    out = df.sort_values(["mmsi", "ts"]).copy()

                        
    counts = out["mmsi"].value_counts()
    valid_mmsi = counts[counts >= 2].index.to_numpy()

    n_select = max(1, int(len(valid_mmsi) * float(intensity)))
    chosen = rng.choice(valid_mmsi, size=n_select, replace=False)

    jump_deg = float(jump_distance_km) / 111.0                      

    modified = 0
    skipped_no_pair = 0

    for mmsi in chosen:
        sub = out[out["mmsi"] == mmsi]
        if len(sub) < 2:
            continue

                                   
        dt = sub["ts"].diff().dt.total_seconds().to_numpy()[1:]

               
        cand_pos = np.where((dt > 0) & (dt <= max_dt_seconds))[0]         
        if len(cand_pos) == 0:
            skipped_no_pair += 1
            continue

        i = int(rng.choice(cand_pos))
        idx_i = sub.index[i]
        idx_j = sub.index[i + 1]                   

                               
        out.at[idx_j, "lat"] = float(out.at[idx_j, "lat"]) + rng.choice([-1, 1]) * jump_deg
        modified += 1

    print(f"[S2] selected_mmsi={n_select}, modified_pairs={modified}, skipped_no_pair={skipped_no_pair}")
    return out

def _sample_point_in_bbox(rng, bbox):
                                                 
    min_lon, min_lat, max_lon, max_lat = map(float, bbox)
    lon = rng.uniform(min_lon, max_lon)
    lat = rng.uniform(min_lat, max_lat)
    return lon, lat


def _make_hotspot_bbox(bbox, shrink=0.2):
                        
    min_lon, min_lat, max_lon, max_lat = map(float, bbox)
    dx = (max_lon - min_lon)
    dy = (max_lat - min_lat)
    return [
        min_lon + dx * shrink,
        min_lat + dy * shrink,
        max_lon - dx * shrink,
        max_lat - dy * shrink,
    ]


def inject_s1_identity_flood(
    df: pd.DataFrame,
    intensity: float,
    message_multiplier: float,
    start_window: int,
    end_window: int,
    bbox_vals,
    target_zone: str = "hotspot",                     
    seed: int = 42,
) -> pd.DataFrame:
    """
    S1: identity flood (new MMSI injection + message burst)
    - Window 범위 [start_window, end_window] 내에서만 주입
    - 각 윈도우에서 기존 unique MMSI 수에 비례해 신규 MMSI를 생성
    - message_multiplier로 신규 MMSI 당 메시지 수를 늘림(버스트 유도)
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    if "window_id" not in out.columns:
        raise KeyError("S1 injection requires 'window_id' (run after windowing).")

                
    bbox = bbox_vals
    if target_zone == "hotspot":
        bbox = _make_hotspot_bbox(bbox_vals, shrink=0.2)

               
    wmask = (out["window_id"] >= int(start_window)) & (out["window_id"] <= int(end_window))
    target = out.loc[wmask].copy()
    if target.empty:
        print("[S1] target window range has no rows; skip injection.")
        return out

                            
    max_mmsi = int(pd.to_numeric(out["mmsi"], errors="coerce").fillna(0).max())
    next_mmsi = max_mmsi + 1

                                        
    msgs_per_new = max(1, int(round(float(message_multiplier))))

    injected_rows = []
    modified_windows = 0
    total_new_mmsi = 0
    total_new_msgs = 0

                      
    for win_id, grp in target.groupby("window_id", sort=True):
        base_unique = int(grp["mmsi"].nunique())
        n_new = max(1, int(round(base_unique * float(intensity))))
        modified_windows += 1
        total_new_mmsi += n_new

                                     
                                        
        for _ in range(n_new):
            new_id = next_mmsi
            next_mmsi += 1

                                    
            sample_idx = rng.choice(grp.index.to_numpy(), size=msgs_per_new, replace=True)
            samp = out.loc[sample_idx].copy()

            samp["mmsi"] = int(new_id)
                                             
                                                           
            lon, lat = _sample_point_in_bbox(rng, bbox)
            samp["lon"] = lon
            samp["lat"] = lat

            injected_rows.append(samp)
            total_new_msgs += len(samp)

    if injected_rows:
        inj = pd.concat(injected_rows, ignore_index=True)
        out = pd.concat([out, inj], ignore_index=True)

    print(
        f"[S1] windows={modified_windows}, new_mmsi={total_new_mmsi}, "
        f"new_msgs={total_new_msgs}, zone={target_zone}"
    )
    return out

def inject_s3_hybrid(
    df: pd.DataFrame,
    intensity: float,
    message_multiplier: float,
    jump_distance_km: float,
    affect_existing_fraction: float,
    start_window: int,
    end_window: int,
    bbox_vals,
    target_zone: str = "hotspot",
    seed: int = 42,
    max_dt_seconds: int = 3600,
) -> pd.DataFrame:
    """
    S3: hybrid attack = S1(identity flood + burst) + S2(position jump on existing identities)
    - Injection is limited to window_id in [start_window, end_window].
    - Part A (S1-like): inject new MMSIs within the window range with message bursts.
    - Part B (S2-like): apply position-jump on a fraction of EXISTING MMSIs, but only on pairs whose later point falls in the window range.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    if "window_id" not in out.columns:
        raise KeyError("S3 injection requires 'window_id' (run after windowing).")
    if not {"mmsi", "ts", "lat", "lon"}.issubset(out.columns):
        raise KeyError("df must contain mmsi, ts, lat, lon")

                                           
    def _sample_point_in_bbox(_bbox):
        min_lon, min_lat, max_lon, max_lat = map(float, _bbox)
        lon = rng.uniform(min_lon, max_lon)
        lat = rng.uniform(min_lat, max_lat)
        return lon, lat

    def _make_hotspot_bbox(_bbox, shrink=0.2):
        min_lon, min_lat, max_lon, max_lat = map(float, _bbox)
        dx = (max_lon - min_lon)
        dy = (max_lat - min_lat)
        return [
            min_lon + dx * shrink,
            min_lat + dy * shrink,
            max_lon - dx * shrink,
            max_lat - dy * shrink,
        ]

                                       
    w0 = int(start_window)
    w1 = int(end_window)
    wmask = (out["window_id"] >= w0) & (out["window_id"] <= w1)
    target = out.loc[wmask].copy()

    if target.empty:
        print("[S3] target window range has no rows; skip injection.")
        return out

                                     
    bbox = bbox_vals
    if target_zone == "hotspot":
        bbox = _make_hotspot_bbox(bbox_vals, shrink=0.2)

                                                                  
                                                                      
                                                                  
    max_mmsi = int(pd.to_numeric(out["mmsi"], errors="coerce").fillna(0).max())
    next_mmsi = max_mmsi + 1

                                                        
    msgs_per_new = max(1, int(round(float(message_multiplier))))

    injected_rows = []
    modified_windows = 0
    total_new_mmsi = 0
    total_new_msgs = 0

    for win_id, grp in target.groupby("window_id", sort=True):
        base_unique = int(grp["mmsi"].nunique())
        n_new = max(1, int(round(base_unique * float(intensity))))
        modified_windows += 1
        total_new_mmsi += n_new

        for _ in range(n_new):
            new_id = next_mmsi
            next_mmsi += 1

            sample_idx = rng.choice(grp.index.to_numpy(), size=msgs_per_new, replace=True)
            samp = out.loc[sample_idx].copy()

            samp["mmsi"] = int(new_id)
            lon, lat = _sample_point_in_bbox(bbox)
            samp["lon"] = lon
            samp["lat"] = lat

            injected_rows.append(samp)
            total_new_msgs += len(samp)

    if injected_rows:
        inj = pd.concat(injected_rows, ignore_index=True)
        out = pd.concat([out, inj], ignore_index=True)

                                                                  
                                                       
                                                                                  
                                                                  
    out = out.sort_values(["mmsi", "ts"]).copy()

                                                                   
    exist_mmsi = target["mmsi"].value_counts().index.to_numpy()
    n_affect = max(1, int(len(exist_mmsi) * float(affect_existing_fraction)))
    chosen = rng.choice(exist_mmsi, size=n_affect, replace=False)

    jump_deg = float(jump_distance_km) / 111.0

    modified_pairs = 0
    skipped_no_pair = 0

    for mmsi in chosen:
        sub = out[out["mmsi"] == mmsi]
        if len(sub) < 2:
            continue

                                                        
        dt = sub["ts"].diff().dt.total_seconds().to_numpy()[1:]

        win_next = sub["window_id"].to_numpy()[1:]                    
        cand_pos = np.where((dt > 0) & (dt <= max_dt_seconds) & (win_next >= w0) & (win_next <= w1))[0]

        if len(cand_pos) == 0:
            skipped_no_pair += 1
            continue

        i = int(rng.choice(cand_pos))
        idx_j = sub.index[i + 1]         

        out.at[idx_j, "lat"] = float(out.at[idx_j, "lat"]) + rng.choice([-1, 1]) * jump_deg
        modified_pairs += 1

    print(
        f"[S3] windows={modified_windows}, new_mmsi={total_new_mmsi}, new_msgs={total_new_msgs}, "
        f"affect_exist_mmsi={n_affect}, modified_pairs={modified_pairs}, skipped_no_pair={skipped_no_pair}, zone={target_zone}"
    )
    return out
