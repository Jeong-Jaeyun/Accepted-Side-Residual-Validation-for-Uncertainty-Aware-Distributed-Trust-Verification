from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from qbm.train import apply_overrides, load_config, run_simulation, save_outputs


ATTACK_ORDER = {"A0": 0, "A4P": 1, "A5": 2}


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df.get(column, pd.Series(dtype=float)), errors="coerce")


def _numeric_mean(df: pd.DataFrame, column: str) -> float:
    series = _numeric_series(df, column).dropna()
    if series.empty:
        return float("nan")
    return float(series.mean())


def _numeric_quantile(df: pd.DataFrame, column: str, q: float) -> float:
    series = _numeric_series(df, column).dropna()
    if series.empty:
        return float("nan")
    return float(series.quantile(q))


def _numeric_last(df: pd.DataFrame, column: str) -> float:
    series = _numeric_series(df, column).dropna()
    if series.empty:
        return float("nan")
    return float(series.iloc[-1])


def _bool_mask(df: pd.DataFrame, column: str) -> pd.Series:
    return _numeric_series(df, column).fillna(0.0) > 0.0


def _parse_float_list(spec: str) -> List[float]:
    return [float(token.strip()) for token in str(spec).split(",") if token.strip()]


def _parse_int_list(spec: str) -> List[int]:
    return [max(1, int(token.strip())) for token in str(spec).split(",") if token.strip()]


def _parse_attack_ids(spec: str) -> List[str]:
    values = [str(token).strip().upper() for token in str(spec).split(",") if str(token).strip()]
    return sorted(values, key=lambda item: ATTACK_ORDER.get(item, 99))


def _slug_float(value: float) -> str:
    return str(f"{float(value):.4f}").replace("-", "m").replace(".", "p")


def _condition_tag(
    *,
    section: str,
    backend_mode: str,
    shots: int,
    noise_1q: float,
    noise_2q: float,
    readout_error: float,
) -> str:
    return "_".join(
        [
            str(section).strip().lower(),
            str(backend_mode).strip().lower(),
            f"shots{int(shots)}",
            f"n1q{_slug_float(noise_1q)}",
            f"n2q{_slug_float(noise_2q)}",
            f"ro{_slug_float(readout_error)}",
        ]
    )


def _copy_condition_artifacts(paths: Dict[str, str], condition_tag: str) -> Dict[str, str]:
    copied = dict(paths)
    for key in ("sim_csv", "meta_json"):
        raw_path = copied.get(key)
        if not raw_path:
            continue
        src = Path(raw_path)
        if not src.exists():
            continue
        dst = src.with_name(f"{src.stem}_{condition_tag}{src.suffix}")
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        copied[key] = str(dst)
    return copied


def _results_tables_dir(cfg_path: str) -> Path:
    cfg = load_config(cfg_path)
    return Path(cfg.get("project", {}).get("results_dir", "results")) / "tables"


def _robustness_artifact_path(cfg_path: str, condition_tag: str) -> Path:
    cfg = load_config(cfg_path)
    artifacts_dir = Path(cfg.get("project", {}).get("artifacts_dir", "artifacts"))
    return artifacts_dir / "qbm_calibration" / "robustness" / f"{condition_tag}.json"


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


def _prepare_cfg(
    cfg_path: str,
    *,
    scenario: str,
    attack_id: str,
    pipeline_mode: str,
    backend_mode: str,
    shots: int,
    noise_1q: float,
    noise_2q: float,
    readout_error: float,
    artifact_path: Path,
    seed: int | None,
    max_windows: int | None,
) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    cfg = apply_overrides(cfg, scenario=scenario, attack_id=attack_id, verifier_impl="s3_mev", seed=seed)
    cfg.setdefault("verification", {})
    cfg["verification"]["pipeline_mode"] = str(pipeline_mode).strip().lower()
    cfg["verification"]["qbm_use_saved_calibration"] = True
    cfg["verification"]["qbm_save_calibration"] = True
    cfg["verification"]["qbm_auto_threshold_from_a0"] = True
    cfg["verification"]["qbm_calibration_artifact"] = str(artifact_path.as_posix())

    cfg.setdefault("qbm", {})
    cfg["qbm"]["qiskit_backend_mode"] = str(backend_mode).strip().lower()
    cfg["qbm"]["shots"] = int(shots)
    cfg["qbm"]["qiskit_noise_1q"] = float(noise_1q)
    cfg["qbm"]["qiskit_noise_2q"] = float(noise_2q)
    cfg["qbm"]["qiskit_readout_error"] = float(readout_error)

    cfg.setdefault("experiments", {})
    cfg["experiments"]["enable_injection"] = str(attack_id).upper() != "A0"
    cfg["experiments"]["attack_id"] = str(attack_id).upper()
    _adjust_injection_window(cfg, max_windows)
    return cfg


def _summary_row(
    *,
    section: str,
    sim_df: pd.DataFrame,
    summary: Any,
    pipeline_mode: str,
    backend_mode: str,
    shots: int,
    noise_1q: float,
    noise_2q: float,
    readout_error: float,
    artifact_path: Path,
    paths: Mapping[str, str],
) -> Dict[str, Any]:
    malicious_mask = _bool_mask(sim_df, "malicious_visible")
    benign_mask = ~malicious_mask
    eligible_mask = _bool_mask(sim_df, "qbm_stage2_eligible")
    veto_mask = _bool_mask(sim_df, "qbm_stage2_veto")
    shadow_scores = _numeric_series(sim_df, "q_score_shadow")
    threshold = _numeric_last(sim_df, "qbm_threshold")
    delta = shadow_scores - float(threshold)

    return {
        "section": section,
        "scenario": summary.scenario,
        "attack_id": summary.attack_id,
        "pipeline_mode": pipeline_mode,
        "backend_mode": backend_mode,
        "shots": int(shots),
        "noise_1q": float(noise_1q),
        "noise_2q": float(noise_2q),
        "readout_error": float(readout_error),
        "noise_level": float(max(noise_1q, noise_2q, readout_error)),
        "qbm_threshold_final": float(threshold),
        "qbm_shadow_windows": int(shadow_scores.dropna().shape[0]),
        "qbm_stage2_eligible_windows": int(eligible_mask.sum()),
        "qbm_veto_count": int(veto_mask.sum()),
        "qbm_benign_veto_count": int((veto_mask & benign_mask).sum()),
        "qbm_malicious_veto_count": int((veto_mask & malicious_mask).sum()),
        "qbm_veto_rate": float(veto_mask.mean()),
        "qbm_shadow_mean": _numeric_mean(sim_df, "q_score_shadow"),
        "qbm_shadow_mean_benign": float(shadow_scores[benign_mask].dropna().mean()),
        "qbm_shadow_mean_malicious": float(shadow_scores[malicious_mask].dropna().mean()),
        "qbm_shadow_std_observed": float(shadow_scores.dropna().std(ddof=0)) if shadow_scores.dropna().shape[0] > 0 else float("nan"),
        "qbm_shadow_delta_mean": float(delta.dropna().mean()) if delta.dropna().shape[0] > 0 else float("nan"),
        "qbm_shadow_delta_mean_benign": float(delta[benign_mask].dropna().mean()) if delta[benign_mask].dropna().shape[0] > 0 else float("nan"),
        "qbm_shadow_delta_mean_malicious": float(delta[malicious_mask].dropna().mean()) if delta[malicious_mask].dropna().shape[0] > 0 else float("nan"),
        "qbm_shadow_delta_q05": float(delta.dropna().quantile(0.05)) if delta.dropna().shape[0] > 0 else float("nan"),
        "qbm_shot_std_mean": _numeric_mean(sim_df, "qbm_shot_std"),
        "qbm_shot_std_q95": _numeric_quantile(sim_df, "qbm_shot_std", 0.95),
        "qbm_uncertainty_proxy_mean": _numeric_mean(sim_df, "qbm_uncertainty_proxy"),
        "asr": float(summary.asr),
        "ftr": float(summary.ftr),
        "tcp": float(summary.tcp),
        "processed_tps_mean": float(summary.processed_tps_mean),
        "latency_ms_mean": float(summary.latency_ms_mean),
        "qbm_calibration_artifact": str(artifact_path.as_posix()),
        **dict(paths),
    }


def _backend_window_frame(
    *,
    sim_df: pd.DataFrame,
    attack_id: str,
    backend_mode: str,
    shots: int,
    noise_1q: float,
    noise_2q: float,
    readout_error: float,
) -> pd.DataFrame:
    keep = pd.DataFrame(
        {
            "attack_id": attack_id,
            "backend_mode": backend_mode,
            "shots": int(shots),
            "noise_1q": float(noise_1q),
            "noise_2q": float(noise_2q),
            "readout_error": float(readout_error),
            "malicious_visible": _numeric_series(sim_df, "malicious_visible"),
            "qbm_stage2_veto": _numeric_series(sim_df, "qbm_stage2_veto"),
            "q_score_shadow": _numeric_series(sim_df, "q_score_shadow"),
            "qbm_threshold": _numeric_series(sim_df, "qbm_threshold"),
        }
    )
    keep["q_score_shadow_minus_threshold"] = keep["q_score_shadow"] - keep["qbm_threshold"]
    return keep.dropna(subset=["q_score_shadow"]).reset_index(drop=True)


def _run_condition_group(
    cfg_path: str,
    *,
    section: str,
    scenario: str,
    attack_ids: Sequence[str],
    pipeline_mode: str,
    backend_mode: str,
    shots: int,
    noise_1q: float,
    noise_2q: float,
    readout_error: float,
    max_windows: int | None,
    seed: int | None,
    force_refit_calibration: bool,
    collect_backend_windows: bool = False,
) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame]]:
    attack_plan = sorted([str(item).upper() for item in attack_ids], key=lambda item: ATTACK_ORDER.get(item, 99))
    condition_tag = _condition_tag(
        section=section,
        backend_mode=backend_mode,
        shots=shots,
        noise_1q=noise_1q,
        noise_2q=noise_2q,
        readout_error=readout_error,
    )
    artifact_path = _robustness_artifact_path(cfg_path, condition_tag)
    if force_refit_calibration and artifact_path.exists():
        artifact_path.unlink()

    rows: List[Dict[str, Any]] = []
    backend_windows: List[pd.DataFrame] = []
    for attack_id in attack_plan:
        cfg = _prepare_cfg(
            cfg_path,
            scenario=scenario,
            attack_id=attack_id,
            pipeline_mode=pipeline_mode,
            backend_mode=backend_mode,
            shots=shots,
            noise_1q=noise_1q,
            noise_2q=noise_2q,
            readout_error=readout_error,
            artifact_path=artifact_path,
            seed=seed,
            max_windows=max_windows,
        )
        sim_df, summary = run_simulation(cfg, max_windows=max_windows)
        paths = _copy_condition_artifacts(save_outputs(cfg, sim_df, summary), condition_tag)
        rows.append(
            _summary_row(
                section=section,
                sim_df=sim_df,
                summary=summary,
                pipeline_mode=pipeline_mode,
                backend_mode=backend_mode,
                shots=shots,
                noise_1q=noise_1q,
                noise_2q=noise_2q,
                readout_error=readout_error,
                artifact_path=artifact_path,
                paths=paths,
            )
        )
        if collect_backend_windows:
            backend_windows.append(
                _backend_window_frame(
                    sim_df=sim_df,
                    attack_id=summary.attack_id,
                    backend_mode=backend_mode,
                    shots=shots,
                    noise_1q=noise_1q,
                    noise_2q=noise_2q,
                    readout_error=readout_error,
                )
            )
        print(
            f"section={section} attack={summary.attack_id} backend={backend_mode} shots={shots} "
            f"noise=({noise_1q:.4f},{noise_2q:.4f},{readout_error:.4f}) "
            f"FTR={summary.ftr:.4f} TCP={summary.tcp:.4f} veto={rows[-1]['qbm_veto_count']}"
        )
    return rows, backend_windows


def run_robustness(
    cfg_path: str,
    *,
    scenario: str,
    pipeline_mode: str,
    sections: Sequence[str],
    shot_attack_ids: Sequence[str],
    noise_attack_ids: Sequence[str],
    backend_attack_ids: Sequence[str],
    shots_list: Sequence[int],
    noise_levels: Sequence[float],
    backend_compare_shots: int,
    max_windows: int | None,
    seed: int | None,
    force_refit_calibration: bool,
) -> Dict[str, pd.DataFrame]:
    base_cfg = load_config(cfg_path)
    base_qbm = dict(base_cfg.get("qbm", {}))
    base_noise_1q = float(base_qbm.get("qiskit_noise_1q", 0.0015))
    base_noise_2q = float(base_qbm.get("qiskit_noise_2q", 0.0080))
    base_readout = float(base_qbm.get("qiskit_readout_error", 0.0100))
    tables_dir = _results_tables_dir(cfg_path)
    tables_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, pd.DataFrame] = {}
    sections_set = {str(item).strip().lower() for item in sections}

    if "shot" in sections_set:
        shot_rows: List[Dict[str, Any]] = []
        for shots in shots_list:
            rows, _ = _run_condition_group(
                cfg_path,
                section="shot",
                scenario=scenario,
                attack_ids=shot_attack_ids,
                pipeline_mode=pipeline_mode,
                backend_mode="aer_shot",
                shots=int(shots),
                noise_1q=0.0,
                noise_2q=0.0,
                readout_error=0.0,
                max_windows=max_windows,
                seed=seed,
                force_refit_calibration=force_refit_calibration,
            )
            shot_rows.extend(rows)
        shot_df = pd.DataFrame(shot_rows).sort_values(["shots", "attack_id"]).reset_index(drop=True)
        shot_path = tables_dir / "qiskit_qbm_robustness_shot_sensitivity.csv"
        shot_df.to_csv(shot_path, index=False)
        print(f"saved: {shot_path}")
        outputs["shot"] = shot_df

    if "noise" in sections_set:
        noise_rows: List[Dict[str, Any]] = []
        for noise_level in noise_levels:
            rows, _ = _run_condition_group(
                cfg_path,
                section="noise",
                scenario=scenario,
                attack_ids=noise_attack_ids,
                pipeline_mode=pipeline_mode,
                backend_mode="aer_shot",
                shots=int(backend_compare_shots),
                noise_1q=float(noise_level),
                noise_2q=float(noise_level),
                readout_error=float(noise_level),
                max_windows=max_windows,
                seed=seed,
                force_refit_calibration=force_refit_calibration,
            )
            noise_rows.extend(rows)
        noise_df = pd.DataFrame(noise_rows).sort_values(["noise_level", "attack_id"]).reset_index(drop=True)
        noise_path = tables_dir / "qiskit_qbm_robustness_noise_sweep.csv"
        noise_df.to_csv(noise_path, index=False)
        print(f"saved: {noise_path}")
        outputs["noise"] = noise_df

    if "backend" in sections_set:
        backend_rows: List[Dict[str, Any]] = []
        backend_windows: List[pd.DataFrame] = []
        specs = [
            ("exact_state", int(backend_compare_shots), 0.0, 0.0, 0.0),
            ("aer_shot", int(backend_compare_shots), base_noise_1q, base_noise_2q, base_readout),
        ]
        for backend_mode, shots, noise_1q, noise_2q, readout_error in specs:
            rows, window_frames = _run_condition_group(
                cfg_path,
                section="backend",
                scenario=scenario,
                attack_ids=backend_attack_ids,
                pipeline_mode=pipeline_mode,
                backend_mode=backend_mode,
                shots=shots,
                noise_1q=noise_1q,
                noise_2q=noise_2q,
                readout_error=readout_error,
                max_windows=max_windows,
                seed=seed,
                force_refit_calibration=force_refit_calibration,
                collect_backend_windows=True,
            )
            backend_rows.extend(rows)
            backend_windows.extend(window_frames)
        backend_df = pd.DataFrame(backend_rows).sort_values(["backend_mode", "attack_id"]).reset_index(drop=True)
        backend_windows_df = pd.concat(backend_windows, ignore_index=True) if backend_windows else pd.DataFrame()
        backend_path = tables_dir / "qiskit_qbm_robustness_backend_compare.csv"
        backend_windows_path = tables_dir / "qiskit_qbm_robustness_backend_windows.csv"
        backend_df.to_csv(backend_path, index=False)
        backend_windows_df.to_csv(backend_windows_path, index=False)
        print(f"saved: {backend_path}")
        print(f"saved: {backend_windows_path}")
        outputs["backend"] = backend_df
        outputs["backend_windows"] = backend_windows_df

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness appendix experiments for the Qiskit forensic gate.")
    parser.add_argument("--config", default="configs/experiments/qiskit_qbm_robustness.yaml", help="Path to configuration yaml.")
    parser.add_argument("--scenario", default="S3", help="Scenario to run. Default: S3")
    parser.add_argument("--pipeline-mode", default="s3_qbm_strongq", help="Pipeline mode. Default: s3_qbm_strongq")
    parser.add_argument("--sections", default="shot,noise,backend", help="Comma-separated sections: shot,noise,backend")
    parser.add_argument("--shot-attack-ids", default="A0,A4P", help="Comma-separated attack ids for shot sensitivity.")
    parser.add_argument("--noise-attack-ids", default="A0,A4P", help="Comma-separated attack ids for noise robustness.")
    parser.add_argument("--backend-attack-ids", default="A0,A4P,A5", help="Comma-separated attack ids for backend comparison.")
    parser.add_argument("--shots", default="128,256,512,1024,2048", help="Comma-separated shot counts for shot sensitivity.")
    parser.add_argument("--noise-levels", default="0,0.001,0.005,0.01", help="Comma-separated uniform Aer noise levels.")
    parser.add_argument("--backend-compare-shots", type=int, default=1024, help="Shot count for aer_shot backend comparison and noise sweep.")
    parser.add_argument("--max-windows", type=int, default=180, help="Run first N windows.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument(
        "--force-refit-calibration",
        action="store_true",
        help="Delete existing robustness calibration artifacts for each sweep condition before running.",
    )
    args = parser.parse_args()

    run_robustness(
        cfg_path=args.config,
        scenario=str(args.scenario).upper(),
        pipeline_mode=str(args.pipeline_mode).strip().lower(),
        sections=[item.strip().lower() for item in str(args.sections).split(",") if item.strip()],
        shot_attack_ids=_parse_attack_ids(args.shot_attack_ids),
        noise_attack_ids=_parse_attack_ids(args.noise_attack_ids),
        backend_attack_ids=_parse_attack_ids(args.backend_attack_ids),
        shots_list=_parse_int_list(args.shots),
        noise_levels=_parse_float_list(args.noise_levels),
        backend_compare_shots=max(1, int(args.backend_compare_shots)),
        max_windows=args.max_windows,
        seed=args.seed,
        force_refit_calibration=bool(args.force_refit_calibration),
    )


if __name__ == "__main__":
    main()
