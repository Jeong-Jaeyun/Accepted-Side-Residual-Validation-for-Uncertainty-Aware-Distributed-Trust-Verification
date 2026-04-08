from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from qbm.train import apply_overrides, load_config, run_simulation, save_outputs


def _prepare_cfg(
    cfg_path: str,
    *,
    scenario: str,
    seed: int | None,
    mode: str,
    contrastive_attack_ids: List[str],
    backend_mode: str,
) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    cfg = apply_overrides(cfg, scenario=scenario, attack_id="A0", verifier_impl="qbm", seed=seed)
    cfg.setdefault("experiments", {})
    cfg["experiments"]["enable_injection"] = False
    cfg["experiments"]["attack_id"] = "A0"
    cfg.setdefault("qbm", {})
    cfg["qbm"]["qiskit_backend_mode"] = backend_mode
    cfg.setdefault("verification", {})
    cfg["verification"]["qbm_auto_threshold_from_a0"] = True
    cfg["verification"]["qbm_use_saved_calibration"] = False
    cfg["verification"]["qbm_save_calibration"] = True
    cfg["verification"]["qbm_calibration_mode"] = mode
    cfg["verification"]["qbm_contrastive_attack_ids"] = [attack_id.upper() for attack_id in contrastive_attack_ids]
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and save a reusable QBM calibration artifact.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to configuration yaml.")
    parser.add_argument("--scenario", default="S3", help="Scenario to run. Default: S3")
    parser.add_argument("--max-windows", type=int, default=180, help="Run first N windows for calibration.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--mode", default="benign_only", help="Calibration mode: benign_only | contrastive")
    parser.add_argument("--contrastive-attack-ids", default="", help="Comma-separated attack ids used only when mode=contrastive")
    parser.add_argument("--backend-mode", default="exact_state", help="exact_state | aer_shot")
    args = parser.parse_args()

    contrastive_attack_ids = [item.strip().upper() for item in str(args.contrastive_attack_ids).split(",") if item.strip()]
    cfg = _prepare_cfg(
        args.config,
        scenario=str(args.scenario).upper(),
        seed=args.seed,
        mode=str(args.mode).strip().lower(),
        contrastive_attack_ids=contrastive_attack_ids,
        backend_mode=str(args.backend_mode).strip().lower(),
    )
    sim_df, summary = run_simulation(cfg, max_windows=args.max_windows)
    paths = save_outputs(cfg, sim_df, summary)
    runtime = dict(sim_df.attrs.get("verifier_runtime", {}))
    print(f"scenario={summary.scenario} verifier={summary.verifier_name}")
    print(f"qbm_threshold={runtime.get('qbm_threshold_value')}")
    print(f"artifact={runtime.get('qbm_calibration_artifact')}")
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
