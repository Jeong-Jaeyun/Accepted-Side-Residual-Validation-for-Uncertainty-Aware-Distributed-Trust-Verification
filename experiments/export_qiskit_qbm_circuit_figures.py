from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import yaml

from qbm.backend.qiskit_qbm import ResidualForensicQiskitScorer


FEATURE_LABELS = {
    "r_trust_sim_gap": "trust-sim gap",
    "r_overlap_spread": "overlap spread",
    "r_context_dev": "context dev",
    "r_policy_gray_contra": "policy-gray contra",
    "r_trust_sim_prod": "trust-sim prod",
    "r_benign_delta": "benign delta",
    "r_temporal_inconsistency": "temporal inconsistency",
    "r_witness_disagree": "witness disagree",
}


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_effective_config(default_path: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    default_cfg = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}
    main_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    merged = _merge_dict(default_cfg, main_cfg)
    qbm_cfg = dict(merged.get("qbm", {}))
    verification_cfg = dict(merged.get("verification", {}))
    return qbm_cfg, verification_cfg


def load_scorer(default_path: Path, config_path: Path) -> tuple[ResidualForensicQiskitScorer, Path]:
    qbm_cfg, verification_cfg = load_effective_config(default_path, config_path)
    scorer = ResidualForensicQiskitScorer(qbm_cfg, shots=1024)
    artifact_path = Path(str(verification_cfg["qbm_calibration_artifact"]))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    scorer.apply_calibration(artifact)
    return scorer, artifact_path


def save_mpl_figure(fig: plt.Figure, path_stem: Path, *, dpi: int) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def export_full_circuit_views(scorer: ResidualForensicQiskitScorer, out_dir: Path) -> None:
    qc = scorer.template_circuit

    wide = qc.draw(output="mpl", fold=-1, idle_wires=False, style="bw")
    wide.set_size_inches(36, 7)
    wide.tight_layout()
    save_mpl_figure(wide, out_dir / "qiskit_qbm_circuit_full_wide", dpi=320)

    folded = qc.draw(output="mpl", fold=18, idle_wires=False, style="bw")
    folded.set_size_inches(18, 11)
    folded.tight_layout()
    save_mpl_figure(folded, out_dir / "qiskit_qbm_circuit_full_folded", dpi=360)


def _draw_block(ax, x: float, y_center: float, width: float, height: float, label: str, *, fc: str) -> None:
    rect = FancyBboxPatch(
        (x, y_center - (height / 2.0)),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4,
        edgecolor="#222222",
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(
        x + (width / 2.0),
        y_center,
        label,
        ha="center",
        va="center",
        fontsize=10,
        family="DejaVu Sans",
    )


def export_schematic_view(scorer: ResidualForensicQiskitScorer, out_dir: Path) -> None:
    num_visible = scorer.num_visible
    num_hidden = scorer.num_hidden
    num_qubits = scorer.num_qubits
    y_positions = {idx: float(num_qubits - 1 - idx) for idx in range(num_qubits)}

    fig, ax = plt.subplots(figsize=(20, 7))
    ax.set_xlim(0.0, 17.4)
    ax.set_ylim(-1.0, float(num_qubits))
    ax.axis("off")

    x_start = 0.7
    x_end = 16.7
    for idx in range(num_qubits):
        y = y_positions[idx]
        ax.plot([x_start, x_end], [y, y], color="#444444", lw=1.2, zorder=1)

    for idx, key in enumerate(scorer.feature_keys):
        y = y_positions[idx]
        label = FEATURE_LABELS.get(key, key)
        ax.text(0.05, y, f"q{idx}  {label}", ha="left", va="center", fontsize=10, family="DejaVu Sans")

    for hidden_idx in range(num_hidden):
        qidx = num_visible + hidden_idx
        y = y_positions[qidx]
        ax.text(0.05, y, f"q{qidx}  hidden {hidden_idx}", ha="left", va="center", fontsize=10, family="DejaVu Sans")

    visible_center = (y_positions[0] + y_positions[num_visible - 1]) / 2.0
    hidden_center = (y_positions[num_visible] + y_positions[num_qubits - 1]) / 2.0
    full_center = (y_positions[0] + y_positions[num_qubits - 1]) / 2.0

    _draw_block(ax, 2.0, visible_center, 1.7, 6.8, "Visible\nencoding\nRY + RZ", fc="#dbeafe")
    _draw_block(ax, 2.0, hidden_center, 1.7, 1.5, "Hidden\ninit RY", fc="#fde68a")

    _draw_block(ax, 4.5, full_center, 1.9, 8.5, "Layer 1\nvisible-hidden\nRZZ + CRY", fc="#dcfce7")
    _draw_block(ax, 6.8, hidden_center, 1.5, 1.7, "Layer 1\nhidden\nmixers RX", fc="#fee2e2")
    _draw_block(ax, 8.8, full_center, 2.2, 8.5, "Layer 1\npairwise RZZ\n(hidden-hidden\n+ visible-visible)", fc="#f3e8ff")

    _draw_block(ax, 11.5, full_center, 1.9, 8.5, "Layer 2\nvisible-hidden\nRZZ + CRY", fc="#dcfce7")
    _draw_block(ax, 13.8, hidden_center, 1.5, 1.7, "Layer 2\nhidden\nmixers RX", fc="#fee2e2")
    _draw_block(ax, 15.8, full_center, 2.2, 8.5, "Layer 2\npairwise RZZ\n(hidden-hidden\n+ visible-visible)", fc="#f3e8ff")

    ax.text(
        8.6,
        num_qubits - 0.1,
        "QBM-inspired residual forensic circuit (exact-state main configuration)",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        family="DejaVu Sans",
    )
    ax.text(
        8.6,
        -0.7,
        f"{num_visible} visible residual features + {num_hidden} hidden qubits, {scorer.layers} entangling layers, "
        f"observable-based stage-2 veto",
        ha="center",
        va="top",
        fontsize=10,
        family="DejaVu Sans",
        color="#333333",
    )

    save_mpl_figure(fig, out_dir / "qiskit_qbm_circuit_schematic", dpi=360)


def write_metadata(scorer: ResidualForensicQiskitScorer, artifact_path: Path, out_dir: Path) -> Path:
    qc = scorer.template_circuit
    out_path = out_dir / "qiskit_qbm_circuit_meta.json"
    payload = {
        "name": qc.name,
        "backend_mode": scorer.backend_mode,
        "num_qubits": qc.num_qubits,
        "num_visible": scorer.num_visible,
        "num_hidden": scorer.num_hidden,
        "layers": scorer.layers,
        "depth": qc.depth(),
        "size": qc.size(),
        "feature_keys": list(scorer.feature_keys),
        "visible_qubits": {f"q_{idx}": key for idx, key in enumerate(scorer.feature_keys)},
        "hidden_qubits": {f"q_{scorer.num_visible + idx}": f"hidden_{idx}" for idx in range(scorer.num_hidden)},
        "calibration_artifact": str(artifact_path).replace("\\", "/"),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def mirror_outputs(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.glob("qiskit_qbm_circuit_*"):
        shutil.copy2(item, dst_dir / item.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export readable circuit figures for the current QBM configuration.")
    parser.add_argument("--config", default="configs/experiments/qiskit_qbm_main.yaml")
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default="results/figures/qiskit_qbm/main")
    parser.add_argument("--mirror-dir", default="IEEE/fig")
    args = parser.parse_args()

    default_path = Path(args.default_config)
    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    mirror_dir = Path(args.mirror_dir)

    scorer, artifact_path = load_scorer(default_path, config_path)
    export_full_circuit_views(scorer, out_dir)
    export_schematic_view(scorer, out_dir)
    write_metadata(scorer, artifact_path, out_dir)
    mirror_outputs(out_dir, mirror_dir)

    print(f"exported to {out_dir}")
    print(f"mirrored to {mirror_dir}")


if __name__ == "__main__":
    main()
