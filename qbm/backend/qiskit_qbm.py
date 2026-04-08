from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from qbm.backend.forensic_features import (
    FORENSIC_DIRECTIONALITY_MODE,
    FORENSIC_FEATURE_KEYS,
    FORENSIC_FEATURE_SCHEMA_VERSION,
    FORENSIC_REFERENCE_MODE,
    FORENSIC_EXPECTED_CONTEXT_MODE,
    MINIMAL_SIGNAL_KEYS,
    build_forensic_features,
    can_build_forensic_features,
    encode_forensic_feature,
    forensic_feature_schema,
    suspicious_feature_value,
)
from qbm.model import clamp01

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp
except Exception as exc:  # pragma: no cover
    QuantumCircuit = None
    ParameterVector = None
    StatevectorEstimator = None
    SparsePauliOp = None
    _QISKIT_IMPORT_ERROR = exc
else:
    _QISKIT_IMPORT_ERROR = None

try:
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    from qiskit_aer.primitives import Estimator as AerEstimator
except Exception:  # pragma: no cover
    NoiseModel = None
    ReadoutError = None
    depolarizing_error = None
    AerEstimator = None


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class _ResidualForensicBase:
    DEFAULT_FEATURE_KEYS = FORENSIC_FEATURE_KEYS
    REQUIRED_SIGNAL_KEYS = MINIMAL_SIGNAL_KEYS

    def __init__(self, cfg: Mapping[str, float | int | Sequence[float]] | None = None, *, shots: int = 2048) -> None:
        qcfg = dict(cfg or {})
        self.cfg = qcfg
        self.shots = max(int(shots), 1)
        self.seed = int(qcfg.get("qiskit_seed", 42))
        self.layers = max(1, int(qcfg.get("qiskit_layers", 2)))
        self.num_hidden = max(1, int(qcfg.get("qiskit_num_hidden", 2)))
        self.input_phase_scale = float(qcfg.get("qiskit_input_phase_scale", 0.28))
        self.entangle_scale = float(qcfg.get("qiskit_entangle_scale", 0.18))
        self.score_scale = float(qcfg.get("qiskit_score_scale", 1.28))
        self.score_offset = float(qcfg.get("qiskit_score_offset", 0.82))
        self.shot_penalty = float(qcfg.get("qiskit_shot_penalty", 0.22))
        self.precision = float(max(0.0, qcfg.get("qiskit_precision", 0.0)))
        self.feature_keys = self._coerce_feature_keys(qcfg.get("qiskit_feature_keys"))
        self.feature_set_version = str(qcfg.get("qbm_feature_set_version", FORENSIC_FEATURE_SCHEMA_VERSION))
        self.calibration_version = str(qcfg.get("qbm_calibration_version", "benign_only_v1"))
        self.num_visible = len(self.feature_keys)
        self.num_qubits = self.num_visible + self.num_hidden
        self.benign_reference = self._coerce_mapping(qcfg.get("qiskit_benign_reference"))
        self.expected_context_weights = self._coerce_mapping(qcfg.get("qiskit_expected_context_weights"))

        self.visible_weights = self._coerce_vector(
            qcfg.get("qiskit_visible_weights"),
            default=[self._default_visible_weight(key) for key in self.feature_keys],
            size=self.num_visible,
        )
        hidden_weight_default = [0.16 + (0.04 * idx) for idx in range(self.num_hidden)]
        if "qiskit_hidden_weight" in qcfg:
            hidden_weight_default = [float(qcfg.get("qiskit_hidden_weight", 0.20))] * self.num_hidden
        self.hidden_weights = self._coerce_vector(
            qcfg.get("qiskit_hidden_weights"),
            default=hidden_weight_default,
            size=self.num_hidden,
        )
        hidden_bias_default = [0.54 + (0.12 * min(idx, 1)) for idx in range(self.num_hidden)]
        if "qiskit_hidden_bias" in qcfg:
            hidden_bias_default = [float(qcfg.get("qiskit_hidden_bias", 0.66))] * self.num_hidden
        self.hidden_biases = self._coerce_vector(
            qcfg.get("qiskit_hidden_biases"),
            default=hidden_bias_default,
            size=self.num_hidden,
        )
        hidden_mixer_default = [0.10 + (0.04 * min(idx, 1)) for idx in range(self.num_hidden)]
        if "qiskit_hidden_mixer" in qcfg:
            hidden_mixer_default = [float(qcfg.get("qiskit_hidden_mixer", 0.14))] * self.num_hidden
        self.hidden_mixers = self._coerce_vector(
            qcfg.get("qiskit_hidden_mixers"),
            default=hidden_mixer_default,
            size=self.num_hidden,
        )
        self.hidden_couplings = self._coerce_matrix(
            qcfg.get("qiskit_hidden_couplings"),
            default=self._default_hidden_couplings(),
            rows=self.num_hidden,
            cols=self.num_visible,
        )
        self.visible_hidden_weights = self._coerce_matrix(
            qcfg.get("qiskit_visible_hidden_weights"),
            default=self._default_visible_hidden_weights(),
            rows=self.num_hidden,
            cols=self.num_visible,
        )
        self.hidden_hidden_weight = float(qcfg.get("qiskit_hidden_hidden_weight", 0.12))
        self.visible_pair_couplings = self._coerce_triplets(
            qcfg.get("qiskit_visible_pair_couplings"),
            default=self._default_visible_pair_couplings(),
        )
        self.visible_pair_weights = self._coerce_triplets(
            qcfg.get("qiskit_visible_pair_weights"),
            default=self._default_visible_pair_weights(),
        )

    @staticmethod
    def _coerce_feature_keys(values: object) -> tuple[str, ...]:
        if isinstance(values, (list, tuple)):
            keys = [str(value).strip() for value in values if str(value).strip()]
            if keys:
                return tuple(keys)
        return _ResidualForensicBase.DEFAULT_FEATURE_KEYS

    @staticmethod
    def _coerce_mapping(values: object) -> dict[str, float]:
        if not isinstance(values, Mapping):
            return {}
        return {str(k): float(v) for k, v in values.items()}

    @staticmethod
    def _coerce_vector(values: object, *, default: Sequence[float], size: int) -> tuple[float, ...]:
        if isinstance(values, (list, tuple)):
            coerced = [float(v) for v in values[:size]]
        else:
            coerced = []
        if len(coerced) < size:
            coerced.extend(float(v) for v in default[len(coerced) : size])
        return tuple(coerced[:size])

    @staticmethod
    def _coerce_matrix(values: object, *, default: Sequence[Sequence[float]], rows: int, cols: int) -> tuple[tuple[float, ...], ...]:
        if isinstance(values, (list, tuple)) and values and isinstance(values[0], (list, tuple)):
            matrix = [tuple(float(v) for v in row[:cols]) for row in values[:rows]]
        elif isinstance(values, (list, tuple)):
            flat = [float(v) for v in values]
            matrix = [tuple(flat[:cols])]
        else:
            matrix = []

        default_rows = [tuple(float(v) for v in row[:cols]) for row in default[:rows]]
        while len(matrix) < rows:
            matrix.append(default_rows[len(matrix)] if len(default_rows) > len(matrix) else tuple(0.0 for _ in range(cols)))

        out: list[tuple[float, ...]] = []
        for row_idx in range(rows):
            row = list(matrix[row_idx])
            default_row = list(default_rows[row_idx]) if row_idx < len(default_rows) else [0.0] * cols
            if len(row) < cols:
                row.extend(default_row[len(row) : cols])
            out.append(tuple(row[:cols]))
        return tuple(out)

    @staticmethod
    def _coerce_triplets(values: object, *, default: Sequence[Sequence[float]]) -> tuple[tuple[int, int, float], ...]:
        source = values if isinstance(values, (list, tuple)) else default
        triplets: list[tuple[int, int, float]] = []
        for item in source:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            triplets.append((int(item[0]), int(item[1]), float(item[2])))
        return tuple(triplets)

    @staticmethod
    def _default_visible_weight(feature_key: str) -> float:
        defaults = {
            "r_trust_sim_gap": 0.92,
            "r_overlap_spread": 0.84,
            "r_context_dev": 0.88,
            "r_policy_gray_contra": 0.90,
            "r_trust_sim_prod": 0.56,
            "r_benign_delta": 0.86,
            "r_temporal_inconsistency": 0.82,
            "r_witness_disagree": 0.88,
        }
        return float(defaults.get(feature_key, 0.72))

    def _default_hidden_couplings(self) -> tuple[tuple[float, ...], ...]:
        if self.num_visible < 8:
            base = [0.24] * self.num_visible
            return tuple(tuple(base) for _ in range(self.num_hidden))
        default = (
            (0.34, 0.30, 0.12, 0.08, 0.28, 0.14, 0.16, 0.26),
            (0.10, 0.10, 0.32, 0.36, 0.10, 0.20, 0.28, 0.16),
        )
        return tuple(tuple(row[: self.num_visible]) for row in default[: self.num_hidden])

    def _default_visible_hidden_weights(self) -> tuple[tuple[float, ...], ...]:
        if self.num_visible < 8:
            base = [0.14] * self.num_visible
            return tuple(tuple(base) for _ in range(self.num_hidden))
        default = (
            (0.18, 0.16, 0.10, 0.08, 0.14, 0.10, 0.10, 0.16),
            (0.08, 0.08, 0.16, 0.20, 0.08, 0.12, 0.16, 0.10),
        )
        return tuple(tuple(row[: self.num_visible]) for row in default[: self.num_hidden])

    def _default_visible_pair_couplings(self) -> tuple[tuple[int, int, float], ...]:
        pairs = (
            (0, 1, 0.18),
            (0, 4, 0.14),
            (2, 3, 0.18),
            (2, 5, 0.20),
            (6, 7, 0.20),
            (3, 6, 0.16),
        )
        return tuple(pair for pair in pairs if pair[0] < self.num_visible and pair[1] < self.num_visible)

    def _default_visible_pair_weights(self) -> tuple[tuple[int, int, float], ...]:
        pairs = (
            (0, 1, 0.10),
            (0, 4, 0.08),
            (2, 3, 0.12),
            (2, 5, 0.12),
            (6, 7, 0.14),
            (3, 6, 0.10),
        )
        return tuple(pair for pair in pairs if pair[0] < self.num_visible and pair[1] < self.num_visible)

    def can_score(self, signals: Mapping[str, Any]) -> bool:
        return can_build_forensic_features(signals)

    def build_features(self, signals: Mapping[str, Any]) -> dict[str, float]:
        return build_forensic_features(
            signals,
            benign_reference=self.benign_reference,
            expected_context_weights=self.expected_context_weights,
        )

    def visible_values_from_signals(self, signals: Mapping[str, Any]) -> tuple[dict[str, float], list[float]]:
        features = self.build_features(signals)
        visible_values = [encode_forensic_feature(key, features) for key in self.feature_keys]
        return features, visible_values

    def export_calibration(self) -> dict[str, Any]:
        return {
            "feature_schema_version": self.feature_set_version,
            "qbm_feature_set_version": self.feature_set_version,
            "qbm_calibration_version": self.calibration_version,
            "reference_mode": FORENSIC_REFERENCE_MODE,
            "expected_context_mode": FORENSIC_EXPECTED_CONTEXT_MODE,
            "feature_keys": list(self.feature_keys),
            "benign_reference": dict(self.benign_reference),
            "expected_context_weights": dict(self.expected_context_weights),
            "visible_weights": list(self.visible_weights),
            "hidden_weights": list(self.hidden_weights),
            "hidden_biases": list(self.hidden_biases),
            "hidden_mixers": list(self.hidden_mixers),
            "hidden_couplings": [list(row) for row in self.hidden_couplings],
            "visible_hidden_weights": [list(row) for row in self.visible_hidden_weights],
            "hidden_hidden_weight": float(self.hidden_hidden_weight),
            "visible_pair_couplings": [list(item) for item in self.visible_pair_couplings],
            "visible_pair_weights": [list(item) for item in self.visible_pair_weights],
            "score_scale": float(self.score_scale),
            "score_offset": float(self.score_offset),
            "shot_penalty": float(self.shot_penalty),
        }

    def apply_calibration(self, artifact: Mapping[str, Any]) -> None:
        if not isinstance(artifact, Mapping):
            return
        if "qbm_feature_set_version" in artifact or "feature_schema_version" in artifact:
            self.feature_set_version = str(
                artifact.get("qbm_feature_set_version", artifact.get("feature_schema_version", self.feature_set_version))
            )
        if "qbm_calibration_version" in artifact:
            self.calibration_version = str(artifact.get("qbm_calibration_version", self.calibration_version))
        if isinstance(artifact.get("benign_reference"), Mapping):
            self.benign_reference = {str(k): float(v) for k, v in artifact.get("benign_reference", {}).items()}
        if isinstance(artifact.get("expected_context_weights"), Mapping):
            self.expected_context_weights = {
                str(k): float(v) for k, v in artifact.get("expected_context_weights", {}).items()
            }
        if isinstance(artifact.get("visible_weights"), (list, tuple)):
            self.visible_weights = self._coerce_vector(artifact.get("visible_weights"), default=self.visible_weights, size=self.num_visible)
        if isinstance(artifact.get("hidden_weights"), (list, tuple)):
            self.hidden_weights = self._coerce_vector(artifact.get("hidden_weights"), default=self.hidden_weights, size=self.num_hidden)
        if isinstance(artifact.get("hidden_biases"), (list, tuple)):
            self.hidden_biases = self._coerce_vector(artifact.get("hidden_biases"), default=self.hidden_biases, size=self.num_hidden)
        if isinstance(artifact.get("hidden_mixers"), (list, tuple)):
            self.hidden_mixers = self._coerce_vector(artifact.get("hidden_mixers"), default=self.hidden_mixers, size=self.num_hidden)
        if isinstance(artifact.get("hidden_couplings"), (list, tuple)):
            self.hidden_couplings = self._coerce_matrix(
                artifact.get("hidden_couplings"),
                default=self.hidden_couplings,
                rows=self.num_hidden,
                cols=self.num_visible,
            )
        if isinstance(artifact.get("visible_hidden_weights"), (list, tuple)):
            self.visible_hidden_weights = self._coerce_matrix(
                artifact.get("visible_hidden_weights"),
                default=self.visible_hidden_weights,
                rows=self.num_hidden,
                cols=self.num_visible,
            )
        if isinstance(artifact.get("visible_pair_couplings"), (list, tuple)):
            self.visible_pair_couplings = self._coerce_triplets(
                artifact.get("visible_pair_couplings"),
                default=self.visible_pair_couplings,
            )
        if isinstance(artifact.get("visible_pair_weights"), (list, tuple)):
            self.visible_pair_weights = self._coerce_triplets(
                artifact.get("visible_pair_weights"),
                default=self.visible_pair_weights,
            )
        if "hidden_hidden_weight" in artifact:
            self.hidden_hidden_weight = float(artifact.get("hidden_hidden_weight", self.hidden_hidden_weight))
        if "score_scale" in artifact:
            self.score_scale = float(artifact.get("score_scale", self.score_scale))
        if "score_offset" in artifact:
            self.score_offset = float(artifact.get("score_offset", self.score_offset))
        if "shot_penalty" in artifact:
            self.shot_penalty = float(artifact.get("shot_penalty", self.shot_penalty))


class ResidualForensicQiskitScorer(_ResidualForensicBase):
    """
    QBM-inspired residual forensic energy validator with reusable Qiskit circuits.

    This is not a trainable generative QBM. It is a shallow, non-deep,
    residualized forensic energy model that consumes engineered inconsistency
    features and evaluates them through a small Qiskit circuit for stage-2
    veto decisions.
    """

    def __init__(self, cfg: Mapping[str, float | int | Sequence[float]] | None = None, *, shots: int = 2048) -> None:
        if _QISKIT_IMPORT_ERROR is not None:
            raise RuntimeError(f"Qiskit backend is unavailable: {_QISKIT_IMPORT_ERROR}") from _QISKIT_IMPORT_ERROR
        super().__init__(cfg, shots=shots)
        mode = str(self.cfg.get("qiskit_backend_mode", self.cfg.get("qiskit_eval_mode", "exact_state"))).strip().lower()
        if mode in {"aer", "aer_shot", "shot", "shot_noise", "noisy_aer"}:
            self.backend_mode = "aer_shot"
        elif mode in {"ibm_backend", "ibm_runtime", "hardware"}:
            self.backend_mode = "ibm_backend"
        else:
            self.backend_mode = "exact_state"
        self.noise_1q = float(max(0.0, self.cfg.get("qiskit_noise_1q", 0.0015)))
        self.noise_2q = float(max(0.0, self.cfg.get("qiskit_noise_2q", 0.0080)))
        self.readout_error = float(max(0.0, self.cfg.get("qiskit_readout_error", 0.0100)))
        self.transpile_optimization_level = int(self.cfg.get("qiskit_transpile_optimization_level", 1))
        self.model_family = "qbm_inspired_residual_forensic_energy_validator"
        self.model_note = "non_generative_non_deep_stage2_forensic_validator"
        self.score_name = "residual_forensic_score"
        if self.backend_mode == "exact_state":
            self.uncertainty_mode = "analytic_expectation"
        elif self.backend_mode == "aer_shot":
            self.uncertainty_mode = "sampling_variance"
        else:
            self.uncertainty_mode = "device_and_sampling_variance"
        self._init_runtime()

    def _build_noise_model(self) -> NoiseModel | None:
        if NoiseModel is None or depolarizing_error is None or ReadoutError is None:
            return None
        noise_model = NoiseModel()
        if self.noise_1q > 0.0:
            error_1q = depolarizing_error(self.noise_1q, 1)
            for gate in ("ry", "rz", "rx"):
                noise_model.add_all_qubit_quantum_error(error_1q, gate)
        if self.noise_2q > 0.0:
            error_2q = depolarizing_error(self.noise_2q, 2)
            for gate in ("rzz", "cry", "cx"):
                noise_model.add_all_qubit_quantum_error(error_2q, gate)
        if self.readout_error > 0.0:
            ro = min(self.readout_error, 0.49)
            ro_error = ReadoutError([[1.0 - ro, ro], [ro, 1.0 - ro]])
            noise_model.add_all_qubit_readout_error(ro_error)
        return noise_model

    def _init_runtime(self) -> None:
        self.observable = SparsePauliOp.from_list(self._observable_terms())
        self.theta_params = ParameterVector("theta", self.num_visible)
        self.phase_params = ParameterVector("phase", self.num_visible)
        self.center_params = ParameterVector("center", self.num_visible)
        self.drive_params = ParameterVector("drive", self.num_visible)
        self._param_order = (
            [self.theta_params[idx] for idx in range(self.num_visible)]
            + [self.phase_params[idx] for idx in range(self.num_visible)]
            + [self.center_params[idx] for idx in range(self.num_visible)]
            + [self.drive_params[idx] for idx in range(self.num_visible)]
        )
        self.template_circuit = self._build_parameterized_template()
        self.template_depth = float(self.template_circuit.depth())
        self.template_size = float(self.template_circuit.size())

        if self.backend_mode == "aer_shot":
            if AerEstimator is None:
                raise RuntimeError("qiskit-aer is required for qiskit_backend_mode='aer_shot'")
            backend_options = {"noise_model": self._build_noise_model()}
            transpile_options = {
                "seed_transpiler": self.seed,
                "optimization_level": self.transpile_optimization_level,
            }
            run_options = {"shots": self.shots, "seed": self.seed}
            self.estimator = AerEstimator(
                backend_options=backend_options,
                transpile_options=transpile_options,
                run_options=run_options,
                approximation=False,
                skip_transpilation=False,
            )
            self.qbm_backend_name = "qiskit_aer_estimator"
        elif self.backend_mode == "ibm_backend":
            raise NotImplementedError(
                "qiskit_backend_mode='ibm_backend' is reserved for hardware feasibility runs and is not configured yet."
            )
        else:
            self.estimator = StatevectorEstimator(default_precision=self.precision, seed=self.seed)
            self.qbm_backend_name = "qiskit_statevector_exact"

    def apply_calibration(self, artifact: Mapping[str, Any]) -> None:
        super().apply_calibration(artifact)
        self._init_runtime()

    def _pauli_term(self, qubits: Sequence[int]) -> str:
        chars = ["I"] * self.num_qubits
        for qubit in qubits:
            chars[self.num_qubits - 1 - int(qubit)] = "Z"
        return "".join(chars)

    def _observable_terms(self) -> list[tuple[str, float]]:
        terms: list[tuple[str, float]] = []
        for idx, weight in enumerate(self.visible_weights):
            terms.append((self._pauli_term((idx,)), -float(weight)))

        hidden_indices = [self.num_visible + idx for idx in range(self.num_hidden)]
        for hidden_offset, hidden_idx in enumerate(hidden_indices):
            terms.append((self._pauli_term((hidden_idx,)), -float(self.hidden_weights[hidden_offset])))
            for visible_idx, weight in enumerate(self.visible_hidden_weights[hidden_offset]):
                terms.append((self._pauli_term((visible_idx, hidden_idx)), -float(weight)))

        if self.num_hidden >= 2:
            terms.append((self._pauli_term(tuple(hidden_indices[:2])), -float(self.hidden_hidden_weight)))
        for left, right, weight in self.visible_pair_weights:
            if 0 <= left < self.num_visible and 0 <= right < self.num_visible and left != right:
                terms.append((self._pauli_term((left, right)), -float(weight)))
        return terms

    def _build_parameterized_template(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name="residual_forensic_qbm")
        for idx in range(self.num_visible):
            qc.ry(self.theta_params[idx], idx)
            qc.rz(self.phase_params[idx], idx)

        hidden_indices = [self.num_visible + idx for idx in range(self.num_hidden)]
        for hidden_offset, hidden_idx in enumerate(hidden_indices):
            qc.ry(float(self.hidden_biases[hidden_offset]), hidden_idx)

        for _layer in range(self.layers):
            for hidden_offset, hidden_idx in enumerate(hidden_indices):
                for visible_idx, coupling in enumerate(self.hidden_couplings[hidden_offset]):
                    qc.rzz(float(coupling) * self.center_params[visible_idx], visible_idx, hidden_idx)
                    qc.cry(self.drive_params[visible_idx], visible_idx, hidden_idx)
                qc.rx(float(self.hidden_mixers[hidden_offset]), hidden_idx)

            if len(hidden_indices) >= 2:
                qc.rzz(float(self.hidden_hidden_weight) * 0.75, hidden_indices[0], hidden_indices[1])
            for left, right, coupling in self.visible_pair_couplings:
                if 0 <= left < self.num_visible and 0 <= right < self.num_visible and left != right:
                    pair_expr = 0.5 * float(coupling) * (self.center_params[left] + self.center_params[right])
                    qc.rzz(pair_expr, left, right)
        return qc

    def _parameter_values_from_visible(self, visible_values: Sequence[float]) -> list[float]:
        values = [clamp01(float(v)) for v in visible_values[: self.num_visible]]
        while len(values) < self.num_visible:
            values.append(0.5)

        theta_values = [2.0 * math.asin(math.sqrt(x)) for x in values]
        phase_values = [float(self.input_phase_scale) * math.pi * ((2.0 * x) - 1.0) for x in values]
        center_values = [(2.0 * x) - 1.0 for x in values]
        drive_values = [self.entangle_scale * (0.45 + x) for x in values]
        return theta_values + phase_values + center_values + drive_values

    def _energy_stats_from_parameter_values(self, parameter_values_batch: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
        if not parameter_values_batch:
            return [], []
        if self.backend_mode == "aer_shot":
            circuits = [self.template_circuit] * len(parameter_values_batch)
            observables = [self.observable] * len(parameter_values_batch)
            result = self.estimator.run(circuits, observables, list(parameter_values_batch)).result()
            energies = [float(value) for value in result.values]
            energy_stds: list[float] = []
            for meta in result.metadata:
                variance = float(meta.get("variance", 0.0)) if isinstance(meta, Mapping) else 0.0
                energy_stds.append(math.sqrt(max(variance, 0.0) / max(self.shots, 1)))
            return energies, energy_stds

        pubs = [(self.template_circuit, self.observable, list(params)) for params in parameter_values_batch]
        result = self.estimator.run(pubs, precision=self.precision or None).result()
        energies = [float(pub.data.evs) for pub in result]
        return energies, [0.0 for _ in energies]

    def _build_score_map(
        self,
        *,
        features: Mapping[str, float],
        visible_values: Sequence[float],
        energy: float,
        energy_std: float,
    ) -> dict[str, float | str]:
        energy_raw = float(energy)
        energy_adjusted = float(energy_raw)
        score_raw = _sigmoid((self.score_scale * energy_adjusted) - self.score_offset)
        shot_penalty_applied = 0.0
        uncertainty_proxy = 0.0
        if self.backend_mode == "aer_shot":
            shot_penalty_applied = float(self.shot_penalty / math.sqrt(self.shots))
            score_adjusted = clamp01(score_raw - shot_penalty_applied)
            if energy_std > 0.0:
                raw_std = abs(self.score_scale * score_raw * (1.0 - score_raw)) * float(energy_std)
                shot_std = float(max(raw_std, 0.0))
            else:
                shot_std = math.sqrt((score_adjusted * (1.0 - score_adjusted)) / self.shots)
            uncertainty_proxy = float(max(energy_std, 0.0))
        else:
            score_adjusted = float(score_raw)
            shot_std = float("nan")
        scores: dict[str, float | str] = {
            "q_score": float(score_adjusted),
            "qbm_score_raw": float(score_raw),
            "qbm_score_adjusted": float(score_adjusted),
            "qbm_backend": self.qbm_backend_name,
            "qbm_backend_label": self.qbm_backend_name,
            "qbm_backend_mode": self.backend_mode,
            "qbm_uncertainty_mode": self.uncertainty_mode,
            "qbm_model_family": self.model_family,
            "qbm_model_note": self.model_note,
            "qbm_score_name": self.score_name,
            "qbm_energy": float(energy_adjusted),
            "qbm_energy_raw": float(energy_raw),
            "qbm_energy_adjusted": float(energy_adjusted),
            "qbm_energy_std": float(max(energy_std, 0.0)),
            "qbm_uncertainty_proxy": float(uncertainty_proxy),
            "qbm_shot_penalty_applied": float(shot_penalty_applied),
            "qbm_shot_std": float(shot_std),
            "qbm_circuit_depth": float(self.template_depth),
            "qbm_circuit_size": float(self.template_size),
            "qbm_n_qubits": float(self.template_circuit.num_qubits),
            "qbm_layers": float(self.layers),
            "qbm_hidden_count": float(self.num_hidden),
            "qbm_feature_dim": float(len(visible_values)),
            "qbm_pair_coupling_count": float(len(self.visible_pair_couplings)),
            "qbm_visible_pair_weight_count": float(len(self.visible_pair_weights)),
            "qbm_feature_keys": "|".join(self.feature_keys),
            "qbm_visible_values": "|".join(f"{value:.6f}" for value in visible_values),
            "qbm_required_signal_keys": "|".join(self.REQUIRED_SIGNAL_KEYS),
            "qbm_optional_signal_keys": "|".join(forensic_feature_schema().get("optional_signal_keys", [])),
            "qbm_feature_schema_version": FORENSIC_FEATURE_SCHEMA_VERSION,
            "qbm_feature_set_version": self.feature_set_version,
            "qbm_calibration_version": self.calibration_version,
            "qbm_feature_directionality_mode": FORENSIC_DIRECTIONALITY_MODE,
            "qbm_reference_mode": FORENSIC_REFERENCE_MODE,
            "qbm_expected_context_mode": FORENSIC_EXPECTED_CONTEXT_MODE,
            "qbm_noise_enabled": 1.0 if self.backend_mode == "aer_shot" else 0.0,
            "qbm_measurement_shots": float(self.shots if self.backend_mode == "aer_shot" else 0.0),
            "qbm_template_reused": 1.0,
            "qbm_batch_eval_enabled": 1.0,
        }
        for key, value in features.items():
            scores[f"qbm_{key}"] = float(value)
        return scores

    def score_many(self, signals_batch: Sequence[Mapping[str, Any]]) -> list[dict[str, float | str]]:
        feature_batch: list[dict[str, float]] = []
        visible_batch: list[list[float]] = []
        params_batch: list[list[float]] = []
        for signals in signals_batch:
            features, visible_values = self.visible_values_from_signals(signals)
            feature_batch.append(features)
            visible_batch.append(list(visible_values))
            params_batch.append(self._parameter_values_from_visible(visible_values))

        energies, energy_stds = self._energy_stats_from_parameter_values(params_batch)
        out: list[dict[str, float | str]] = []
        for idx, energy in enumerate(energies):
            out.append(
                self._build_score_map(
                    features=feature_batch[idx],
                    visible_values=visible_batch[idx],
                    energy=energy,
                    energy_std=energy_stds[idx] if idx < len(energy_stds) else 0.0,
                )
            )
        return out

    def score(self, signals: Mapping[str, Any]) -> dict[str, float | str]:
        return self.score_many([signals])[0]


ResidualForensicEnergyValidator = ResidualForensicQiskitScorer
ShallowQiskitQBMScorer = ResidualForensicQiskitScorer
