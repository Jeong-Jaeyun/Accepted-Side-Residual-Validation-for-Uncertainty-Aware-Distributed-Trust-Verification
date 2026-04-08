# Accepted-Side Residual Validation for Uncertainty-Aware Distributed Trust Verification: Supplementary Material

**IEEE Transactions on Dependable and Secure Computing (TDSC)**  
**Authors**: Jaeyun Jeong and HwaYoung Jeong

This repository contains the complete implementation and supplementary material for an uncertainty-aware multi-stage verification framework for distributed trust systems, evaluated on maritime Automatic Identification System (AIS) data.

## Overview

This work presents a novel framework for robust trust verification in distributed systems that addresses two critical failure modes:

1. **Ambiguous evidence near the decision boundary** that resists confident classification
2. **Statistically plausible but residually inconsistent evidence** that survives upstream filtering but enters the accepted region as false trust

The framework comprises three stages:
- **Stage 1: Uncertainty Filtering** - Multipartite Evidence Validation (MEV) front-end for lightweight statistical decision-making
- **Stage 2: Forensic Validation** - Selective StrongQ escalation for gray-zone candidates
- **Stage 3: Residual Validation** - Accepted-side compatibility gate re-examining already accepted windows against calibrated benign residual structure

Key contributions:
- Formalization of the accepted-region false-trust failure mode
- A three-stage verification architecture with explicit accept-side residual compatibility gating
- Comprehensive experimental validation on public 11-port AIS dataset under adversarial scenarios
- 62.2% reduction in false trust rate (from 0.2065 to 0.0761) while maintaining controlled benign acceptance

- **End-to-end AIS data processing pipeline** for multiple maritime ports
- **Quantum machine learning models** leveraging quantum circuits and hybrid classical-quantum approaches
- **Synthetic attack injection framework** for robustness evaluation (S1, S2, S3 attack patterns)
- **Comprehensive experimental evaluation** across Busan, Antwerp, Cape Town, Los Angeles, and Singapore

## Project Structure

```
├── qbm/                          # Quantum Bit Machine (exact-state backend)
│   ├── model.py                  # QBM architecture for residual validation
│   ├── train.py                  # Training/calibration routines
│   ├── infer.py                  # Inference and compatibility scoring
│   ├── score.py                  # Scoring and evaluation metrics
│   ├── backends/                 # Backend implementations
│   │   ├── exact_state.py        # Exact-state quantum backend
│   │   ├── aer_shot.py           # Noisy shot-based backend
│   │   └── simulator.py          # State vector simulator
│   └── __init__.py
│
├── preprocess/                   # Data preprocessing pipeline
│   ├── load_and_map_schema.py    # AIS schema mapping and loading
│   ├── filter_by_port.py         # Geographic port-based filtering
│   ├── windowing.py              # Temporal windowing of trajectories
│   ├── grid_mapping.py           # Grid discretization for geographic data
│   ├── feature_extraction.py     # Feature engineering from AIS records
│   ├── discretize_quantiles.py   # Quantile-based discretization
│   ├── encode_onehot.py          # One-hot encoding for categorical features
│   └── residual_construction.py  # Forensic residual feature engineering
│
├── experiments/                  # Experimental scripts and analysis
│   ├── run_pipeline.py           # End-to-end data preprocessing pipeline
│   ├── mev_calibration.py        # MEV front-end threshold calibration
│   ├── strongq_evaluation.py     # StrongQ forensic stage evaluation
│   ├── residual_validation.py    # Accepted-side residual validator
│   ├── full_system_evaluation.py # Complete three-stage pipeline
│   ├── attack_injection.py       # Adversarial scenario generation (A4, A4P, A5)
│   ├── cross_port_evaluation.py  # Multi-port generalization testing
│   ├── build_figures.py          # Figure generation for paper
│   ├── build_tables.py           # Table generation for results
│   └── robustness_analysis.py    # Shot-sensitivity and noise-injection analysis
│
├── blockchain_net/               # Maritime blockchain network simulator
│   └── simulator.py              # Network environment for integration
│
├── policy/                       # Trust policy engine
│   ├── engine.py                 # Policy execution and integration
│   └── policy_table.yaml         # Policy definitions and triggers
│
├── configs/                      # Configuration files
│   ├── default.yaml              # Default configuration template
│   ├── datasets/                 # Per-port dataset configurations
│   │   ├── busan.yaml            # Busan (primary calibration)
│   │   ├── antwerp.yaml          # Antwerp (cross-port validation)
│   │   ├── cape_town.yaml        # Cape Town
│   │   ├── los_angeles.yaml      # Los Angeles
│   │   └── singapore.yaml        # Singapore
│   └── experiments/              # Per-experiment configurations
│       ├── mev_calibration.yaml
│       ├── strongq_eval.yaml
│       └── full_system.yaml
│
├── data/                         # Data organization
│   ├── raw/                      # Raw AIS data (11 ports)
│   └── processed/                # Processed feature windows
│
├── results/                      # Experimental results
│   ├── tables/                   # Output tables (CSV, TeX, JSON)
│   ├── figures/                  # Generated visualizations
│   ├── metrics/                  # Performance metrics (FTR, ASR, TCP)
│   └── per_port/                 # Per-port analysis results
│
├── artifacts/                    # Calibration and model artifacts
│   ├── mev_calibration/          # MEV threshold parameters
│   ├── strongq_witness/          # StrongQ witness definitions
│   ├── residual_calibration/     # Accepted-side residual thresholds
│   └── qbm_backends/             # QBM backend state snapshots
│
├── utils/                        # Utility modules
│   ├── geo.py                    # Geographic calculations
│   ├── io.py                     # I/O operations for parquet/CSV
│   ├── logging.py                # Logging infrastructure
│   ├── time.py                   # Temporal utilities
│   ├── metrics.py                # Performance metric computation
│   └── random.py                 # Reproducibility seeding
│
├── ports/                        # Port definitions
│   └── ports.yaml                # Geographic metadata and bounding boxes
│
├── Accepted-Side Residual Validation for Uncertainty-Aware Distributed Trust Verification.tex
│                                  # Original LaTeX manuscript
│
└── requirements.txt              # Python dependencies
```

## Dependencies

### Core Runtime Dependencies
- **numpy** (≥1.20)
- **pandas** (≥1.5)
- **PyYAML** (≥6.0)

### Optional Dependencies
- **matplotlib** (≥3.5): Required for figure generation
- **pyarrow** (≥10.0): Recommended for efficient parquet I/O
- **qiskit** (≥1.2): Required for quantum backend experiments

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Optional: Install quantum backend support
pip install qiskit>=1.2
```

## Quick Start

### 1. Data Preprocessing

Process raw AIS data into feature-engineered evidence windows:

```bash
# Preprocess Busan dataset (primary calibration port)
python experiments/run_pipeline.py \
  --config configs/datasets/busan.yaml \
  --ports ports/ports.yaml

# Preprocess cross-port datasets
for port in antwerp cape_town los_angeles singapore; do
  python experiments/run_pipeline.py \
    --config configs/datasets/${port}.yaml \
    --ports ports/ports.yaml
done
```

**Pipeline stages:**
1. Load and sanitize raw AIS records
2. Filter by port geographic bounds (optional polygon-based filtering)
3. Create temporal windows (default: 5-minute aggregate windows)
4. Map coordinates to grid cells for spatial discretization
5. Extract numerical and behavioral features
6. Apply quantile-based discretization
7. Encode categorical features (one-hot)
8. Save processed parquet + calibration artifacts

### 2. MEV Front-End Calibration

Calibrate the Multipartite Evidence Validation threshold on benign data:

```bash
python experiments/mev_calibration.py \
  --config configs/experiments/mev_calibration.yaml \
  --benign-data data/processed/busan/benign_windows.parquet \
  --output artifacts/mev_calibration/
```

### 3. System Evaluation (Three-Stage Pipeline)

Evaluate the complete verification framework:

```bash
# Full system with all stages enabled
python experiments/full_system_evaluation.py \
  --config configs/experiments/full_system.yaml \
  --attack-scenarios A0 A4 A4P A5 \
  --backend exact_state
```

Stages evaluated:
- **S3**: Base MEV + lightweight statistical checks
- **S3 + StrongQ**: MEV with forensic gray-zone escalation
- **Full System**: S3 + StrongQ + accepted-side residual validation

### 4. Cross-Port Generalization Testing

Validate framework generalization across heterogeneous ports:

```bash
python experiments/cross_port_evaluation.py \
  --calibration-port busan \
  --test-ports antwerp cape_town los_angeles singapore \
  --calibration-artifacts artifacts/mev_calibration/busan
```

### 5. Robustness Analysis

Test backend stability under noise and quantum shot variations:

```bash
# Exact-state vs. shot-based backend comparison
python experiments/robustness_analysis.py \
  --backends exact_state aer_shot \
  --noise-injection true \
  --shot-limits 1024 2048 4096 8192
```

### 6. Generate Paper Results

Reproduce all paper tables and figures:

```bash
# Generate main paper visualizations
python experiments/build_figures.py --output results/figures/

# Generate result tables
python experiments/build_tables.py \
  --metrics FTR ASR TCP latency \
  --output results/tables/
```

Output locations:
- **Tables**: `results/tables/`
- **Figures**: `results/figures/`
- **Metrics**: `results/metrics/`

## Minimal example (fast test)
python experiments/full_system_evaluation.py \
  --config configs/experiments/full_system.yaml \
  --attack-scenarios A4P \
  --backend exact_state

## Configuration System

The project uses hierarchical YAML configurations:

1. **Default config** (`configs/default.yaml`): Base configuration template
2. **Custom override**: User-provided config via `--config` flag
3. **Deep merge**: Custom config merged with defaults

### Key Configuration Parameters

```yaml
project:
  seed: 42                    # Random seed for reproducibility
  port: "busan"               # Target port identifier
  raw_path: "data/raw/*.csv"  # Raw data input path
  processed_dir: "data/processed/{port}"  # Processed output directory
  artifacts_dir: "artifacts"  # Model artifacts directory

schema_mapping:
  ts: "timestamp"             # Column names for AIS records
  mmsi: "MMSI"
  lat: "Latitude"
  lon: "Longitude"
  sog: "speed"
  cog: "Cog"

time:
  timezone: "Asia/Seoul"      # Timezone for temporal alignment
  dt_minutes: 5               # Temporal window duration

port_filter:
  use_polygon: false          # true for polygon-based filtering
  bbox_override: null         # Geographic bounding box [min_lon, min_lat, max_lon, max_lat]

grid:
  nx: 10                      # Grid columns
  ny: 10                      # Grid rows
```

## Datasets

## Data Setup

1. Download AIS dataset from:
   https://data.mendeley.com/datasets/r37vwd493d/1

2. Place files under:
   data/raw/{port}_anonymized.csv

3. Expected format:
   - MMSI
   - timestamp
   - latitude / longitude
   - speed / heading
### Analyzed Ports

| Port | Country | Primary Use | Status |
|------|---------|-------------|--------|
| **Busan** | South Korea | Calibration & main experiments | Full analysis |
| **Antwerp** | Belgium | Cross-port validation | Cross-port test |
| **Cape Town** | South Africa | Geographic diversity | Cross-port test |
| **Los Angeles** | USA | Long-range shipping | Cross-port test |
| **Singapore** | Singapore | Straits congestion | Cross-port test |

### Attack Scenarios

| Scenario | Type | Description | Detection Difficulty |
|----------|------|-------------|----------------------|
| **A0** | Benign | Normal traffic baseline | — |
| **A4** | Boundary Manipulation | Direct near-threshold attacks | Medium |
| **A4P** | Near-Replay | Perturbed historical windows | High |
| **A5** | Node Compromise | Coordinated context/policy manipulation | High |

**A4P and A5** are primary evaluation scenarios, as they represent statistically plausible attacks where base MEV scores remain close to the operating threshold but residual structure deviates from benign patterns.

### Data Format

Raw data format: CSV with AIS record fields (MMSI, timestamp, position, speed, heading, etc.)

**Data location**: `data/raw/{port}_anonymized.csv`

**Processed data**: `data/processed/{port}/` (Parquet format with preprocessed features)

## Verification Framework Architecture

### Stage 1: Uncertainty Filtering (MEV Front-End)

**Input**: Evidence window $\mathcal{E} = \{e_1, e_2, ..., e_n\}$ from distributed nodes

**Processing**:
- Standardize node features against benign calibration statistics
- Extract compact explanation tuple and cross-node consistency metrics
- Compute MEV soft trust score $S$ from:
  - Trust consistency ($C_{\text{var}}$)
  - Mean explanation similarity ($S_{\text{sim}}$)
  - Dominant-context agreement ($C_{\text{ctx}}$)
  - Mean node trust ($\bar{T}$)

**Decision**:
```
if S > τ + δ:
    ACCEPT   # High confidence
elif S < τ - δ:
    REJECT   # High confidence
else:
    GRAY_ZONE(S)  # Ambiguous; escalate if risk triggers met
```

### Stage 2: Forensic Validation (StrongQ)

**Activation Condition**:
- Window in upper-half gray zone: $|S - τ| < δ$ AND $S ≥ τ$
- Risk trigger satisfied: $\mathcal{F}(r) = 1$

**Witness Function**:
- Aggregates benign inconsistency, node instability, pairwise structural inconsistency, gray-zone pressure
- Fixed thresholding at $θ_{SQ}$
- Selective invocation: only high-risk uncertainties escalate

**Benefit**: Addresses boundary ambiguity without universally expanding forensic cost

### Stage 3: Residual Validation (Accepted-Side Compatibility Gate)

**Input**: Windows already accepted by MEV+StrongQ base path

**Residual Feature Vector** ($r(\mathcal{E}) \in \mathbb{R}^8$):
1. Trust-similarity alignment mismatch
2. Pairwise overlap dispersion
3. Contextual deviation from benign baseline
4. Policy-gray contradiction
5. Benign structural coherence
6. Manifold deviation
7. Temporal inconsistency
8. Witness disagreement

**Compatibility Scoring**:
- Shallow exact-state quantum circuit encodes residuals
- Outputs compatibility score $Q(\mathcal{E})$
- Calibrated via lower-tail quantile on benign accepted windows:

$$\theta_Q = Q_q(Q_b^+), \quad q = 0.01$$

**Veto Decision**:
- If $Q(\mathcal{E}) < \theta_Q$: veto (reject accepted window)
- If $Q(\mathcal{E}) \geq \theta_Q$: allow (maintain acceptance)

**Purpose**: Detects residually inconsistent windows that remain statistically plausible but structurally suspicious

## Experimental Results

### Output Organization

**Processed Results**:
- `results/tables/paper_main/`: Main paper tables (CSV, JSON, TeX, Markdown)
- `results/tables/paper_appendix/`: Appendix tables
- `results/figures/ieee_tdsc_pack/`: Publication-ready figures

**Raw Results**:
- `results/tables/`: Raw experiment outputs (JSON metadata, CSV)
- `results/figures/`: Figure source data

**Per-Port State**:
- `results/{antwerp,busan,cape_town,los_angeles,singapore}/`: Per-port summaries for cross-port analysis

### Key Metrics

- **Detection Rate (Recall)**: True positive rate for anomaly detection
## Evaluation Metrics

### Primary Metrics

- **False Trust Rate (FTR)**: Fraction of adversarial windows accepted by verifier
- **Adversarial Success Rate (ASR)**: Attack objective attainment rate
- **Trust Collapse Probability (TCP)**: Fraction exceeding safety threshold

### Secondary Metrics

- **Gray-zone Frequency**: Proportion of windows in uncertainty regime
- **StrongQ Invocation Rate**: Forensic stage activation frequency
- **Mean Verification Latency**: Per-window processing time

### Key Results

| Configuration | A4P FTR | A5 FTR | TCP increase |
|---------------|---------|--------|--------------|
| S3 (MEV baseline) | 0.2065 | 0.0870 | — |
| S3 + StrongQ | 0.1739 | 0.0815 | Minimal |
| Full (+ Residual) | **0.0761** | **0.0750** | +0.0134 (*benign→TCP*) |

**Interpretation**: Residual validation reduces false trust by 62.2% (A4P) with tolerable benign-acceptance trade-off.


## Configuration System

Hierarchical YAML configuration with default + custom override:

```yaml
# configs/default.yaml (template)
project:
  seed: 42
  port: "busan"
  raw_path: "data/raw/Busan_anonymized.csv"
  processed_dir: "data/processed/busan"
  artifacts_dir: "artifacts"

mev:
  threshold_tau: 0.9372              # Calibrated MEV threshold
  gray_zone_width_delta: 0.001       # Fixed gray-zone margin
  risk_trigger_floor: 0.15           # Policy risk threshold

strongq:
  witness_threshold: 0.75             # Fixed StrongQ operating point
  enable_escalation: true             # Toggle gray-zone escalation

residual:
  backend: "exact_state"              # exact_state | aer_shot | simulator
  quantile_q: 0.01                    # Lower-tail benign quantile for θ_Q
  enable_veto: true                   # Toggle accepted-side residual check

grid:
  nx: 10                              # Grid columns
  ny: 10                              # Grid rows

time:
  timezone: "UTC"
  window_minutes: 5                   # Temporal aggregation window
  t0: null                            # Null = auto-detect from data
```

## Quantum Backend Details

### Exact-State Instantiation (Main)

- **Visible qubits**: 8 (encode residual feature vector)
- **Hidden qubits**: 2 (entanglement and expressivity)
- **Circuit depth**: Shallow variational with single block
- **Observable**: Stabilizer-compatible measurements
- **Advantage**: Deterministic, backend-consistent decision semantics
- **Use case**: Accepted-side residual validation (post-MEV filtering)

### Aer Shot-Based Backend (Robustness)

- **Shot count**: 1024, 2048, 4096, 8192
- **Noise model**: Optional depolarization channels
- **Advantage**: Realistic quantum hardware simulation
- **Use case**: Stress-testing under noise and sampling variance

### Instantiation Template

```python
from qbm.backend.exact_state import ExactStateBackend
from qbm.model import ResidualValidator

# Load residual features
residuals = extract_residuals(mev_scores, node_features)

# Initialize backend
backend = ExactStateBackend(
    n_visible=8,
    n_hidden=2,
    observable="Z0*Z1",  # Example observable
)

# Score compatibility
compatibility_score = backend.score(residuals)

# Apply veto threshold
if compatibility_score < theta_Q:
    decision = REJECT
else:
    decision = ACCEPT
```

## Reproducibility

### Seeding and Random State

```python
from utils.random import seed_everything
seed_everything(seed=42)
```

All experiments use deterministic seeding via configuration files. Results match reference outputs modulo floating-point precision.

### Results Reconstruction

```bash
# Step-by-step reproduction
python experiments/run_pipeline.py --config configs/datasets/busan.yaml

python experiments/mev_calibration.py \
  --config configs/experiments/mev_calibration.yaml

python experiments/full_system_evaluation.py \
  --config configs/experiments/full_system.yaml \
  --attack-scenarios A0 A4 A4P A5
```

Expected outputs documented in [results/README.md](results/README.md).

## Logging and Debugging

### Logging Setup

```python
from utils.logging import setup_logging

logger = setup_logging(
    level="INFO",
    logfile="logs/verification.log"
)
```

Log files capture:
- Data pipeline progress
- MEV calibration diagnostics
- Gray-zone frequency and risk triggers
- StrongQ invocation patterns
- Residual validation veto decisions
- Latency breakdowns

### Common Debugging Tasks

1. **Check MEV threshold calibration**:
   ```bash
   python -c "from config import load_config; cfg = load_config('configs/default.yaml'); print('MEV τ:', cfg['mev']['threshold_tau'])"
   ```

2. **Verify residual features**:
   ```bash
   python experiments/residual_validation.py --validate-features --output debug/
   ```

3. **Inspect StrongQ witness**:
   ```bash
   python experiments/strongq_evaluation.py --dump-witness artifacts/strongq_witness/busan.json
   ```

4. **Latency profiling**:
   ```bash
   python experiments/full_system_evaluation.py --profile --output profiling/
   ```

## Known Limitations and Future Work

1. **Quantum Hardware Constraints**
   - Currently using simulators; real quantum hardware introduces additional noise
   - Shot-based analysis reveals scaling challenges under noise injection

2. **Port Generalization**
   - Residual calibration is Busan-specific in main study
   - Cross-port results demonstrate MEV+StrongQ generalization but not residual thresholds
   - Future work: port-adaptive calibration strategies

3. **Batch Processing**
   - Framework processes evidence windows independently
   - Temporal correlation across windows not explicitly modeled
   - Opportunity for RNN/Transformer-based sequential models

4. **Class Imbalance**
   - Adversarial scenarios are synthetic; real attack distribution unknown
   - Strong assumption: attacks remain statistically plausible


## Reproducibility Checklist

- Python version: 3.9+
- Random seed: Fixed (default: 42)
- Deterministic backend: exact_state (recommended)
- Hardware: CPU/GPU optional (no dependency on GPU)
- Expected runtime: ~X hours for full pipeline

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- AIS data providers (Mendeley Data, port authorities)
- Qiskit development team for quantum circuit framework
- IEEE TDSC editorial board for feedback and guidance

---

**Note:** This is supplementary material accompanying the peer-reviewed publication. All code has been tested on Python 3.9+ with dependencies specified in `requirements.txt`. For exact reproducibility, use the locked requirements or create an isolated virtual environment.
