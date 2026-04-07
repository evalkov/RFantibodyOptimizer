# RFantibodyOptimizer

Native macOS app for nanobody design on Apple Silicon. Runs the full RFdiffusion → ProteinMPNN → RF2 pipeline locally via MLX, with a SwiftUI interface for configuring campaigns, monitoring progress in real time, and analyzing results with an integrated 3D protein viewer.

![macOS 15+](https://img.shields.io/badge/macOS-15%2B-blue)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-orange)
![Python 3.10](https://img.shields.io/badge/Python-3.10-green)

## Install from DMG

Download `RFantibodyOptimizer.dmg`, open it, and drag the app to `/Applications`. The DMG is fully self-contained (~1.2 GB) — Python runtime, ML models, and all dependencies are bundled inside the app. No setup required.

## Requirements

### Running the pre-built DMG

- **macOS 15+** (Sequoia or later)
- **Apple Silicon** Mac (M1, M2, M3, or M4)

### Building from source

- Everything above, plus:
- **Xcode 16+**
- **Python 3.10** with [uv](https://github.com/astral-sh/uv) package manager
- ~750 MB disk space for model checkpoints
- ~1.4 GB disk space for the Python virtual environment

## Building from Source

### 1. Clone the repository

```bash
git clone https://github.com/evalkov/RFantibodyOptimizer.git
cd RFantibodyOptimizer
```

### 2. Download model weights

Three checkpoints are required (~748 MB total):

```bash
mkdir -p models
curl -L -o models/RFdiffusion_Ab.pt    https://files.ipd.uw.edu/pub/RFantibody/RFdiffusion_Ab.pt
curl -L -o models/ProteinMPNN_v48_noise_0.2.pt https://files.ipd.uw.edu/pub/RFantibody/ProteinMPNN_v48_noise_0.2.pt
curl -L -o models/RF2_ab.pt             https://files.ipd.uw.edu/pub/RFantibody/RF2_ab.pt
```

| Model | File | Size | Purpose |
|-------|------|------|---------|
| RFdiffusion | `RFdiffusion_Ab.pt` | 461 MB | Backbone generation |
| ProteinMPNN | `ProteinMPNN_v48_noise_0.2.pt` | 6.4 MB | CDR sequence design |
| RF2 | `RF2_ab.pt` | 281 MB | Structure validation |

### 3. Set up the Python environment

```bash
uv venv pilot_mps/.venv --python 3.10
source pilot_mps/.venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install mlx mlx-metal
uv pip install omegaconf hydra-core biopython biotite scipy numpy
```

### 4. Build and run (development)

Open the Xcode project and build:

```bash
xcodebuild -project RFantibodyOptimizer.xcodeproj \
  -scheme RFantibodyOptimizer -configuration Debug \
  -derivedDataPath /tmp/RFantibodyOptimizer-build build
```

Launch:

```bash
open /tmp/RFantibodyOptimizer-build/Build/Products/Debug/RFantibodyOptimizer.app
```

Or open `RFantibodyOptimizer.xcodeproj` in Xcode and press Run.

### 5. Build a distributable DMG

The `package_dmg.sh` script builds a self-contained `.app` bundle with everything embedded and wraps it in a DMG:

```bash
./package_dmg.sh
```

This will:
1. Build the Release configuration via `xcodebuild`
2. Bundle the standalone Python 3.10 runtime and all pip packages into `Contents/Resources/python/`
3. Bundle the pipeline source (`src/`, `include/`, `scripts/`) into `Contents/Resources/`
4. Copy model checkpoints into `Contents/Resources/models/`
5. Ad-hoc code sign the app
6. Create `RFantibodyOptimizer.dmg` in the repo root

The script expects `models/` and `pilot_mps/` to exist (either as real directories or symlinks that resolve). The resulting DMG is ~1.2 GB compressed and requires no external dependencies to run.

## Usage

### GUI App

1. **Target PDB** — select the antigen structure to design against
2. **Framework PDB** — nanobody framework scaffold (NbBCII10 bundled as default)
3. **CDR Loops** — toggle H1/H2/H3 on/off and set length ranges with the dual sliders
4. **Diffusion settings** — number of designs, timesteps, mode, template scheme
5. **Hotspot residues** — optional target residues to bias binding (e.g. `A100,A103`)
6. **MPNN** — sequence design temperature and number of sequences per backbone
7. **RF2** — structure validation with recycling convergence

Press **Start Design Campaign** to run. The app shows live progress with a 3D protein viewer, then a sortable results table with per-design metrics (pLDDT, PAE, iPAE, P(bind), CDR RMSD).

After a campaign completes, two export buttons appear in the results toolbar:
- **Export TSV** (table icon) — saves all design metrics as a tab-separated file for downstream analysis
- **Export Archive** (archive icon) — bundles all PDB files (backbone, sequence, validated) into a `.tgz` archive

### Command Line

Run the pipeline directly without the GUI:

```bash
PYTHONPATH=src:include/SE3Transformer \
  pilot_mps/.venv/bin/python scripts/run_mlx_design.py
```

#### Environment Variables

| Variable | Options | Default | Description |
|----------|---------|---------|-------------|
| `MLX_MODE` | `full`, `fast`, `draft` | `full` | Speed/quality tradeoff |
| `MLX_DIFFUSION_T` | integer | `50` | Diffusion timesteps |
| `NUM_BACKBONES` | integer | `1` | Number of designs |
| `SKIP_MPNN` | `0`, `1` | `0` | Skip sequence design |
| `SKIP_RF2` | `0`, `1` | `0` | Skip structure validation |
| `MPNN_NUM_SEQS` | integer | `1` | Sequences per backbone |
| `MPNN_TEMP` | float | `0.1` | MPNN sampling temperature |
| `RF2_RECYCLES` | integer | `10` | Max RF2 recycling iterations |
| `RF2_THRESHOLD` | float | `0.5` | Convergence threshold (Angstroms) |

#### Examples

```bash
# Fast mode, 5 designs
MLX_MODE=fast NUM_BACKBONES=5 \
  PYTHONPATH=src:include/SE3Transformer \
  pilot_mps/.venv/bin/python scripts/run_mlx_design.py

# Draft mode, skip RF2 validation
MLX_MODE=draft SKIP_RF2=1 \
  PYTHONPATH=src:include/SE3Transformer \
  pilot_mps/.venv/bin/python scripts/run_mlx_design.py
```

## Performance

Benchmarked on M4 Max (128 GB), nanobody L~163 residues:

| Mode | RFdiffusion | MPNN | RF2 | Total per design |
|------|-------------|------|-----|------------------|
| Full (T=50) | ~2.1 s/step | ~1 s | ~35 s | ~140 s |
| Fast (T=25) | ~1.9 s/step | ~1 s | ~35 s | ~85 s |
| Draft (T=15) | ~1.3 s/step | ~1 s | ~35 s | ~55 s |

## Pipeline Architecture

```
┌─────────────────────────────────────────────────┐
│  RFantibodyOptimizer (SwiftUI)                  │
│  ┌───────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ ConfigPanel│  │ Progress │  │ 3D Viewer    │ │
│  │           │  │ View     │  │ (3Dmol.js)   │ │
│  └─────┬─────┘  └────▲─────┘  └──────────────┘ │
│        │ JSON stdin   │ JSON-lines stderr        │
└────────┼──────────────┼──────────────────────────┘
         │              │
┌────────▼──────────────┼──────────────────────────┐
│  design_service.py (Python subprocess)           │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │RFdiffusion│─▶│ProteinMPNN│─▶│     RF2      │  │
│  │  (MLX)   │  │  (MLX)   │  │    (MLX)     │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
│        ▼              ▼              ▼           │
│    backbone.pdb   sequence.pdb   validated.pdb   │
└──────────────────────────────────────────────────┘
         │              │              │
         └──────── Apple Metal GPU ────┘
```

## Project Structure

```
RFantibodyOptimizer/
├── RFantibodyOptimizer/          # SwiftUI macOS app
│   ├── App.swift
│   ├── Models/                   # DesignCampaign, DesignConfig, NanobodyDesign
│   ├── Views/                    # ConfigPanel, ContentView, ProteinViewer, ...
│   ├── Services/                 # PipelineRunner, PDBParser
│   └── Resources/                # Bundled NbBCII10 framework PDB
├── src/rfantibody/               # Python pipeline
│   ├── rfdiffusion/              # RFdiffusion (PyTorch + MLX backend)
│   │   ├── mlx/                  # MLX model, sampler, SE3, weight converter
│   │   └── inference/            # AbSampler, preprocessing, denoising
│   ├── proteinmpnn/              # ProteinMPNN (PyTorch + MLX backend)
│   │   └── mlx/                  # MLX model wrapper, weight converter
│   ├── rf2/                      # RF2 structure prediction
│   │   ├── mlx/                  # MLX predictor, model, weight converter
│   │   ├── modules/              # Model runner, preprocessing, RMSD
│   │   └── network/              # PyTorch network architecture
│   └── util/                     # PDB I/O, pose utilities
├── include/SE3Transformer/       # Equivariant SE3 transformer layers
├── scripts/
│   ├── design_service.py         # JSON-lines bridge for Swift app
│   ├── run_mlx_design.py         # Standalone CLI pipeline
│   └── config/inference/         # Hydra/OmegaConf YAML configs
├── models/                       # Model checkpoints (not tracked in git)
└── pilot_mps/.venv/              # Python virtual environment
```

## License

Research use only. See individual component licenses for RFdiffusion, ProteinMPNN, and RF2.

## Citation

If you use this software, please cite:

```
Watson et al. (2023). De novo design of protein structure and function with RFdiffusion. Nature.
Dauparas et al. (2022). Robust deep learning–based protein sequence design using ProteinMPNN. Science.
Ruffolo et al. (2023). Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies. Nature Communications.
```
