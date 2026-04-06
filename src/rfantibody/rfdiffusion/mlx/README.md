# MLX Backend for RFdiffusion

MLX inference backend for RFdiffusion antibody design on Apple Silicon. Replaces the PyTorch GPU forward pass with [MLX](https://ml-explore.github.io/mlx/), Apple's native ML framework that compiles directly to Metal shaders.

## Performance

| Metric | PyTorch MPS | MLX | Speedup |
|--------|------------|-----|---------|
| Per-step (L=378) | ~20s | ~8.5s | 2.3x |
| Per-step (L=100) | ~6s | ~1.8s | 3.4x |
| BCMA nanobody (L=163, T=50) | — | 2.3s/step | — |

With optimizations enabled (mixed precision + top-k=64 graph).

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- MLX (`pip install mlx`)
- PyTorch (CPU-only is sufficient)
- RFdiffusion antibody checkpoint: `RFdiffusion_Ab.pt`

## Quick Start

```bash
# Download model weights
curl -L -o models/RFdiffusion_Ab.pt \
  https://files.ipd.uw.edu/pub/RFantibody/RFdiffusion_Ab.pt

# Run nanobody design
PYTHONPATH=src:include/SE3Transformer \
  python scripts/run_mlx_design.py
```

Edit `scripts/run_mlx_design.py` to set your target PDB, framework PDB, and output paths.

## Architecture

The MLX backend is a drop-in replacement for the PyTorch model forward pass. All preprocessing, denoising, and postprocessing remain in PyTorch/numpy.

```
PyTorch (preprocessing)
  → MLXModelWrapper (torch→mlx conversion)
    → MLX RoseTTAFoldModule (Metal GPU)
  → MLXModelWrapper (mlx→torch conversion)
PyTorch (denoising, output)
```

### Key Classes

| Class | File | Role |
|-------|------|------|
| `MLXAbSampler` | `sampler.py` | Drop-in for `AbSampler`, uses MLX model |
| `MLXModelWrapper` | `model_wrapper.py` | Torch↔MLX tensor bridge |
| `RoseTTAFoldModule` | `model.py` | Top-level MLX model |
| `IterativeSimulator` | `track.py` | 4 extra + 32 main + 4 ref IterBlocks |
| `SE3Transformer` | `se3/transformer.py` | Equivariant structure module |

### File Layout

```
mlx/
├── __init__.py              # Public exports
├── model.py                 # RoseTTAFoldModule (top-level)
├── model_wrapper.py         # Torch↔MLX bridge
├── sampler.py               # MLXAbSampler
├── track.py                 # IterBlock, IterativeSimulator, Str2Str
├── attention.py             # BiasedAxialAttention, MSA/Pair attention
├── embeddings.py            # MSA_emb, Extra_emb, Templ_emb, Timestep_emb
├── predictors.py            # Distance, MaskedToken, LDDT, ExpResolved
├── util_module.py           # RBF, graph construction, ComputeAllAtomCoords
├── graph_ops.py             # SimpleGraph, scatter ops
├── rotations.py             # Quaternion/matrix conversions
├── weight_converter.py      # PyTorch state_dict → MLX weights
├── precompute_cg.py         # Clebsch-Gordan coefficient generation
├── test_utils.py            # Testing utilities
└── se3/
    ├── __init__.py
    ├── transformer.py       # SE3Transformer, GraphSequential
    ├── wrapper.py           # SE3TransformerWrapper
    ├── fiber.py             # Fiber type system
    ├── basis.py             # Spherical harmonics, CG coefficients
    └── layers/
        ├── convolution.py   # RadialProfile, ConvSE3
        ├── attention.py     # AttentionSE3
        ├── linear.py        # LinearSE3
        ├── norm.py          # NormSE3
        └── utils.py         # Degree-wise operations
```

## Optimizations

### Mixed Precision

Converts PairStr2Pair attention weights to float16. Inputs are cast to fp16 at entry, back to fp32 at exit. ~8% speedup per IterBlock.

```python
sampler.model.enable_mixed_precision()
```

### Top-k Graph

Replaces full L² graph with k-nearest-neighbor graph for SE3 convolutions. `top_k=64` gives ~2.5x SE3 speedup with minimal quality impact.

```python
sampler.model.set_topk_graph(64)
```

### Both via Config

```yaml
mlx:
  mixed_precision: true
  topk_graph: 64
```

## How Weight Conversion Works

`weight_converter.py` maps PyTorch state dict keys to MLX parameter paths:
- `module.layers.N.` → MLX sequential indexing
- `ParameterDict` keys → nested dict structure
- All 5998 parameters are converted automatically from any RFdiffusion antibody checkpoint

## Differences from PyTorch

- **No dropout**: All dropout is no-op at inference
- **No JIT**: PyTorch `@torch.jit.script` functions replaced with plain Python
- **No e3nn dependency**: Spherical harmonics reimplemented from closed-form polynomials (degrees 0-4). Clebsch-Gordan coefficients precomputed and loaded from `.npz`
- **Lazy evaluation**: MLX evaluates lazily; `mx.eval()` called after each forward pass
- **CPU tensors**: PyTorch side uses CPU tensors; MLX handles Metal GPU internally

## Numerical Notes

Outputs match PyTorch within float32 tolerance (~1e-5 per layer, ~0.01 Å on coordinates after full forward pass). The top-k graph approximation introduces larger differences since it changes which edges participate in message passing.
