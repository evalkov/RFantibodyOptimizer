#!/usr/bin/env python3
"""
Run full nanobody design pipeline on Apple Silicon using MLX.

Pipeline: RFdiffusion (backbone) → ProteinMPNN (sequence) → RF2 (validation)

Usage:
    PYTHONPATH=src:include/SE3Transformer pilot_mps/.venv/bin/python scripts/run_mlx_design.py

Environment variables:
    MLX_MODE=full|fast|draft   RFdiffusion speed/quality (default: full)
    NUM_BACKBONES=1            Number of backbone designs (batched on GPU)
    SKIP_MPNN=1                Skip MPNN sequence design
    SKIP_RF2=1                 Skip RF2 structure validation
    MPNN_NUM_SEQS=1            Sequences per backbone (default: 1)
    MPNN_TEMP=0.1              MPNN sampling temperature (default: 0.1)
    RF2_RECYCLES=10            Max RF2 recycling iterations (default: 10)
    RF2_THRESHOLD=0.5          Ca RMSD convergence threshold in A (default: 0.5)

Checkpoints (download to models/):
    RFdiffusion_Ab.pt          https://files.ipd.uw.edu/pub/RFantibody/RFdiffusion_Ab.pt
    ProteinMPNN_v48_noise_0.2.pt  https://files.ipd.uw.edu/pub/RFantibody/ProteinMPNN_v48_noise_0.2.pt
    RF2_ab.pt                  https://files.ipd.uw.edu/pub/RFantibody/RF2_ab.pt
"""
import os
import pickle
import random
import sys
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'include', 'SE3Transformer'))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("mlx_design")

import torch
import numpy as np
from omegaconf import OmegaConf

from rfantibody.rfdiffusion.chemical import num2aa
from rfantibody.rfdiffusion.util import generate_Cbeta
from rfantibody.util.io import ab_write_pdblines

conversion = 'ARNDCQEGHILKMFPSTWYV-'

# Load configs
script_dir = os.path.dirname(os.path.abspath(__file__))
base_cfg = OmegaConf.load(os.path.join(script_dir, 'config', 'inference', 'base.yaml'))
ab_cfg = OmegaConf.load(os.path.join(script_dir, 'config', 'inference', 'antibody.yaml'))
if 'defaults' in ab_cfg:
    del ab_cfg['defaults']
conf = OmegaConf.merge(base_cfg, ab_cfg)

# Paths
target_pdb = os.path.expanduser('~/Desktop/bcma.pdb')
framework_pdb = os.path.join(script_dir, '..', 'test', 'rfdiffusion', 'inputs_for_test', 'h-NbBCII10.pdb')
output_dir = os.path.expanduser('~/Desktop')
ckpt_path = os.path.join(script_dir, '..', 'models', 'RFdiffusion_Ab.pt')

conf.antibody.target_pdb = target_pdb
conf.antibody.framework_pdb = framework_pdb
conf.inference.output_prefix = os.path.join(output_dir, 'bcma_nb_mlx')
conf.inference.num_designs = 1
conf.inference.ckpt_override_path = ckpt_path
conf.inference.write_trajectory = False
conf.inference.deterministic = True
conf.diffuser.T = int(os.environ.get('MLX_DIFFUSION_T', '50'))

# Nanobody design (heavy chain CDR loops only)
conf.antibody.design_loops = ['H1:', 'H2:', 'H3:']

# --- Mode selection ---
# "full"  = highest quality (2.1s/step)
# "fast"  = balanced speed/quality (1.9s/step, SE3 stride=4)
# "draft" = maximum speed for screening (1.3s/step, SE3 stride=4, 24 blocks)
MODE = os.environ.get('MLX_MODE', 'full')

MODE_CONFIGS = {
    'full':  dict(top_k=64, se3_stride=1, n_main=32),
    'fast':  dict(top_k=64, se3_stride=4, n_main=32),
    'draft': dict(top_k=64, se3_stride=4, n_main=24),
}

if MODE not in MODE_CONFIGS:
    print(f"Unknown mode '{MODE}', using 'full'. Options: full, fast, draft")
    MODE = 'full'

mode_cfg = MODE_CONFIGS[MODE]

def make_deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

print("=" * 60)
print("  MLX Nanobody Design against BCMA")
print("=" * 60)
print(f"  Target:    {target_pdb}")
print(f"  Framework: {framework_pdb}")
print(f"  Output:    {conf.inference.output_prefix}")
print(f"  T={conf.diffuser.T} diffusion steps")
print(f"  Mode:      {MODE} (top_k={mode_cfg['top_k']}, "
      f"se3_stride={mode_cfg['se3_stride']}, "
      f"n_main={mode_cfg['n_main']})")
print(f"  Set MLX_MODE=fast|draft for faster inference")
print()

# --- Initialize MLX Sampler ---
t_start = time.time()
print("Initializing MLX sampler...")
from rfantibody.rfdiffusion.mlx.sampler import MLXAbSampler
sampler = MLXAbSampler(conf)
t_init = time.time()
print(f"  Init time: {t_init - t_start:.1f}s")

# Enable optimizations
sampler.model.enable_mixed_precision()
if os.environ.get('MLX_FUSED', '1') == '1':
    sampler.model.enable_fused_kernels()
sampler.model.set_topk_graph(mode_cfg['top_k'])
sampler.model.set_se3_stride(mode_cfg['se3_stride'])
sampler.model.set_n_main_block(mode_cfg['n_main'])
eval_stride = int(os.environ.get('MLX_EVAL_STRIDE', '8'))
sampler.model.set_eval_stride(eval_stride)
cache_enabled = os.environ.get('CACHE_ENABLED', '1') == '1'
cache_threshold = float(os.environ.get('CACHE_THRESHOLD', '0.15'))
print(f"  Optimizations: fp16 pair, fused Metal SE3, SDPA attention, "
      f"top_k={mode_cfg['top_k']}, eval_stride={eval_stride}")
if cache_enabled:
    print(f"  Adaptive step cache: threshold={cache_threshold}, "
          f"warmup={os.environ.get('CACHE_WARMUP', '3')}")

# --- Run design ---
NUM_BACKBONES = int(os.environ.get('NUM_BACKBONES', '1'))

t_design_start = time.time()

if NUM_BACKBONES > 1:
    # ── Batched design: B backbones in one model call ──
    from rfantibody.rfdiffusion.mlx.batched_design import BatchedDesigner

    print(f"\nRunning batched diffusion (B={NUM_BACKBONES})...")
    designer = BatchedDesigner(sampler, batch_size=NUM_BACKBONES)
    design_results = designer.run(seed=0)
    step_times = design_results[0].step_times
    L = designer.L
else:
    # ── Single design (original path) ──
    print("\nRunning diffusion...")
    make_deterministic(0)

    x_init, seq_init = sampler.sample_init()
    x_t = torch.clone(x_init)
    seq_t = torch.clone(seq_init)
    L = x_init.shape[0] if x_init.ndim == 3 else x_init.shape[1]

    denoised_xyz_list = []
    px0_xyz_list = []
    seq_list = []
    plddt_list = []
    step_times = []

    final_step = getattr(conf.inference, 'final_step', 1)
    for t in range(int(sampler.t_step_input), final_step - 1, -1):
        step_t0 = time.time()
        px0, x_t, seq_t, tors_t, plddt = sampler.sample_step(
            t=t, seq_t=seq_t, x_t=x_t, seq_init=seq_init,
            final_step=final_step)
        step_dt = time.time() - step_t0
        step_times.append(step_dt)
        px0_xyz_list.append(px0)
        denoised_xyz_list.append(x_t)
        seq_list.append(seq_t)
        plddt_list.append(plddt[0])
        print(f"  t={t:>3d}: {step_dt*1000:>7.0f}ms")

    # Wrap as single DesignResult for unified downstream handling
    from rfantibody.rfdiffusion.mlx.batched_design import DesignResult, DesignState
    _state = DesignState(x_t, seq_t, seq_init, sampler.denoiser)
    _state.px0_stack = px0_xyz_list
    _state.xyz_stack = denoised_xyz_list
    _state.seq_stack = seq_list
    _state.plddt_stack = plddt_list
    design_results = [DesignResult(_state, step_times)]

t_design_end = time.time()
n_steps = len(step_times)
total_design = t_design_end - t_design_start
avg_step = np.mean(step_times)

# Report adaptive step cache statistics
cache_hits = getattr(sampler, '_cache_hits', 0)
cache_misses = getattr(sampler, '_cache_misses', 0)
cache_total = cache_hits + cache_misses

print()
print("=" * 60)
print(f"  RESULTS")
print("=" * 60)
print(f"  Protein length:   L={L}")
print(f"  Backbones:        {NUM_BACKBONES}")
print(f"  Diffusion steps:  {n_steps}")
print(f"  Total design:     {total_design:.1f}s")
print(f"  Avg step:         {avg_step*1000:.0f}ms ({avg_step:.2f}s)")
if cache_total > 0:
    hit_rate = cache_hits / cache_total * 100
    print(f"  Cache hits:       {cache_hits}/{cache_total} ({hit_rate:.0f}%)")
    if cache_hits > 0:
        # Estimate speedup: cached steps are ~free vs full forward pass
        computed_times = [t for i, t in enumerate(step_times) if i < cache_misses]
        if computed_times:
            avg_computed = np.mean(computed_times)
            theoretical = avg_computed * n_steps
            print(f"  Cache speedup:    ~{theoretical / total_design:.1f}x "
                  f"({total_design:.1f}s vs ~{theoretical:.0f}s uncached)")
if NUM_BACKBONES > 1:
    print(f"  Throughput:       {avg_step/NUM_BACKBONES*1000:.0f}ms/design/step")
print(f"  Init time:        {t_init - t_start:.1f}s")
print(f"  Total wall time:  {time.time() - t_start:.1f}s")

# --- Write PDB outputs (one per backbone) ---
backbone_pdbs = []  # list of (pdb_path, design_idx) for MPNN
bb_nan_frac = 0.0

for di, result in enumerate(design_results):
    out_prefix = f'{conf.inference.output_prefix}_{di}'
    out = f'{out_prefix}.pdb'

    denoised_xyz_stack = torch.flip(result.denoised_xyz, [0])
    px0_xyz_stack = torch.flip(result.px0_xyz, [0])
    plddt_stack = result.plddt

    # Final sequence: glycines for diffused positions, original for fixed
    seq_init_i = result.seq_init
    final_seq = torch.where(
        torch.argmax(seq_init_i, dim=-1) == 21, 7,
        torch.argmax(seq_init_i, dim=-1))

    # B-factors: 0 for diffused, 1 for fixed
    bfacts = torch.ones_like(final_seq.squeeze())
    bfacts[torch.where(torch.argmax(seq_init_i, dim=-1) == 21, True, False)] = 0
    if hasattr(sampler, 'ab_item') and hasattr(sampler.ab_item, 'hotspots'):
        bfacts[sampler.ab_item.hotspots] = 0

    backbone_coords = denoised_xyz_stack[0, :, :4]
    bb_nan_frac = max(bb_nan_frac, torch.isnan(backbone_coords).float().mean().item())

    if torch.isnan(backbone_coords).float().mean().item() > 0.5:
        print(f"\n  WARNING: backbone {di} has NaN coords — skipping")
        continue

    pdblines = ab_write_pdblines(
        atoms=denoised_xyz_stack[0, :, :4].cpu().numpy(),
        seq=final_seq.cpu().numpy(),
        chain_idx=sampler.chain_idx,
        bfacts=bfacts.cpu().numpy(),
        loop_map=sampler.loop_map,
        num2aa=num2aa,
    )
    with open(out, 'w') as f_out:
        f_out.write('\n'.join(pdblines))

    trb = dict(
        config=OmegaConf.to_container(sampler._conf, resolve=True),
        plddt=plddt_stack.cpu().numpy(),
        device='MLX (Apple Silicon)',
        time=time.time() - t_start,
        design_idx=di,
        batch_size=NUM_BACKBONES,
    )
    for loop in sampler.loop_map:
        trb[f"{loop.upper()}_len"] = len(sampler.loop_map[loop])
    with open(f'{out_prefix}.trb', 'wb') as f_out:
        pickle.dump(trb, f_out)

    backbone_pdbs.append((out, di))
    print(f"\n  Backbone {di}: {out}")

# For downstream MPNN — use first backbone's PDB as reference
out = backbone_pdbs[0][0] if backbone_pdbs else None

print()

# ==========================================================================
#  Stage 2: MPNN Sequence Design (optional)
# ==========================================================================
mpnn_ckpt = os.path.join(script_dir, '..', 'models', 'ProteinMPNN_v48_noise_0.2.pt')
rf2_ckpt = os.path.join(script_dir, '..', 'models', 'RF2_ab.pt')

SKIP_MPNN = os.environ.get('SKIP_MPNN', '0') == '1'
SKIP_RF2 = os.environ.get('SKIP_RF2', '0') == '1'
NUM_SEQS = int(os.environ.get('MPNN_NUM_SEQS', '1'))
MPNN_TEMP = float(os.environ.get('MPNN_TEMP', '0.1'))
RF2_RECYCLES = int(os.environ.get('RF2_RECYCLES', '10'))
RF2_THRESHOLD = float(os.environ.get('RF2_THRESHOLD', '0.5'))

designed_pdbs = []  # list of (pdb_path, mpnn_score) for RF2
t_mpnn_start = t_mpnn_end = None

if (not SKIP_MPNN and os.path.exists(mpnn_ckpt)
        and bb_nan_frac < 0.5 and backbone_pdbs):

    print("=" * 60)
    print("  Stage 2: MPNN Sequence Design")
    print("=" * 60)

    t_mpnn_start = time.time()

    from rfantibody.proteinmpnn.mlx.model_wrapper import MLXMPNNWrapper
    import rfantibody.proteinmpnn.util_protein_mpnn as mpnn_util
    from rfantibody.util.pose import Pose as SimplePose
    from rfantibody.proteinmpnn.sample_features import SampleFeatures

    print("  Loading MLX ProteinMPNN...")
    mpnn = MLXMPNNWrapper.from_checkpoint(mpnn_ckpt)
    print(f"  Loaded in {time.time() - t_mpnn_start:.1f}s")

    design_loops = ','.join([l.rstrip(':') for l in conf.antibody.design_loops])

    # Design sequences for each backbone
    for bb_pdb, bb_idx in backbone_pdbs:
        bb_prefix = os.path.splitext(bb_pdb)[0]
        backbone_pose = SimplePose.from_pdb(bb_pdb)
        sample_feats = SampleFeatures(backbone_pose, tag=os.path.basename(bb_prefix))
        sample_feats.loop_string2fixed_res(design_loops)

        feature_dict = mpnn_util.generate_seqopt_features(bb_pdb, sample_feats.chains)
        arg_dict = mpnn_util.set_default_args(NUM_SEQS, omit_AAs=['C', 'X'])
        arg_dict['temperature'] = MPNN_TEMP

        masked_chains = sample_feats.chains[:-1]
        visible_chains = [sample_feats.chains[-1]]
        fixed_pos_dict = {feature_dict['name']: sample_feats.fixed_res}

        print(f"\n  Backbone {bb_idx}: {design_loops}, {NUM_SEQS} seqs (T={MPNN_TEMP})...")

        make_deterministic(bb_idx)
        seqs_scores = mpnn_util.generate_sequences(
            mpnn, 'cpu', feature_dict, arg_dict,
            masked_chains, visible_chains, fixed_pos_dict
        )

        for i, (seq, score) in enumerate(seqs_scores):
            # Reuse cached backbone_pose instead of re-reading PDB from disk
            import copy
            pose_i = copy.deepcopy(backbone_pose)
            sf_i = SampleFeatures(pose_i, tag=f'bb{bb_idx}_seq{i}')
            sf_i.loop_string2fixed_res(design_loops)
            sf_i.thread_mpnn_seq(seq)
            seq_pdb = f'{bb_prefix}_seq{i}.pdb'
            pose_i.dump_pdb(seq_pdb)
            designed_pdbs.append((seq_pdb, score))

    designed_pdbs.sort(key=lambda x: x[1])
    t_mpnn_end = time.time()

    print(f"\n  {len(designed_pdbs)} sequences in {t_mpnn_end - t_mpnn_start:.1f}s:")
    for i, (pdb, score) in enumerate(designed_pdbs):
        print(f"    #{i+1}: score={score:.3f}  {os.path.basename(pdb)}")
    print()

elif SKIP_MPNN:
    print("MPNN skipped (SKIP_MPNN=1)")
elif not os.path.exists(mpnn_ckpt):
    print(f"MPNN skipped (checkpoint not found: {mpnn_ckpt})")

# ==========================================================================
#  Stage 3: RF2 Structure Validation (optional)
# ==========================================================================
rf2_results = []
t_rf2_start = t_rf2_end = None

if (not SKIP_RF2 and os.path.exists(rf2_ckpt)
        and designed_pdbs and bb_nan_frac < 0.5):

    print("=" * 60)
    print("  Stage 3: RF2 Structure Validation")
    print("=" * 60)
    print(f"  Recycles: max {RF2_RECYCLES}, converge at Ca RMSD < {RF2_THRESHOLD} A")
    print()

    t_rf2_start = time.time()

    import rfantibody.rf2.modules.pose_util as pu
    from rfantibody.rf2.modules.preprocess import Preprocess, pose_to_inference_RFinput
    from rfantibody.rf2.mlx.predictor import MLXAbPredictor

    # Apply RF2 mode-specific optimizations
    RF2_MODE_CONFIGS = {
        'full':  dict(se3_stride=1, n_main=0, recycles=10),
        'fast':  dict(se3_stride=2, n_main=24, recycles=5),
        'draft': dict(se3_stride=4, n_main=18, recycles=3),
    }
    rf2_mode_cfg = RF2_MODE_CONFIGS.get(MODE, RF2_MODE_CONFIGS['full'])
    os.environ['RF2_SE3_STRIDE'] = str(rf2_mode_cfg['se3_stride'])
    os.environ['RF2_N_MAIN'] = str(rf2_mode_cfg['n_main'])
    rf2_recycles = min(RF2_RECYCLES, rf2_mode_cfg['recycles'])
    print(f"  RF2 mode: se3_stride={rf2_mode_cfg['se3_stride']}, "
          f"n_main={rf2_mode_cfg['n_main'] or 'all'}, recycles={rf2_recycles}")

    # Build RF2 config
    rf2_conf = OmegaConf.create({
        'model': {'model_weights': rf2_ckpt},
        'inference': {
            'num_recycles': rf2_recycles,
            'converge_threshold': RF2_THRESHOLD,
            'hotspot_show_proportion': 0.1,
        },
        'output': {
            'pdb_dir': output_dir,
            'quiver': None,
            'output_intermediates': False,
        },
        'input': {
            'pdb': 'pipeline',  # non-None to enable RMSD calculation
            'pdb_dir': None,
            'quiver': None,
        },
    })

    print("  Loading MLX RF2 model...")
    preprocessor = Preprocess(pose_to_inference_RFinput, rf2_conf)
    predictor = MLXAbPredictor(rf2_conf, preprocess_fn=preprocessor, device='cpu')
    print(f"  Loaded in {time.time() - t_rf2_start:.1f}s")
    print()

    # Validate each designed structure
    for pdb_path, mpnn_score in designed_pdbs:
        tag = os.path.splitext(os.path.basename(pdb_path))[0]
        try:
            rf2_pose = pu.pose_from_remarked(pdb_path)
            metrics = predictor(rf2_pose, tag)
            rf2_results.append((tag, mpnn_score, True, metrics))
        except Exception as e:
            print(f"  WARNING: RF2 failed for {tag}: {e}")
            import traceback; traceback.print_exc()
            rf2_results.append((tag, mpnn_score, False, None))

    t_rf2_end = time.time()

    n_ok = sum(1 for r in rf2_results if r[2])
    print()
    print(f"  Validated: {n_ok}/{len(rf2_results)} structures in {t_rf2_end - t_rf2_start:.1f}s")
    print(f"  Outputs:   {output_dir}/")
    print()

elif SKIP_RF2:
    print("RF2 skipped (SKIP_RF2=1)")
elif not os.path.exists(rf2_ckpt):
    print(f"RF2 skipped (checkpoint not found: {rf2_ckpt})")
elif not designed_pdbs:
    print("RF2 skipped (no MPNN designs to validate)")

# ==========================================================================
#  Pipeline Summary
# ==========================================================================
print("=" * 60)
print("  Pipeline Summary")
print("=" * 60)

total_wall = time.time() - t_start
print(f"\n  Timing:")
s1_desc = f"{n_steps} steps, {avg_step*1000:.0f}ms/step"
if NUM_BACKBONES > 1:
    s1_desc += f", B={NUM_BACKBONES} ({avg_step/NUM_BACKBONES*1000:.0f}ms/design/step)"
print(f"    Stage 1 (RFdiffusion): {total_design:.1f}s  ({s1_desc})")
if t_mpnn_start and t_mpnn_end:
    print(f"    Stage 2 (MPNN):        {t_mpnn_end - t_mpnn_start:.1f}s  ({len(designed_pdbs)} sequences)")
if t_rf2_start and t_rf2_end:
    n_ok = sum(1 for r in rf2_results if r[2])
    print(f"    Stage 3 (RF2):         {t_rf2_end - t_rf2_start:.1f}s  ({n_ok} structures validated)")
print(f"    Total wall time:       {total_wall:.1f}s")

# Print backbone geometry stats for each design
print(f"\n  Backbone geometry:")
print(f"    Protein length:     L={L}")
for di, result in enumerate(design_results):
    xyz_final = torch.flip(result.denoised_xyz, [0])
    ca_coords = xyz_final[0, :, 1].cpu().numpy()
    ca_bonds = np.linalg.norm(np.diff(ca_coords, axis=0), axis=-1)
    ca_center = ca_coords.mean(axis=0)
    rg = np.sqrt(np.mean(np.sum((ca_coords - ca_center)**2, axis=-1)))
    label = f"    Backbone {di}" if NUM_BACKBONES > 1 else "   "
    print(f"{label} CA-CA bond mean:    {ca_bonds.mean():.3f} A (expected ~3.8 A)")
    print(f"{label} Radius of gyration: {rg:.1f} A")

# Print MPNN results
if designed_pdbs:
    print(f"\n  MPNN sequence design:")
    for pdb, score in designed_pdbs:
        print(f"    {os.path.basename(pdb)}: score={score:.3f}")

# Print RF2 validation metrics
if rf2_results:
    print(f"\n  RF2 validation metrics:")
    for tag, mpnn_score, ok, metrics in rf2_results:
        if not ok or metrics is None:
            print(f"    {tag}: FAILED")
            continue
        plddt = metrics.get('pred_lddt')
        pae = metrics.get('pae')
        ipae = metrics.get('interaction_pae')
        p_bind = metrics.get('p_bind')

        print(f"\n    {tag}:")
        print(f"      MPNN score:        {mpnn_score:.3f}")
        if plddt is not None:
            plddt_mean = plddt.mean().item()
            print(f"      pLDDT (mean):      {plddt_mean:.3f}")
        if pae is not None:
            pae_mean = pae.mean().item()
            print(f"      PAE (mean):        {pae_mean:.2f} A")
        if ipae is not None:
            ipae_val = ipae.item() if torch.is_tensor(ipae) else ipae
            print(f"      Interaction PAE:   {ipae_val:.2f} A")
        if p_bind is not None:
            print(f"      P(bind):           {p_bind:.4f}")

        # Print RMSD metrics if available
        rmsd_keys = [k for k in metrics if 'rmsd' in k.lower()]
        if rmsd_keys:
            print(f"      RMSDs:")
            for k in sorted(rmsd_keys):
                v = metrics[k]
                val = v.item() if torch.is_tensor(v) else v
                label = k.replace('_', ' ').replace('framework aligned', 'fw-aligned').replace('target aligned', 'tgt-aligned')
                print(f"        {label:30s} {val:.3f} A")

print()
print(f"  Output files: {output_dir}/")
print()
