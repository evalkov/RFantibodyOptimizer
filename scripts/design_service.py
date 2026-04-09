#!/usr/bin/env python3
"""
JSON-lines design service for the NanobodyDesigner Swift app.

Reads a JSON config object from stdin, runs the full pipeline
(RFdiffusion → ProteinMPNN → RF2), and emits structured JSON events
to stderr for the Swift UI to consume. Human-readable output goes to stdout.

Usage (standalone test):
    echo '{"target_pdb": "~/Desktop/bcma.pdb", ...}' | \
    PYTHONPATH=src:include/SE3Transformer pilot_mps/.venv/bin/python scripts/design_service.py
"""
import json
import math
import os
import pickle
import random
import signal
import sys
import time
import logging
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'include', 'SE3Transformer'))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# Suppress icecream
try:
    from icecream import ic; ic.disable()
except ImportError:
    pass


def _sanitize_for_json(obj):
    """Replace NaN/Inf floats with None so json.dumps produces valid JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def emit(event_dict):
    """Write a JSON event to stderr (for Swift to parse)."""
    line = json.dumps(_sanitize_for_json(event_dict), default=str)
    sys.stderr.write(line + '\n')
    sys.stderr.flush()


def emit_error(message, tb=None):
    """Emit an error event."""
    evt = {'event': 'error', 'message': str(message)}
    if tb:
        evt['traceback'] = tb
    emit(evt)


def _rewrite_bfacts(pdb_path, bfacts_1d):
    """Rewrite B-factor column of a PDB file using a per-residue array.

    bfacts_1d: 1D array of length L (one value per residue), 0=CDR, 1=framework/target.
    """
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
    res_counter = -1
    prev_key = None
    new_lines = []
    for line in lines:
        if line.startswith('ATOM'):
            key = line[21:27]          # chain + resSeq (fixed-width)
            if key != prev_key:
                res_counter += 1
                prev_key = key
            if res_counter < len(bfacts_1d):
                bf = float(bfacts_1d[res_counter])
                line = line[:60] + f'{bf:6.2f}' + line[66:]
        new_lines.append(line)
    with open(pdb_path, 'w') as f:
        f.writelines(new_lines)


# --- Signal handling ---
_cancelled = False

def _handle_sigterm(signum, frame):
    global _cancelled
    _cancelled = True
    emit({'event': 'cancelled'})
    sys.exit(0)

signal.signal(signal.SIGTERM, _handle_sigterm)


# --- Config defaults ---
DEFAULTS = {
    'target_pdb': '',
    'framework_pdb': '',
    'design_loops': ['H1:', 'H2:', 'H3:'],
    'num_designs': 1,
    'mode': 'draft',
    'diffusion_T': 15,
    't_scheme': 'single_T',
    'hotspot_res': '',
    'noise_scale_ca': 1.0,
    'noise_scale_frame': 1.0,
    'mpnn_temp': 0.1,
    'mpnn_seqs': 1,
    'rf2_recycles': 10,
    'rf2_threshold': 0.5,
    'output_dir': '/tmp/NanobodyDesigner',
    'seed': 0,
    'skip_mpnn': False,
    'skip_rf2': False,
    'cache_enabled': True,
    'cache_threshold': 0.15,
    'cache_warmup': 3,
    'validator': 'rf2',
}

MODE_CONFIGS = {
    'full':  dict(top_k=64, se3_stride=1, n_main=32),
    'fast':  dict(top_k=64, se3_stride=4, n_main=32),
    'draft': dict(top_k=64, se3_stride=4, n_main=24),
}

# RF2 mode-specific settings: se3_stride, n_main_block, recycles
RF2_MODE_CONFIGS = {
    'full':  dict(se3_stride=1, n_main=0, recycles=10),   # 0 = all blocks
    'fast':  dict(se3_stride=2, n_main=24, recycles=5),
    'draft': dict(se3_stride=4, n_main=18, recycles=3),
}


def make_deterministic(seed=0):
    import torch
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_pipeline(cfg):
    """Main pipeline: RFdiffusion → MPNN → RF2."""
    import torch
    import numpy as np
    from omegaconf import OmegaConf

    from rfantibody.rfdiffusion.chemical import num2aa
    from rfantibody.util.io import ab_write_pdblines

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(script_dir, '..', 'models', 'RFdiffusion_Ab.pt')
    mpnn_ckpt = os.path.join(script_dir, '..', 'models', 'ProteinMPNN_v48_noise_0.2.pt')
    rf2_ckpt = os.path.join(script_dir, '..', 'models', 'RF2_ab.pt')

    # Resolve paths
    target_pdb = os.path.expanduser(cfg['target_pdb'])
    framework_pdb = os.path.expanduser(cfg['framework_pdb'])
    output_dir = os.path.expanduser(cfg['output_dir'])
    os.makedirs(output_dir, exist_ok=True)

    num_designs = cfg['num_designs']
    mode = cfg.get('mode', 'draft')
    diffusion_T = cfg['diffusion_T']
    t_scheme = cfg.get('t_scheme', 'single_T')
    hotspot_res = cfg.get('hotspot_res', '')
    noise_scale_ca = cfg.get('noise_scale_ca', 1.0)
    noise_scale_frame = cfg.get('noise_scale_frame', 1.0)

    # Design loops come pre-formatted from Swift: ["H1:", "H2:5-10", "H3:12"]
    design_loops_conf = cfg['design_loops']
    # Strip length specs for MPNN (just need loop names)
    design_loops_str = ','.join(l.split(':')[0] for l in design_loops_conf)

    mode_cfg = MODE_CONFIGS.get(mode, MODE_CONFIGS['draft'])

    emit({
        'event': 'pipeline_start',
        'num_designs': num_designs,
        'T': diffusion_T,
        'mode': mode,
        't_scheme': t_scheme,
        'design_loops': design_loops_conf,
        'hotspot_res': hotspot_res or None,
        'target_pdb': target_pdb,
        'framework_pdb': framework_pdb,
        'output_dir': output_dir,
    })

    # --- Build OmegaConf ---
    base_cfg = OmegaConf.load(os.path.join(script_dir, 'config', 'inference', 'base.yaml'))
    ab_cfg = OmegaConf.load(os.path.join(script_dir, 'config', 'inference', 'antibody.yaml'))
    if 'defaults' in ab_cfg:
        del ab_cfg['defaults']
    conf = OmegaConf.merge(base_cfg, ab_cfg)

    conf.antibody.target_pdb = target_pdb
    conf.antibody.framework_pdb = framework_pdb
    conf.inference.output_prefix = os.path.join(output_dir, 'design')
    conf.inference.num_designs = 1
    conf.inference.ckpt_override_path = ckpt_path
    conf.inference.write_trajectory = False
    conf.inference.deterministic = True
    conf.diffuser.T = diffusion_T
    conf.antibody.design_loops = design_loops_conf
    conf.antibody.T_scheme = t_scheme

    # Hotspot residues
    if hotspot_res:
        conf.ppi.hotspot_res = hotspot_res

    # Noise scales
    conf.denoiser.noise_scale_ca = noise_scale_ca
    conf.denoiser.noise_scale_frame = noise_scale_frame

    # Adaptive step cache settings (passed via JSON, consumed via os.environ)
    os.environ['CACHE_ENABLED'] = '1' if cfg.get('cache_enabled', True) else '0'
    os.environ['CACHE_THRESHOLD'] = str(cfg.get('cache_threshold', 0.15))
    os.environ['CACHE_WARMUP'] = str(cfg.get('cache_warmup', 3))

    # --- Init sampler ---
    emit({'event': 'init_start', 'component': 'sampler'})
    t0 = time.time()

    from rfantibody.rfdiffusion.mlx.sampler import MLXAbSampler
    sampler = MLXAbSampler(conf)
    sampler.model.enable_mixed_precision()
    sampler.model.enable_fused_kernels()
    sampler.model.set_topk_graph(mode_cfg['top_k'])
    sampler.model.set_se3_stride(mode_cfg['se3_stride'])
    sampler.model.set_n_main_block(mode_cfg['n_main'])
    sampler.model.set_eval_stride(4)

    sampler_time = time.time() - t0
    print(f"Sampler initialized in {sampler_time:.1f}s", flush=True)

    # --- Init MPNN ---
    mpnn = None
    mpnn_time = 0
    if not cfg.get('skip_mpnn', False) and os.path.exists(mpnn_ckpt):
        emit({'event': 'init_start', 'component': 'mpnn'})
        t0 = time.time()
        from rfantibody.proteinmpnn.mlx.model_wrapper import MLXMPNNWrapper
        mpnn = MLXMPNNWrapper.from_checkpoint(mpnn_ckpt)
        mpnn_time = time.time() - t0
        print(f"MPNN loaded in {mpnn_time:.1f}s", flush=True)

    # --- Init structure validator (RF2 or Protenix) ---
    predictor = None
    rf2_time = 0
    validator = cfg.get('validator', 'rf2')
    protenix_ckpt = os.path.join(script_dir, '..', 'models', 'protenix_mini.pt')

    if validator == 'protenix' and not cfg.get('skip_rf2', False) and os.path.exists(protenix_ckpt):
        emit({'event': 'init_start', 'component': 'protenix'})
        t0 = time.time()
        from rfantibody.protenix.mlx.predictor import ProtenixPredictor

        protenix_conf = {
            'model_weights': protenix_ckpt,
            'n_diffusion_steps': 5,
            'tea_cache_threshold': cfg.get('cache_threshold', 0.15),
        }
        predictor = ProtenixPredictor(protenix_conf, device='cpu')
        rf2_time = time.time() - t0
        print(f"Protenix-Mini-Flow loaded in {rf2_time:.1f}s", flush=True)

    elif not cfg.get('skip_rf2', False) and os.path.exists(rf2_ckpt):
        emit({'event': 'init_start', 'component': 'rf2'})
        t0 = time.time()
        import rfantibody.rf2.modules.pose_util as pu
        from rfantibody.rf2.modules.preprocess import Preprocess, pose_to_inference_RFinput
        from rfantibody.rf2.mlx.predictor import MLXAbPredictor

        # Apply RF2 mode-specific optimizations
        rf2_mode_cfg = RF2_MODE_CONFIGS.get(mode, RF2_MODE_CONFIGS['full'])
        os.environ['RF2_SE3_STRIDE'] = str(rf2_mode_cfg['se3_stride'])
        os.environ['RF2_N_MAIN'] = str(rf2_mode_cfg['n_main'])
        # Use mode-appropriate recycles unless user overrode
        rf2_recycles = cfg.get('rf2_recycles', rf2_mode_cfg['recycles'])
        if rf2_recycles > rf2_mode_cfg['recycles'] and mode != 'full':
            rf2_recycles = rf2_mode_cfg['recycles']

        rf2_conf = OmegaConf.create({
            'model': {'model_weights': rf2_ckpt},
            'inference': {
                'num_recycles': rf2_recycles,
                'converge_threshold': cfg['rf2_threshold'],
                'hotspot_show_proportion': 0.1,
            },
            'output': {
                'pdb_dir': output_dir,
                'quiver': None,
                'output_intermediates': False,
            },
            'input': {
                'pdb': 'pipeline',
                'pdb_dir': None,
                'quiver': None,
            },
        })
        preprocessor = Preprocess(pose_to_inference_RFinput, rf2_conf)
        predictor = MLXAbPredictor(rf2_conf, preprocess_fn=preprocessor, device='cpu')
        rf2_time = time.time() - t0
        print(f"RF2 loaded in {rf2_time:.1f}s", flush=True)

    emit({
        'event': 'init_complete',
        'sampler_time_s': round(sampler_time, 2),
        'mpnn_time_s': round(mpnn_time, 2),
        'rf2_time_s': round(rf2_time, 2),
    })

    # ====================================================================
    #  Design loop
    # ====================================================================
    for design_idx in range(num_designs):
        if _cancelled:
            break

        seed = cfg.get('seed', 0) + design_idx * 100
        make_deterministic(seed)

        emit({'event': 'design_start', 'design_idx': design_idx, 'seed': seed})

        # --- Stage 1: RFdiffusion ---
        emit({'event': 'stage_start', 'stage': 'rfdiffusion', 'design_idx': design_idx})
        t_rfdiff = time.time()

        x_init, seq_init = sampler.sample_init()
        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)
        L = x_init.shape[0] if x_init.ndim == 3 else x_init.shape[1]

        # Pre-compute seq/bfacts for trajectory PDB writing
        traj_seq = torch.where(
            torch.argmax(seq_init, dim=-1) == 21, 7,
            torch.argmax(seq_init, dim=-1))
        traj_bfacts = torch.ones_like(traj_seq.squeeze())
        traj_bfacts[torch.argmax(seq_init, dim=-1) == 21] = 0

        final_step = getattr(conf.inference, 'final_step', 1)
        total_steps = int(sampler.t_step_input) - final_step + 1

        # Write step 0: the noisy initial state
        traj_dir = os.path.join(output_dir, f'traj_{design_idx}')
        os.makedirs(traj_dir, exist_ok=True)
        step0_pdb = os.path.join(traj_dir, 'step_00.pdb')
        pdblines_0 = ab_write_pdblines(
            atoms=x_t[:, :4].cpu().numpy(),
            seq=traj_seq.cpu().numpy(),
            chain_idx=sampler.chain_idx,
            bfacts=traj_bfacts.cpu().numpy(),
            loop_map=sampler.loop_map,
            num2aa=num2aa,
        )
        with open(step0_pdb, 'w') as f:
            f.write('\n'.join(pdblines_0))

        for t in range(int(sampler.t_step_input), final_step - 1, -1):
            if _cancelled:
                break
            step_t0 = time.time()
            px0, x_t, seq_t, tors_t, plddt = sampler.sample_step(
                t=t, seq_t=seq_t, x_t=x_t, seq_init=seq_init,
                final_step=final_step)
            step_dt = time.time() - step_t0

            step_num = int(sampler.t_step_input) - t + 1

            # Write trajectory PDB for this step
            step_pdb = os.path.join(traj_dir, f'step_{step_num:02d}.pdb')
            step_pdblines = ab_write_pdblines(
                atoms=x_t[:, :4].cpu().numpy(),
                seq=traj_seq.cpu().numpy(),
                chain_idx=sampler.chain_idx,
                bfacts=traj_bfacts.cpu().numpy(),
                loop_map=sampler.loop_map,
                num2aa=num2aa,
            )
            with open(step_pdb, 'w') as f:
                f.write('\n'.join(step_pdblines))

            emit({
                'event': 'step_progress',
                'stage': 'rfdiffusion',
                'design_idx': design_idx,
                'step': step_num,
                'total': total_steps,
                'elapsed_s': round(time.time() - t_rfdiff, 2),
                'step_ms': round(step_dt * 1000, 0),
                'pdb_path': step_pdb,
            })

        if _cancelled:
            break

        rfdiff_time = time.time() - t_rfdiff

        # Write backbone PDB (reuse pre-computed seq/bfacts from trajectory)
        bb_pdb = os.path.join(output_dir, f'design_{design_idx}.pdb')
        pdblines = ab_write_pdblines(
            atoms=x_t[:, :4].cpu().numpy(),
            seq=traj_seq.cpu().numpy(),
            chain_idx=sampler.chain_idx,
            bfacts=traj_bfacts.cpu().numpy(),
            loop_map=sampler.loop_map,
            num2aa=num2aa,
        )
        with open(bb_pdb, 'w') as f:
            f.write('\n'.join(pdblines))

        # Backbone geometry
        ca = x_t[:, 1].cpu().numpy()
        ca_bonds = float(np.linalg.norm(np.diff(ca, axis=0), axis=-1).mean())
        ca_center = ca.mean(axis=0)
        rg = float(np.sqrt(np.mean(np.sum((ca - ca_center)**2, axis=-1))))

        emit({
            'event': 'stage_complete',
            'stage': 'rfdiffusion',
            'design_idx': design_idx,
            'pdb_path': bb_pdb,
            'time_s': round(rfdiff_time, 2),
            'L': L,
            'ca_ca_bond': round(ca_bonds, 3),
            'rg': round(rg, 1),
        })

        # --- Stage 2: MPNN ---
        seq_pdb = bb_pdb
        mpnn_score = None

        if mpnn is not None:
            emit({'event': 'stage_start', 'stage': 'mpnn', 'design_idx': design_idx})
            t_mpnn = time.time()

            import rfantibody.proteinmpnn.util_protein_mpnn as mpnn_util
            from rfantibody.util.pose import Pose as SimplePose
            from rfantibody.proteinmpnn.sample_features import SampleFeatures

            make_deterministic(seed)
            pose = SimplePose.from_pdb(bb_pdb)
            sf = SampleFeatures(pose, tag=f'design_{design_idx}')
            sf.loop_string2fixed_res(design_loops_str)

            feature_dict = mpnn_util.generate_seqopt_features(bb_pdb, sf.chains)
            arg_dict = mpnn_util.set_default_args(cfg['mpnn_seqs'], omit_AAs=['C', 'X'])
            arg_dict['temperature'] = cfg['mpnn_temp']

            masked_chains = sf.chains[:-1]
            visible_chains = [sf.chains[-1]]
            fixed_pos_dict = {feature_dict['name']: sf.fixed_res}

            seqs_scores = mpnn_util.generate_sequences(
                mpnn, 'cpu', feature_dict, arg_dict,
                masked_chains, visible_chains, fixed_pos_dict)

            # Use best sequence
            best_seq, best_score = sorted(seqs_scores, key=lambda x: x[1])[0]
            mpnn_score = float(best_score)

            pose2 = SimplePose.from_pdb(bb_pdb)
            sf2 = SampleFeatures(pose2, tag=f'design_{design_idx}_seq')
            sf2.loop_string2fixed_res(design_loops_str)
            sf2.thread_mpnn_seq(best_seq)
            seq_pdb = os.path.join(output_dir, f'design_{design_idx}_seq.pdb')
            pose2.dump_pdb(seq_pdb)

            # Rewrite B-factors: 0=CDR, 1=framework/target
            _rewrite_bfacts(seq_pdb, traj_bfacts.cpu().numpy().flatten())

            mpnn_time_design = time.time() - t_mpnn

            emit({
                'event': 'stage_complete',
                'stage': 'mpnn',
                'design_idx': design_idx,
                'pdb_path': seq_pdb,
                'score': round(mpnn_score, 4),
                'time_s': round(mpnn_time_design, 2),
                'num_seqs': cfg['mpnn_seqs'],
            })

        # --- Stage 3: RF2 ---
        rf2_metrics = {}

        if predictor is not None:
            emit({'event': 'stage_start', 'stage': 'rf2', 'design_idx': design_idx})
            t_rf2 = time.time()

            # Set up recycle callback for live progress
            def _on_recycle(cycle, total, ca_rmsd, plddt, pose=None):
                event = {
                    'event': 'step_progress',
                    'stage': 'rf2',
                    'design_idx': design_idx,
                    'step': cycle,
                    'total': total,
                    'ca_rmsd': round(ca_rmsd, 4) if ca_rmsd != float('inf') else None,
                    'plddt': round(plddt, 4),
                }
                # Write intermediate all-atom PDB for live visualization
                if pose is not None:
                    try:
                        from rfantibody.rf2.modules.util import get_pdblines
                        rf2_intermediate_pdb = os.path.join(output_dir, f'rf2_{design_idx}_cycle_{cycle}.pdb')
                        # Use same B-factor convention: 0=CDR (designed), 1=framework/target
                        lines = get_pdblines(pose, Bfacts=traj_bfacts.cpu())
                        with open(rf2_intermediate_pdb, 'w') as f:
                            f.writelines(lines)
                        event['pdb_path'] = rf2_intermediate_pdb
                    except Exception:
                        pass
                emit(event)
            predictor.on_recycle = _on_recycle

            try:
                import rfantibody.rf2.modules.pose_util as pu
                rf2_pose = pu.pose_from_remarked(seq_pdb)
                tag = f'design_{design_idx}'
                metrics = predictor(rf2_pose, tag)

                def _safe_float(val):
                    """Convert to Python float, returning None for NaN/Inf/None."""
                    if val is None:
                        return None
                    f = float(val.item() if torch.is_tensor(val) else val)
                    if math.isnan(f) or math.isinf(f):
                        return None
                    return f

                plddt_val = _safe_float(metrics['pred_lddt'].mean()) if metrics.get('pred_lddt') is not None else None
                pae_val = _safe_float(metrics['pae'].mean()) if metrics.get('pae') is not None else None
                ipae_val = _safe_float(metrics.get('interaction_pae'))
                # Use ipTM as P(bind) proxy when using Protenix validator
                p_bind_val = _safe_float(metrics.get('p_bind') or metrics.get('iptm', 0)) or 0.0

                # Extract RMSDs
                rmsd_dict = {}
                for k, v in metrics.items():
                    if 'rmsd' in k.lower():
                        f = _safe_float(v)
                        if f is not None:
                            rmsd_dict[k] = round(f, 3)

                cdr_rmsd = rmsd_dict.get('framework_aligned_cdr_rmsd')

                rf2_time_design = time.time() - t_rf2

                rf2_metrics = {
                    'plddt': round(plddt_val, 4) if plddt_val is not None else None,
                    'pae': round(pae_val, 2) if pae_val is not None else None,
                    'ipae': round(ipae_val, 2) if ipae_val is not None else None,
                    'p_bind': round(p_bind_val, 4),
                    'cdr_rmsd': round(cdr_rmsd, 2) if cdr_rmsd is not None else None,
                    'rmsds': rmsd_dict,
                }

                # Get validated PDB path (write_output uses {tag}_best.pdb)
                validated_pdb = os.path.join(output_dir, f'{tag}_best.pdb')

                # Rewrite validated PDB with CDR B-factors (0=CDR, 1=framework/target)
                if os.path.exists(validated_pdb):
                    try:
                        _rewrite_bfacts(validated_pdb, traj_bfacts.cpu().numpy().flatten())
                    except Exception as bfe:
                        print(f"[design_service] B-factor rewrite failed: {bfe}", flush=True)

                emit({
                    'event': 'stage_complete',
                    'stage': 'rf2',
                    'design_idx': design_idx,
                    'pdb_path': validated_pdb if os.path.exists(validated_pdb) else seq_pdb,
                    'time_s': round(rf2_time_design, 2),
                    **rf2_metrics,
                })

            except Exception as e:
                rf2_time_design = time.time() - t_rf2
                emit({
                    'event': 'stage_complete',
                    'stage': 'rf2',
                    'design_idx': design_idx,
                    'time_s': round(rf2_time_design, 2),
                    'error': str(e),
                })
                print(f"RF2 failed for design {design_idx}: {e}", flush=True)
                traceback.print_exc()

        # --- Design complete ---
        total_time = time.time() - t_rfdiff
        emit({
            'event': 'design_complete',
            'design_idx': design_idx,
            'total_time_s': round(total_time, 2),
            'backbone_pdb': bb_pdb,
            'sequence_pdb': seq_pdb,
            'mpnn_score': mpnn_score,
            **rf2_metrics,
        })

        print(f"Design {design_idx}: {total_time:.1f}s  "
              f"pLDDT={rf2_metrics.get('plddt', '?')}  "
              f"P(bind)={rf2_metrics.get('p_bind', '?')}", flush=True)

    # --- Pipeline complete ---
    total_pipeline = time.time() - t0 if 'sampler_time' not in dir() else 0
    emit({
        'event': 'pipeline_complete',
        'total_designs': num_designs,
        'total_time_s': round(time.time() - (t0 - sampler_time), 2),
    })


def main():
    """Read JSON config from stdin, run pipeline."""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            emit_error("No config provided on stdin")
            sys.exit(1)

        cfg = json.loads(raw)

        # Merge with defaults
        for k, v in DEFAULTS.items():
            if k not in cfg:
                cfg[k] = v

        print(f"Config: {json.dumps(cfg, indent=2)}", flush=True)
        run_pipeline(cfg)

    except json.JSONDecodeError as e:
        emit_error(f"Invalid JSON config: {e}")
        sys.exit(1)
    except Exception as e:
        emit_error(str(e), traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
