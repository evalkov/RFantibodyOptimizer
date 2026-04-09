"""
MLX-accelerated Protenix predictor for structure prediction on Apple Silicon.

Provides a pipeline integration class (ProtenixPredictor) that:
  1. Loads a Protenix-Mini-Flow model via MLXProtenixWrapper
  2. Accepts a nanobody+antigen pose (or sequences + chain info)
  3. Extracts features, runs the model, returns metrics

Compatible with the existing write_output() and results table
used by the RF2 pipeline.

Usage:
    predictor = ProtenixPredictor(conf, device='cpu')
    metrics = predictor(pose, tag='design_001')
"""

from __future__ import annotations

import os
import logging
from collections import OrderedDict

import torch
import numpy as np

_log = logging.getLogger(__name__)

# Standard amino acid 1-letter codes -> integer mapping
# Matches Protenix/AF3 restype vocabulary (first 20 = standard AAs)
AA_TO_IDX = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
    'X': 20, 'U': 21, 'B': 22, 'Z': 23, 'O': 24,
    '-': 25, '.': 26,
}

# 3-letter to 1-letter
AA3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'UNK': 'X',
}


def sequence_to_indices(seq: str) -> list[int]:
    """Convert a 1-letter amino acid sequence to integer indices."""
    return [AA_TO_IDX.get(c, 20) for c in seq]


class ProtenixPredictor:
    """MLX-accelerated structure predictor using Protenix-Mini-Flow.

    Drop-in alternative to MLXAbPredictor for cases where Protenix
    is preferred over RF2 for structure prediction.

    Args:
        conf: configuration object with model weights path and options.
        device: PyTorch device for output tensors (default 'cpu').
    """

    def __init__(self, conf, device='cpu'):
        # Support both dict and object-style configs
        if isinstance(conf, dict):
            self._cfg = conf
        else:
            self._cfg = {k: getattr(conf, k, None) for k in dir(conf) if not k.startswith('_')}
        self.device = device

        # Load model
        from rfantibody.protenix.mlx.model_wrapper import MLXProtenixWrapper

        model_weights = self._cfg.get('model_weights')
        if model_weights is None:
            raise ValueError(
                "Configuration must specify 'model_weights' path."
            )

        if not os.path.exists(model_weights):
            # Try relative to module location
            model_weights = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', model_weights
            )

        _log.info(f'Loading Protenix MLX model from {model_weights}')

        model_params = self._cfg.get('model_params', None)

        self.model = MLXProtenixWrapper.from_checkpoint(
            model_weights, model_params=model_params, torch_device=device
        )

        # Apply optimizations
        self.model.enable_mixed_precision()

        eval_stride = int(os.environ.get('PROTENIX_EVAL_STRIDE', '4'))
        self.model.set_eval_stride(eval_stride)

        n_pf_blocks = int(os.environ.get('PROTENIX_N_PF_BLOCKS', '0'))
        if n_pf_blocks > 0:
            self.model.set_n_pairformer_blocks(n_pf_blocks)

        n_diff_steps = int(self._cfg.get('n_diffusion_steps', os.environ.get('PROTENIX_N_DIFF_STEPS', '5')))
        self.model.set_n_diffusion_steps(n_diff_steps)

        tea_cache = float(self._cfg.get('tea_cache_threshold', os.environ.get('PROTENIX_TEA_CACHE', '0')))
        if tea_cache > 0:
            self.model.enable_tea_cache(tea_cache)

        _log.info(
            f'Protenix optimizations: fp16, eval_stride={eval_stride}, '
            f'diff_steps={n_diff_steps}, tea_cache={tea_cache}'
        )

        self.on_progress = None

    def _extract_features_from_pose(self, pose):
        """Extract sequence, chain, and residue features from a Pose object.

        Args:
            pose: Pose object (e.g., from rfantibody.rf2.modules)
                  Must have .seq (str or tensor) and chain info.

        Returns:
            dict with seq, residue_index, chain_id tensors
        """
        # Extract sequence
        if hasattr(pose, 'seq') and isinstance(pose.seq, str):
            seq_str = pose.seq
            seq_indices = sequence_to_indices(seq_str)
        elif hasattr(pose, 'seq') and torch.is_tensor(pose.seq):
            seq_indices = pose.seq.cpu().tolist()
        elif hasattr(pose, 'residues'):
            # Build from residue list
            seq_str = ''
            for res in pose.residues:
                name = getattr(res, 'name', 'UNK')
                seq_str += AA3_TO_1.get(name, 'X')
            seq_indices = sequence_to_indices(seq_str)
        else:
            raise ValueError("Cannot extract sequence from pose. "
                           "Expected .seq (str or tensor) or .residues")

        N = len(seq_indices)
        seq = torch.tensor([seq_indices], dtype=torch.long)

        # Extract chain IDs
        if hasattr(pose, 'chain_dict') and pose.chain_dict:
            # RF2 pose: chain_dict has {'H': bool_mask, 'T': bool_mask, ...}
            chain_id = torch.zeros(N, dtype=torch.long)
            for i, (chain_name, mask) in enumerate(pose.chain_dict.items()):
                if torch.is_tensor(mask):
                    chain_id[mask.bool()] = i
                else:
                    chain_id[torch.tensor(mask, dtype=torch.bool)] = i
            chain_id = chain_id.unsqueeze(0)  # [1, N]
        elif hasattr(pose, 'chain_id'):
            if torch.is_tensor(pose.chain_id):
                chain_id = pose.chain_id.unsqueeze(0).long()
            else:
                chain_id = torch.tensor([pose.chain_id], dtype=torch.long)
        elif hasattr(pose, 'chain_ids'):
            chain_id = torch.tensor([pose.chain_ids], dtype=torch.long)
        else:
            # Default: single chain
            chain_id = torch.zeros(1, N, dtype=torch.long)

        # Extract residue indices
        if hasattr(pose, 'residue_index'):
            if torch.is_tensor(pose.residue_index):
                residue_index = pose.residue_index.unsqueeze(0).long()
            else:
                residue_index = torch.tensor([pose.residue_index], dtype=torch.long)
        elif hasattr(pose, 'idx'):
            residue_index = torch.tensor([pose.idx], dtype=torch.long) if not torch.is_tensor(pose.idx) else pose.idx.unsqueeze(0).long()
        else:
            residue_index = torch.arange(N, dtype=torch.long).unsqueeze(0)

        return {
            'seq': seq,
            'residue_index': residue_index,
            'chain_id': chain_id,
        }

    def _extract_features_from_sequences(
        self,
        sequences: list[str],
        chain_ids: list[int] | None = None,
    ) -> dict:
        """Build input features from raw sequences.

        Args:
            sequences: list of amino acid sequences (one per chain)
            chain_ids: optional chain ID for each sequence

        Returns:
            dict with seq, residue_index, chain_id tensors
        """
        full_seq = ''.join(sequences)
        seq_indices = sequence_to_indices(full_seq)
        N = len(seq_indices)

        seq = torch.tensor([seq_indices], dtype=torch.long)
        residue_index = torch.arange(N, dtype=torch.long).unsqueeze(0)

        # Build chain IDs
        if chain_ids is None:
            chain_ids = list(range(len(sequences)))
        cid_list = []
        for i, s in enumerate(sequences):
            cid_list.extend([chain_ids[i]] * len(s))
        chain_id = torch.tensor([cid_list], dtype=torch.long)

        return {
            'seq': seq,
            'residue_index': residue_index,
            'chain_id': chain_id,
        }

    def __call__(self, pose, tag: str = 'protenix') -> dict:
        """Run structure prediction on a pose.

        Args:
            pose: Pose object with sequence and chain information
            tag: identifier string for logging/output

        Returns:
            dict with prediction metrics:
                pred_lddt: [N] per-residue pLDDT scores
                pae: [N, N] predicted aligned error
                iptm: scalar interface pTM
                ptm: scalar predicted TM-score
                coordinates: [N, 3] predicted coordinates
        """
        _log.info(f'[Protenix] Processing: {tag}')

        # Extract features
        features = self._extract_features_from_pose(pose)

        # Extract backbone Cα coordinates from pose if available
        # (bypasses diffusion module which lacks atom attention weights)
        coords_override = None
        if hasattr(pose, 'xyz') and torch.is_tensor(pose.xyz):
            # pose.xyz: [L, n_atoms, 3] — take Cα (index 1)
            ca_coords = pose.xyz[:, 1, :].unsqueeze(0).float()  # [1, L, 3]
            coords_override = ca_coords

        with torch.no_grad():
            result = self.model(
                seq=features['seq'],
                residue_index=features['residue_index'],
                chain_id=features['chain_id'],
                n_cycle=self._cfg.get('n_cycle', 1),
                run_confidence=True,
                coordinates_override=coords_override,
            )

        # Format output metrics to match RF2 convention
        metrics = OrderedDict()

        if 'plddt' in result:
            metrics['pred_lddt'] = result['plddt'].squeeze(0)  # [N]
        if 'pae_logits' in result:
            # Convert PAE logits to expected distance error
            pae_logits = result['pae_logits'].squeeze(0)  # [N, N, 64]
            pae_probs = torch.softmax(pae_logits, dim=-1)
            bin_centers = torch.linspace(0, 31.75, 64)
            pae = (pae_probs * bin_centers).sum(-1)  # [N, N]
            metrics['pae'] = pae
        if 'ptm' in result:
            metrics['ptm'] = result['ptm']
        if 'iptm' in result:
            metrics['iptm'] = result['iptm']
        if 'coordinates' in result:
            metrics['coordinates'] = result['coordinates'].squeeze(0)  # [N, 3]
        if 'pde_logits' in result:
            metrics['pde_logits'] = result['pde_logits'].squeeze(0)

        plddt_mean = float(metrics.get('pred_lddt', torch.tensor(0)).mean())
        _log.info(f'[Protenix] Completed: {tag} - pLDDT: {plddt_mean:.3f}')

        # Callback for UI progress
        if self.on_progress is not None:
            self.on_progress(
                tag=tag,
                plddt=plddt_mean,
                ptm=float(metrics.get('ptm', 0)),
                iptm=float(metrics.get('iptm', 0)),
            )

        return metrics

    def predict_from_sequences(
        self,
        sequences: list[str],
        chain_ids: list[int] | None = None,
        tag: str = 'protenix',
    ) -> dict:
        """Run prediction directly from amino acid sequences.

        Convenience method that doesn't require a Pose object.

        Args:
            sequences: list of amino acid sequences (one per chain)
            chain_ids: optional chain ID per sequence
            tag: identifier string

        Returns:
            dict with prediction metrics (same format as __call__)
        """
        features = self._extract_features_from_sequences(sequences, chain_ids)

        with torch.no_grad():
            result = self.model(
                seq=features['seq'],
                residue_index=features['residue_index'],
                chain_id=features['chain_id'],
                n_cycle=self._cfg.get('n_cycle', 1),
                run_confidence=True,
            )

        metrics = OrderedDict()
        if 'plddt' in result:
            metrics['pred_lddt'] = result['plddt'].squeeze(0)
        if 'pae_logits' in result:
            pae_logits = result['pae_logits'].squeeze(0)
            pae_probs = torch.softmax(pae_logits, dim=-1)
            bin_centers = torch.linspace(0, 31.75, 64)
            metrics['pae'] = (pae_probs * bin_centers).sum(-1)
        if 'ptm' in result:
            metrics['ptm'] = result['ptm']
        if 'iptm' in result:
            metrics['iptm'] = result['iptm']
        if 'coordinates' in result:
            metrics['coordinates'] = result['coordinates'].squeeze(0)

        plddt_mean = float(metrics.get('pred_lddt', torch.tensor(0)).mean())
        _log.info(f'[Protenix] Completed: {tag} - pLDDT: {plddt_mean:.3f}')

        return metrics
