"""
Top-level Protenix-Mini-Flow model ported to MLX (Apple Silicon).

Assembles:
  1. InputFeatureEmbedder + RelativePositionEncoding (embeddings)
  2. PairformerStack (trunk -- runs once per recycling cycle)
  3. DiffusionModule + FlowMatchingODESampler (structure generation)
  4. ConfidenceHead (quality estimation)

This is the MLX equivalent of protenix/model/protenix.py (Protenix class),
simplified for Mini-Flow inference.

Architecture follows AF3 Algorithm 1 (simplified):
  Line 1-5: Embed inputs -> s_inputs (N, 449)
  Line 6-13: Recycling: init s,z from s_inputs, run trunk
  Line 14-16: Sample diffusion -> coordinates
  Line 17: Confidence head -> pLDDT, PAE, pTM
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from rfantibody.protenix.mlx.embedders import (
    InputFeatureEmbedder,
    LinearNoBias,
    RelativePositionEncoding,
)
from rfantibody.protenix.mlx.pairformer import PairformerStack, Transition
from rfantibody.protenix.mlx.diffusion import (
    DiffusionModule,
    FlowMatchingODESampler,
)
from rfantibody.protenix.mlx.confidence import ConfidenceHead, compute_iptm
from rfantibody.protenix.mlx.teacache import TeaCache


class ProtenixMiniModule(nn.Module):
    """Protenix-Mini-Flow MLX model for protein complex structure prediction.

    This is the top-level model that combines input embedding, the Pairformer
    trunk, flow-matching diffusion, and confidence estimation.

    Architecture (AF3 Algorithm 1, simplified for Mini-Flow):
      1. Embed: sequence + features -> s_inputs (N, c_s_inputs=449)
      2. Init: s = Linear(s_inputs), z = outer_product(s) + relpos
      3. Trunk: PairformerStack(s, z) -> updated s_trunk, z_trunk
      4. Diffuse: FlowMatchingODE(s_inputs, s_trunk, z_trunk) -> coordinates
      5. Confidence: ConfidenceHead(coords, s_inputs, s_trunk, z_trunk) -> metrics

    Args:
        c_s: single embedding dim (default 384).
        c_z: pair embedding dim (default 128).
        c_token: token dim for diffusion transformer (default 768).
        c_s_inputs: input feature dim = c_token + 32 + 32 + 1 (default 449).
        n_token_types: number of residue types (default 32).
        r_max: relative position clip (default 32).
        s_max: relative chain clip (default 2).
        n_pairformer_blocks: trunk blocks (default 16).
        n_diffusion_blocks: diffusion transformer blocks (default 8).
        n_confidence_blocks: confidence pairformer blocks (default 4).
        n_diffusion_steps: ODE steps (default 5).
        n_head_single: attention heads for pairformer single track (default 16).
        n_head_pair: attention heads for triangle attention (default 4).
        sigma_data: data std dev for EDM (default 16.0).
        eval_stride: mx.eval() every N pairformer blocks (default 4).
        tea_cache_threshold: TeaCache threshold (0 = disabled, >0 = enabled).
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_token: int = 768,
        c_s_inputs: int = 449,
        n_token_types: int = 32,
        r_max: int = 32,
        s_max: int = 2,
        n_pairformer_blocks: int = 16,
        n_diffusion_blocks: int = 8,
        n_confidence_blocks: int = 4,
        n_diffusion_steps: int = 5,
        n_head_single: int = 16,
        n_head_pair: int = 4,
        sigma_data: float = 16.0,
        eval_stride: int = 4,
        tea_cache_threshold: float = 0.0,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_inputs = c_s_inputs
        self.n_diffusion_steps = n_diffusion_steps
        self.eval_stride = eval_stride

        # --- 1. Input Embedders ---
        self.input_embedder = InputFeatureEmbedder(
            c_token=c_s,  # project token to c_s (not c_token, since c_s_inputs = c_s+65)
            n_token_types=n_token_types,
            c_s_inputs=c_s_inputs,
        )
        self.relative_position_encoding = RelativePositionEncoding(
            r_max=r_max, s_max=s_max, c_z=c_z
        )

        # --- Init projections (Algorithm 1, lines 4-5) ---
        # s_inputs -> s_init
        self.linear_s_init = LinearNoBias(c_s_inputs, c_s)
        # s_init -> pair init (outer product)
        self.linear_z_init1 = LinearNoBias(c_s, c_z)
        self.linear_z_init2 = LinearNoBias(c_s, c_z)

        # Token bond embedding (optional, 1->c_z)
        self.linear_token_bond = LinearNoBias(1, c_z)

        # Recycling projections (zero-initialized)
        self.ln_z_cycle = nn.LayerNorm(c_z)
        self.linear_z_cycle = LinearNoBias(c_z, c_z)
        self.ln_s_cycle = nn.LayerNorm(c_s)
        self.linear_s_cycle = LinearNoBias(c_s, c_s)

        # Zero-init recycling
        self.linear_z_cycle.weight = mx.zeros_like(self.linear_z_cycle.weight)
        self.linear_s_cycle.weight = mx.zeros_like(self.linear_s_cycle.weight)

        # --- 2. Pairformer trunk ---
        self.pairformer_stack = PairformerStack(
            n_blocks=n_pairformer_blocks,
            c_z=c_z,
            c_s=c_s,
            n_head_pair=n_head_pair,
            n_head_single=n_head_single,
            eval_stride=eval_stride,
        )

        # --- 3. Diffusion module ---
        self.diffusion_module = DiffusionModule(
            sigma_data=sigma_data,
            c_atom=128,
            c_atompair=16,
            c_token=c_token,
            c_s=c_s,
            c_z=c_z,
            c_s_inputs=c_s_inputs,
            n_atom_encoder_blocks=1,
            n_transformer_blocks=n_diffusion_blocks,
            n_atom_decoder_blocks=1,
            n_head=n_head_single,
            n_atom_head=n_head_pair,
        )

        # --- 4. Confidence head ---
        self.confidence_head = ConfidenceHead(
            c_s=c_s,
            c_z=c_z,
            c_s_inputs=c_s_inputs,
            n_blocks=n_confidence_blocks,
        )

        # --- 5. Distogram head (optional, for training loss) ---
        self.ln_distogram = nn.LayerNorm(c_z)
        self.linear_distogram = nn.Linear(c_z, 64)

        # --- ODE sampler (created on demand) ---
        self._n_diffusion_steps = n_diffusion_steps
        self._sigma_data = sigma_data
        self._tea_cache_threshold = tea_cache_threshold

    def _get_sampler(self) -> FlowMatchingODESampler:
        """Create the ODE sampler, optionally with TeaCache."""
        tea_cache = None
        if self._tea_cache_threshold > 0:
            tea_cache = TeaCache(
                threshold=self._tea_cache_threshold,
                sigma_data=self._sigma_data,
            )
        return FlowMatchingODESampler(
            diffusion_module=self.diffusion_module,
            n_steps=self._n_diffusion_steps,
            sigma_data=self._sigma_data,
            tea_cache=tea_cache,
        )

    def get_trunk_output(
        self,
        s_inputs: mx.array,
        residue_index: mx.array,
        chain_id: mx.array,
        entity_id: mx.array | None = None,
        token_index: mx.array | None = None,
        token_bonds: mx.array | None = None,
        n_cycle: int = 1,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Run embedding + trunk (Pairformer) pipeline.

        This is the pre-diffusion stage that computes s_trunk and z_trunk.

        Args:
            s_inputs: [B, N, c_s_inputs] from InputFeatureEmbedder
            residue_index: [B, N] residue indices
            chain_id: [B, N] chain IDs
            entity_id: [B, N] entity IDs (optional, defaults to chain_id)
            token_index: [B, N] token indices (optional, defaults to residue_index)
            token_bonds: [B, N, N] token bond features (optional)
            n_cycle: number of recycling cycles (default 1)

        Returns:
            (s_inputs, s_trunk, z_trunk)
        """
        # Compute relative position features
        relp = self.relative_position_encoding.encode(
            residue_index, chain_id, entity_id, token_index
        )

        # Init single representation
        s_init = self.linear_s_init(s_inputs)

        # Init pair representation (outer product + relpos)
        z_init = (
            mx.expand_dims(self.linear_z_init1(s_init), axis=-2)
            + mx.expand_dims(self.linear_z_init2(s_init), axis=-3)
        )
        z_init = z_init + self.relative_position_encoding(relp)

        # Optional token bond embedding
        if token_bonds is not None:
            z_init = z_init + self.linear_token_bond(
                mx.expand_dims(token_bonds, axis=-1)
            )

        # Recycling
        s = mx.zeros_like(s_init)
        z = mx.zeros_like(z_init)

        for _cycle in range(n_cycle):
            z = z_init + self.linear_z_cycle(self.ln_z_cycle(z))
            s = s_init + self.linear_s_cycle(self.ln_s_cycle(s))
            s, z = self.pairformer_stack(s, z)

        return s_inputs, s, z

    def __call__(
        self,
        seq: mx.array,
        residue_index: mx.array,
        chain_id: mx.array,
        entity_id: mx.array | None = None,
        token_index: mx.array | None = None,
        token_bonds: mx.array | None = None,
        atom_mask: mx.array | None = None,
        profile: mx.array | None = None,
        deletion_mean: mx.array | None = None,
        esm_embeddings: mx.array | None = None,
        n_cycle: int = 1,
        run_confidence: bool = True,
        coordinates_override: mx.array | None = None,
        ref_pos: mx.array | None = None,
        ref_charge: mx.array | None = None,
        ref_mask: mx.array | None = None,
    ) -> dict[str, mx.array]:
        """Full forward pass: embed -> trunk -> diffuse -> confidence.

        Args:
            seq: [B, N] integer sequence tokens (0..31)
            residue_index: [B, N] residue indices
            chain_id: [B, N] chain IDs (asymmetric unit)
            entity_id: [B, N] entity IDs (optional)
            token_index: [B, N] token indices (optional)
            token_bonds: [B, N, N] token bonds (optional)
            atom_mask: [B, N] atom/token mask (optional)
            profile: [B, N, 32] MSA profile (optional)
            deletion_mean: [B, N, 1] deletion mean (optional)
            esm_embeddings: [B, N, esm_dim] ESM embeddings (optional)
            n_cycle: recycling cycles (default 1)
            run_confidence: whether to run confidence head (default True)
            coordinates_override: [B, N, 3] override coordinates (optional,
                bypasses diffusion -- useful for confidence-only mode)
            ref_pos: [B, N, 3] reference atom positions (optional)
            ref_charge: [B, N] reference atom charges (optional)
            ref_mask: [B, N] atom mask (optional, defaults to ones)

        Returns:
            dict with keys:
                coordinates: [B, N, 3] predicted coordinates
                plddt: [B, N] per-residue pLDDT scores
                pae_logits: [B, N, N, 64] PAE logits
                pde_logits: [B, N, N, 64] PDE logits
                ptm: scalar pTM score
                iptm: scalar ipTM score (if multi-chain)
                s_trunk: [B, N, c_s] trunk single embeddings
                z_trunk: [B, N, N, c_z] trunk pair embeddings
        """
        B, N = seq.shape[:2]

        # 1. Embed inputs
        s_inputs = self.input_embedder(
            restype=seq,
            profile=profile,
            deletion_mean=deletion_mean,
            esm_embeddings=esm_embeddings,
        )

        # 2. Run trunk
        s_inputs, s_trunk, z_trunk = self.get_trunk_output(
            s_inputs=s_inputs,
            residue_index=residue_index,
            chain_id=chain_id,
            entity_id=entity_id,
            token_index=token_index,
            token_bonds=token_bonds,
            n_cycle=n_cycle,
        )

        mx.eval(s_trunk, z_trunk)

        # 3. Run diffusion (sample coordinates)
        if coordinates_override is not None:
            # Use provided coordinates (e.g. from RFdiffusion backbone)
            # Useful for confidence-only mode
            coordinates = coordinates_override
        else:
            sampler = self._get_sampler()
            # Initialize from noise
            x_init = mx.random.normal(shape=(B, N, 3)) * sampler.s_max
            coordinates = sampler.sample(
                x_init=x_init,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                ref_pos=ref_pos,
                ref_charge=ref_charge,
                ref_mask=ref_mask,
            )

        result = {
            "coordinates": coordinates,
            "s_trunk": s_trunk,
            "z_trunk": z_trunk,
            "s_inputs": s_inputs,
        }

        # 4. Run confidence head
        if run_confidence:
            confidence = self.confidence_head(
                x_pred=coordinates,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
            )
            result.update(confidence)

            # Compute ipTM if multi-chain
            n_chains = mx.max(chain_id) - mx.min(chain_id) + 1
            if n_chains > mx.array(1):
                iptm = compute_iptm(
                    pae_logits=confidence["pae_logits"],
                    chain_ids=chain_id,
                )
                result["iptm"] = iptm

        return result

    # ------------------------------------------------------------------
    # Optimization methods
    # ------------------------------------------------------------------

    def enable_mixed_precision(self):
        """Enable float16 for bandwidth-bound pair attention layers.

        Converts triangle multiplication and triangle attention weights
        to float16 for reduced memory bandwidth.
        """
        from rfantibody.protenix.mlx.weight_converter import _set_params_from_flat

        def _convert_fp16(module):
            flat = dict(nn.utils.tree_flatten(module.parameters()))
            fp16 = {k: v.astype(mx.float16) for k, v in flat.items()}
            _set_params_from_flat(module, fp16)

        for block in self.pairformer_stack.blocks:
            _convert_fp16(block.tri_mul_out)
            _convert_fp16(block.tri_mul_in)
            _convert_fp16(block.tri_att_start)
            _convert_fp16(block.tri_att_end)

        mx.eval(self.parameters())

    def set_eval_stride(self, stride: int):
        """Set mx.eval() frequency in the Pairformer trunk.

        Lower values = more frequent eval = less memory but slower.
        Higher values = better kernel fusion but more memory.

        Args:
            stride: evaluate every N blocks (default 4).
        """
        self.pairformer_stack.eval_stride = stride

    def set_n_pairformer_blocks(self, n: int):
        """Reduce active pairformer blocks for faster inference.

        Args:
            n: number of blocks to use (clamped to actual block count).
        """
        actual = len(self.pairformer_stack.blocks)
        n = min(n, actual)
        # Store original if not already saved
        if not hasattr(self.pairformer_stack, '_all_blocks'):
            self.pairformer_stack._all_blocks = self.pairformer_stack.blocks
        self.pairformer_stack.blocks = self.pairformer_stack._all_blocks[:n]

    def set_n_diffusion_steps(self, n: int):
        """Set number of ODE diffusion steps.

        Fewer steps = faster but lower quality. Default is 5.
        """
        self._n_diffusion_steps = n

    def enable_tea_cache(self, threshold: float = 0.15):
        """Enable TeaCache for diffusion acceleration.

        Args:
            threshold: change threshold (0.05-0.30). Lower = more accurate.
        """
        self._tea_cache_threshold = threshold

    def disable_tea_cache(self):
        """Disable TeaCache."""
        self._tea_cache_threshold = 0.0
