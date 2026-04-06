"""
MLX track modules for RFdiffusion.

Three-track architecture: MSA, Pair, Structure.
"""

import mlx.core as mx
import mlx.nn as nn

from .attention import (
    FeedForwardLayer, MSARowAttentionWithBias, MSAColAttention,
    MSAColGlobalAttention, BiasedAxialAttention,
)
from .se3.wrapper import SE3TransformerWrapper
from .util_module import rbf, cdist, get_seqsep, make_topk_graph, make_full_graph


class MSAPairStr2MSA(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_state=16,
                 d_hidden=32, p_drop=0.15, use_global_attn=False):
        super().__init__()
        self.norm_pair = nn.LayerNorm(d_pair)
        self.proj_pair = nn.Linear(d_pair + 36, d_pair)
        self.norm_state = nn.LayerNorm(d_state)
        self.proj_state = nn.Linear(d_state, d_msa)
        self.row_attn = MSARowAttentionWithBias(
            d_msa=d_msa, d_pair=d_pair, n_head=n_head, d_hidden=d_hidden)
        if use_global_attn:
            self.col_attn = MSAColGlobalAttention(
                d_msa=d_msa, n_head=n_head, d_hidden=d_hidden)
        else:
            self.col_attn = MSAColAttention(
                d_msa=d_msa, n_head=n_head, d_hidden=d_hidden)
        self.ff = FeedForwardLayer(d_msa, 4, p_drop=p_drop)

    def __call__(self, msa, pair, rbf_feat, state):
        B, N, L = msa.shape[:3]
        pair = self.norm_pair(pair)
        pair = mx.concatenate([pair, rbf_feat], axis=-1)
        pair = self.proj_pair(pair)

        state = self.norm_state(state)
        state = self.proj_state(state).reshape(B, 1, L, -1)
        # index_add: msa[:,0] += state
        msa = msa.at[:, 0].add(state.squeeze(1))

        msa = msa + self.row_attn(msa, pair)
        msa = msa + self.col_attn(msa)
        msa = msa + self.ff(msa)
        return msa


class PairStr2Pair(nn.Module):
    def __init__(self, d_pair=128, n_head=4, d_hidden=32, d_rbf=36, p_drop=0.15):
        super().__init__()
        self.emb_rbf = nn.Linear(d_rbf, d_hidden)
        self.proj_rbf = nn.Linear(d_hidden, d_pair)
        self.row_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden,
                                             p_drop=p_drop, is_row=True)
        self.col_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden,
                                             p_drop=p_drop, is_row=False)
        self.ff = FeedForwardLayer(d_pair, 2)
        self._use_fp16 = False

    def __call__(self, pair, rbf_feat):
        if self._use_fp16:
            orig_dtype = pair.dtype
            pair = pair.astype(mx.float16)
            rbf_feat = rbf_feat.astype(mx.float16)
        rbf_feat = self.proj_rbf(nn.relu(self.emb_rbf(rbf_feat)))
        pair = pair + self.row_attn(pair, rbf_feat)
        pair = pair + self.col_attn(pair, rbf_feat)
        pair = pair + self.ff(pair)
        if self._use_fp16:
            pair = pair.astype(orig_dtype)
        return pair


class MSA2Pair(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_hidden=32, p_drop=0.15):
        super().__init__()
        self.norm = nn.LayerNorm(d_msa)
        self.proj_left = nn.Linear(d_msa, d_hidden)
        self.proj_right = nn.Linear(d_msa, d_hidden)
        self.proj_out = nn.Linear(d_hidden * d_hidden, d_pair)

    def __call__(self, msa, pair):
        B, N, L = msa.shape[:3]
        msa = self.norm(msa)
        left = self.proj_left(msa)
        right = self.proj_right(msa) / float(N)
        out = mx.einsum('bsli,bsmj->blmij', left, right).reshape(B, L, L, -1)
        out = self.proj_out(out)
        return pair + out


class SCPred(nn.Module):
    def __init__(self, d_msa=256, d_state=32, d_hidden=128, p_drop=0.15):
        super().__init__()
        self.norm_s0 = nn.LayerNorm(d_msa)
        self.norm_si = nn.LayerNorm(d_state)
        self.linear_s0 = nn.Linear(d_msa, d_hidden)
        self.linear_si = nn.Linear(d_state, d_hidden)
        self.linear_1 = nn.Linear(d_hidden, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_hidden)
        self.linear_3 = nn.Linear(d_hidden, d_hidden)
        self.linear_4 = nn.Linear(d_hidden, d_hidden)
        self.linear_out = nn.Linear(d_hidden, 20)

    def __call__(self, seq, state):
        B, L = seq.shape[:2]
        seq = self.norm_s0(seq)
        state = self.norm_si(state)
        si = self.linear_s0(seq) + self.linear_si(state)
        si = si + self.linear_2(nn.relu(self.linear_1(nn.relu(si))))
        si = si + self.linear_4(nn.relu(self.linear_3(nn.relu(si))))
        si = self.linear_out(nn.relu(si))
        return si.reshape(B, L, 10, 2)


class Str2Str(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_state=16,
                 SE3_param=None, p_drop=0.1):
        super().__init__()
        if SE3_param is None:
            SE3_param = {'l0_in_features': 32, 'l0_out_features': 16,
                         'num_edge_features': 32}
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_state = nn.LayerNorm(d_state)
        self.embed_x = nn.Linear(d_msa + d_state, SE3_param['l0_in_features'])
        self.embed_e1 = nn.Linear(d_pair, SE3_param['num_edge_features'])
        self.embed_e2 = nn.Linear(
            SE3_param['num_edge_features'] + 36 + 1,
            SE3_param['num_edge_features'])
        self.norm_node = nn.LayerNorm(SE3_param['l0_in_features'])
        self.norm_edge1 = nn.LayerNorm(SE3_param['num_edge_features'])
        self.norm_edge2 = nn.LayerNorm(SE3_param['num_edge_features'])
        self.se3 = SE3TransformerWrapper(**SE3_param)
        self.sc_predictor = SCPred(
            d_msa=d_msa, d_state=SE3_param['l0_out_features'], p_drop=p_drop)

    def __call__(self, msa, pair, R_in, T_in, xyz, state, idx,
                 motif_mask=None, top_k=64, eps=1e-5):
        B, N, L = msa.shape[:3]
        if motif_mask is None:
            motif_mask = mx.zeros(L, dtype=mx.bool_)

        node = self.norm_msa(msa[:, 0])
        pair = self.norm_pair(pair)
        state = self.norm_state(state)

        node = mx.concatenate([node, state], axis=-1)
        node = self.norm_node(self.embed_x(node))
        pair = self.norm_edge1(self.embed_e1(pair))

        neighbor = get_seqsep(idx)
        ca = xyz[:, :, 1]
        rbf_feat = rbf(cdist(ca, ca))
        pair = mx.concatenate([pair, rbf_feat, neighbor], axis=-1)
        pair = self.norm_edge2(self.embed_e2(pair))

        if top_k != 0:
            G, edge_feats = make_topk_graph(ca, pair, idx, top_k=top_k)
        else:
            G, edge_feats = make_full_graph(ca, pair, idx, top_k=top_k)

        l1_feats = xyz - mx.expand_dims(xyz[:, :, 1, :], 2)
        l1_feats = l1_feats.reshape(B * L, -1, 3)

        shift = self.se3(G,
                         node.reshape(B * L, -1, 1),
                         l1_feats,
                         edge_feats)

        state = shift['0'].reshape(B, L, -1)
        offset = shift['1'].reshape(B, L, 2, 3)

        # Freeze motif positions
        motif_3d = mx.expand_dims(mx.expand_dims(motif_mask, -1), -1)
        offset = mx.where(
            mx.broadcast_to(motif_3d, offset.shape),
            mx.zeros_like(offset), offset)

        delTi = offset[:, :, 0, :] / 10.0
        R = offset[:, :, 1, :] / 100.0

        Qnorm = mx.sqrt(1 + mx.sum(R * R, axis=-1))
        qA = 1.0 / Qnorm
        qB = R[:, :, 0] / Qnorm
        qC = R[:, :, 1] / Qnorm
        qD = R[:, :, 2] / Qnorm

        # Build rotation matrix from quaternion components (functional, no scatter)
        delRi = mx.stack([
            mx.stack([qA*qA + qB*qB - qC*qC - qD*qD,
                       2*qB*qC - 2*qA*qD,
                       2*qB*qD + 2*qA*qC], axis=-1),
            mx.stack([2*qB*qC + 2*qA*qD,
                       qA*qA - qB*qB + qC*qC - qD*qD,
                       2*qC*qD - 2*qA*qB], axis=-1),
            mx.stack([2*qB*qD - 2*qA*qC,
                       2*qC*qD + 2*qA*qB,
                       qA*qA - qB*qB - qC*qC + qD*qD], axis=-1),
        ], axis=-2)

        Ri = mx.einsum('bnij,bnjk->bnik', delRi, R_in)
        Ti = delTi + T_in

        alpha = self.sc_predictor(msa[:, 0], state)
        return Ri, Ti, state, alpha


class IterBlock(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head_msa=8, n_head_pair=4,
                 use_global_attn=False, d_hidden=32, d_hidden_msa=None,
                 p_drop=0.15, SE3_param=None):
        super().__init__()
        if SE3_param is None:
            SE3_param = {'l0_in_features': 32, 'l0_out_features': 16,
                         'num_edge_features': 32}
        if d_hidden_msa is None:
            d_hidden_msa = d_hidden

        self.msa2msa = MSAPairStr2MSA(
            d_msa=d_msa, d_pair=d_pair, n_head=n_head_msa,
            d_state=SE3_param['l0_out_features'],
            use_global_attn=use_global_attn,
            d_hidden=d_hidden_msa, p_drop=p_drop)
        self.msa2pair = MSA2Pair(
            d_msa=d_msa, d_pair=d_pair,
            d_hidden=d_hidden // 2, p_drop=p_drop)
        self.pair2pair = PairStr2Pair(
            d_pair=d_pair, n_head=n_head_pair,
            d_hidden=d_hidden, p_drop=p_drop)
        self.str2str = Str2Str(
            d_msa=d_msa, d_pair=d_pair,
            d_state=SE3_param['l0_out_features'],
            SE3_param=SE3_param, p_drop=p_drop)

    def __call__(self, msa, pair, R_in, T_in, xyz, state, idx,
                 motif_mask=None, use_checkpoint=False, top_k=0,
                 skip_se3=False):
        ca = xyz[:, :, 1, :]
        rbf_feat = rbf(cdist(ca, ca))
        msa = self.msa2msa(msa, pair, rbf_feat, state)
        pair = self.msa2pair(msa, pair)
        pair = self.pair2pair(pair, rbf_feat)
        if skip_se3:
            B, L = pair.shape[:2]
            alpha = mx.zeros((B, L, 10, 2))
            return msa, pair, R_in, T_in, state, alpha
        R, T, state, alpha = self.str2str(
            msa, pair, R_in, T_in, xyz, state, idx,
            motif_mask=motif_mask, top_k=top_k)
        return msa, pair, R, T, state, alpha


class IterativeSimulator(nn.Module):
    def __init__(self, n_extra_block=4, n_main_block=12, n_ref_block=4,
                 d_msa=256, d_msa_full=64, d_pair=128, d_hidden=32,
                 n_head_msa=8, n_head_pair=4,
                 SE3_param_full=None, SE3_param_topk=None,
                 p_drop=0.15):
        super().__init__()
        if SE3_param_full is None:
            SE3_param_full = {'l0_in_features': 32, 'l0_out_features': 16,
                              'num_edge_features': 32}
        if SE3_param_topk is None:
            SE3_param_topk = {'l0_in_features': 32, 'l0_out_features': 16,
                              'num_edge_features': 32}

        self.n_extra_block = n_extra_block
        self.n_main_block = n_main_block
        self.n_ref_block = n_ref_block

        self.proj_state = nn.Linear(
            SE3_param_topk['l0_out_features'],
            SE3_param_full['l0_out_features'])

        if n_extra_block > 0:
            self.extra_block = [
                IterBlock(d_msa=d_msa_full, d_pair=d_pair,
                          n_head_msa=n_head_msa, n_head_pair=n_head_pair,
                          d_hidden_msa=8, d_hidden=d_hidden,
                          p_drop=p_drop, use_global_attn=True,
                          SE3_param=SE3_param_full)
                for _ in range(n_extra_block)
            ]

        if n_main_block > 0:
            self.main_block = [
                IterBlock(d_msa=d_msa, d_pair=d_pair,
                          n_head_msa=n_head_msa, n_head_pair=n_head_pair,
                          d_hidden=d_hidden, p_drop=p_drop,
                          use_global_attn=False,
                          SE3_param=SE3_param_full)
                for _ in range(n_main_block)
            ]

        self.proj_state2 = nn.Linear(
            SE3_param_full['l0_out_features'],
            SE3_param_topk['l0_out_features'])

        if n_ref_block > 0:
            self.str_refiner = Str2Str(
                d_msa=d_msa, d_pair=d_pair,
                d_state=SE3_param_topk['l0_out_features'],
                SE3_param=SE3_param_topk, p_drop=p_drop)

    def __call__(self, seq, msa, msa_full, pair, xyz_in, state, idx,
                 use_checkpoint=False, motif_mask=None):
        B, L = pair.shape[:2]
        if motif_mask is None:
            motif_mask = mx.zeros(L, dtype=mx.bool_)

        # top_k=0 means full graph (default). Set top_k_override > 0 for k-NN graph.
        top_k = getattr(self, 'top_k_override', 0)

        R_in = mx.broadcast_to(
            mx.eye(3).reshape(1, 1, 3, 3), (B, L, 3, 3))
        T_in = mx.array(xyz_in[:, :, 1])  # copy
        xyz_in = xyz_in - mx.expand_dims(T_in, -2)

        state = self.proj_state(state)

        R_s, T_s, alpha_s = [], [], []

        eval_stride = getattr(self, 'eval_stride', 1)

        for i_m in range(self.n_extra_block):
            R_in = mx.stop_gradient(R_in)
            T_in = mx.stop_gradient(T_in)
            xyz = mx.einsum('bnij,bnaj->bnai', R_in, xyz_in) + mx.expand_dims(T_in, -2)

            msa_full, pair, R_in, T_in, state, alpha = self.extra_block[i_m](
                msa_full, pair, R_in, T_in, xyz, state, idx,
                motif_mask=motif_mask, top_k=top_k)
            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)
            if (i_m + 1) % eval_stride == 0 or i_m == self.n_extra_block - 1:
                mx.eval(msa_full, pair, R_in, T_in, state, alpha)

        se3_stride = getattr(self, 'se3_stride', 1)
        n_main = getattr(self, 'n_main_block_infer', self.n_main_block)

        for i_m in range(n_main):
            R_in = mx.stop_gradient(R_in)
            T_in = mx.stop_gradient(T_in)
            xyz = mx.einsum('bnij,bnaj->bnai', R_in, xyz_in) + mx.expand_dims(T_in, -2)

            skip = (se3_stride > 1 and i_m % se3_stride != 0)
            msa, pair, R_in, T_in, state, alpha = self.main_block[i_m](
                msa, pair, R_in, T_in, xyz, state, idx,
                motif_mask=motif_mask, top_k=top_k, skip_se3=skip)
            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)
            if (i_m + 1) % eval_stride == 0 or i_m == n_main - 1:
                mx.eval(msa, pair, R_in, T_in, state, alpha)

        state = self.proj_state2(state)
        for i_m in range(self.n_ref_block):
            R_in = mx.stop_gradient(R_in)
            T_in = mx.stop_gradient(T_in)
            xyz = mx.einsum('bnij,bnaj->bnai', R_in, xyz_in) + mx.expand_dims(T_in, -2)

            R_in, T_in, state, alpha = self.str_refiner(
                msa, pair, R_in, T_in, xyz, state, idx,
                top_k=64, motif_mask=motif_mask)
            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)
            if (i_m + 1) % eval_stride == 0 or i_m == self.n_ref_block - 1:
                mx.eval(R_in, T_in, state, alpha)

        R_s = mx.stack(R_s, axis=0)
        T_s = mx.stack(T_s, axis=0)
        alpha_s = mx.stack(alpha_s, axis=0)

        return msa, pair, R_s, T_s, alpha_s, state
