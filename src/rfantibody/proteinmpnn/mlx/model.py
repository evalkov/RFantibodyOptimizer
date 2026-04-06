"""
ProteinMPNN ported to MLX for Apple Silicon inference.

Architecture: k-NN graph → 3 encoder layers → 3 decoder layers → autoregressive sampling.
Port of src/rfantibody/proteinmpnn/model/protein_mpnn_utils.py.
"""
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Gather helpers (replace torch.gather patterns)
# ---------------------------------------------------------------------------

def gather_edges(edges, neighbor_idx):
    """Features [B,N,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]."""
    B, N, K = neighbor_idx.shape
    b_idx = mx.arange(B)[:, None, None]            # (B,1,1)
    n_idx = mx.arange(N)[None, :, None]             # (1,N,1)
    return edges[b_idx, n_idx, neighbor_idx]         # (B,N,K,C)


def gather_nodes(nodes, neighbor_idx):
    """Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]."""
    B, N, K = neighbor_idx.shape
    b_idx = mx.arange(B)[:, None, None]
    return nodes[b_idx, neighbor_idx]                # (B,N,K,C)


def gather_nodes_t(nodes, neighbor_idx):
    """Features [B,N,C] at Neighbor index [B,K] => [B,K,C]."""
    B, K = neighbor_idx.shape
    b_idx = mx.arange(B)[:, None]
    return nodes[b_idx, neighbor_idx]


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    return mx.concatenate([h_neighbors, h_nodes], axis=-1)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff)
        self.W_out = nn.Linear(num_ff, num_hidden)

    def __call__(self, h_V):
        return self.W_out(nn.gelu(self.W_in(h_V)))


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def __call__(self, offset, mask):
        mrf = self.max_relative_feature
        d = mx.clip(offset + mrf, 0, 2 * mrf) * mask + (1 - mask) * (2 * mrf + 1)
        d_onehot = mx.eye(2 * mrf + 1 + 1)[d.astype(mx.int32)]
        return self.linear(d_onehot.astype(mx.float32))


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W2 = nn.Linear(num_hidden, num_hidden)
        self.W3 = nn.Linear(num_hidden, num_hidden)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W12 = nn.Linear(num_hidden, num_hidden)
        self.W13 = nn.Linear(num_hidden, num_hidden)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def __call__(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        # Node message passing
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = mx.expand_dims(h_V, axis=-2)
        h_V_expand = mx.broadcast_to(h_V_expand,
            h_V.shape[:2] + (h_EV.shape[-2],) + (h_V.shape[-1],))
        h_EV = mx.concatenate([h_V_expand, h_EV], axis=-1)
        h_message = self.W3(nn.gelu(self.W2(nn.gelu(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mx.expand_dims(mask_attend, axis=-1) * h_message
        dh = mx.sum(h_message, axis=-2) / self.scale
        h_V = self.norm1(h_V + dh)

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + dh)
        if mask_V is not None:
            h_V = mx.expand_dims(mask_V, axis=-1) * h_V

        # Edge message passing
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = mx.expand_dims(h_V, axis=-2)
        h_V_expand = mx.broadcast_to(h_V_expand,
            h_V.shape[:2] + (h_EV.shape[-2],) + (h_V.shape[-1],))
        h_EV = mx.concatenate([h_V_expand, h_EV], axis=-1)
        h_message = self.W13(nn.gelu(self.W12(nn.gelu(self.W11(h_EV)))))
        h_E = self.norm3(h_E + h_message)
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W2 = nn.Linear(num_hidden, num_hidden)
        self.W3 = nn.Linear(num_hidden, num_hidden)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def __call__(self, h_V, h_E, mask_V=None, mask_attend=None):
        h_V_expand = mx.expand_dims(h_V, axis=-2)
        h_V_expand = mx.broadcast_to(h_V_expand,
            h_V.shape[:2] + (h_E.shape[-2],) + (h_V.shape[-1],))
        h_EV = mx.concatenate([h_V_expand, h_E], axis=-1)

        h_message = self.W3(nn.gelu(self.W2(nn.gelu(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mx.expand_dims(mask_attend, axis=-1) * h_message
        dh = mx.sum(h_message, axis=-2) / self.scale
        h_V = self.norm1(h_V + dh)

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + dh)
        if mask_V is not None:
            h_V = mx.expand_dims(mask_V, axis=-1) * h_V
        return h_V


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        super().__init__()
        self.edge_features = edge_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = mx.expand_dims(mask, 1) * mx.expand_dims(mask, 2)
        dX = mx.expand_dims(X, 1) - mx.expand_dims(X, 2)
        D = mask_2D * mx.sqrt(mx.sum(dX ** 2, axis=3) + eps)
        D_max = mx.max(D, axis=-1, keepdims=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        k = min(self.top_k, X.shape[1])
        # MLX has no topk-smallest; use argsort
        E_idx = mx.argsort(D_adjust, axis=-1)[:, :, :k]
        B, N, K = E_idx.shape
        b_idx = mx.arange(B)[:, None, None]
        n_idx = mx.arange(N)[None, :, None]
        D_neighbors = D_adjust[b_idx, n_idx, E_idx]
        return D_neighbors, E_idx

    def _rbf(self, D):
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = mx.linspace(D_min, D_max, D_count).reshape(1, 1, 1, -1)
        D_sigma = (D_max - D_min) / D_count
        D_expand = mx.expand_dims(D, axis=-1)
        return mx.exp(-((D_expand - D_mu) / D_sigma) ** 2)

    def _get_rbf(self, A, B, E_idx):
        D_A_B = mx.sqrt(mx.sum(
            (A[:, :, None, :] - B[:, None, :, :]) ** 2, axis=-1) + 1e-6)
        # gather neighbor distances
        Bb, N, K = E_idx.shape
        b_idx = mx.arange(Bb)[:, None, None]
        n_idx = mx.arange(N)[None, :, None]
        D_A_B_neighbors = D_A_B[b_idx, n_idx, E_idx]
        return self._rbf(D_A_B_neighbors)

    def __call__(self, X, mask, residue_idx, chain_labels):
        if self.augment_eps > 0:
            X = X + self.augment_eps * mx.random.normal(X.shape)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        # MLX has no cross product; compute manually
        a = mx.stack([
            b[..., 1] * c[..., 2] - b[..., 2] * c[..., 1],
            b[..., 2] * c[..., 0] - b[..., 0] * c[..., 2],
            b[..., 0] * c[..., 1] - b[..., 1] * c[..., 0],
        ], axis=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))
        for A, B_atom in [
            (N, N), (C, C), (O, O), (Cb, Cb),
            (Ca, N), (Ca, C), (Ca, O), (Ca, Cb),
            (N, C), (N, O), (N, Cb),
            (Cb, C), (Cb, O), (O, C),
            (N, Ca), (C, Ca), (O, Ca), (Cb, Ca),
            (C, N), (O, N), (Cb, N),
            (C, Cb), (O, Cb), (C, O),
        ]:
            RBF_all.append(self._get_rbf(A, B_atom, E_idx))
        RBF_all = mx.concatenate(RBF_all, axis=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        Bb, Nn, K = E_idx.shape
        b_idx = mx.arange(Bb)[:, None, None]
        n_idx = mx.arange(Nn)[None, :, None]
        offset = offset[:, :, :, None]  # (B, N, N, 1) for gather_edges
        offset = offset[b_idx, n_idx, E_idx][:, :, :, 0]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).astype(mx.int32)
        E_chains = d_chains[:, :, :, None]
        E_chains = E_chains[b_idx, n_idx, E_idx][:, :, :, 0]
        E_positional = self.embeddings(offset.astype(mx.int32), E_chains)
        E = mx.concatenate([E_positional, RBF_all], axis=-1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ProteinMPNN(nn.Module):
    def __init__(self, num_letters, node_features, edge_features,
                 hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
                 vocab=21, k_neighbors=64, augment_eps=0.05, dropout=0.1,
                 ca_only=False):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(
            node_features, edge_features,
            top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        self.encoder_layers = [
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ]
        self.decoder_layers = [
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ]
        self.W_out = nn.Linear(hidden_dim, num_letters)

    def _encode(self, X, mask, residue_idx, chain_encoding_all):
        """Run feature extraction + encoder. Shared by forward() and sample()."""
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = mx.zeros((E.shape[0], E.shape[1], E.shape[-1]))
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mx.expand_dims(mask, -1), E_idx).squeeze(-1)
        mask_attend = mx.expand_dims(mask, -1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        return h_V, h_E, E_idx

    def _build_masks(self, chain_mask, mask, randn, E_idx):
        """Build autoregressive forward/backward masks."""
        decoding_order = mx.argsort(
            (chain_mask + 0.0001) * mx.abs(randn), axis=-1)
        mask_size = E_idx.shape[1]
        # Permutation matrix
        perm = mx.eye(mask_size)[decoding_order]  # (B, N, N) one-hot
        # order_mask_backward = perm^T @ tril(ones) @ perm
        tril = mx.tril(mx.ones((mask_size, mask_size)))
        # einsum 'ij, biq, bjp -> bqp' with (1-triu) = tril(offset=-1)
        lower = 1.0 - mx.triu(mx.ones((mask_size, mask_size)))
        order_mask_backward = mx.einsum('ij,biq,bjp->bqp', lower, perm, perm)

        B, N, K = E_idx.shape
        b_idx = mx.arange(B)[:, None, None]
        n_idx = mx.arange(N)[None, :, None]
        mask_attend = order_mask_backward[b_idx, n_idx, E_idx]
        mask_attend = mx.expand_dims(mask_attend, axis=-1)
        mask_1D = mask.reshape(mask.shape[0], mask.shape[1], 1, 1)
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)
        return decoding_order, mask_bw, mask_fw

    def __call__(self, X, S, mask, chain_M, residue_idx, chain_encoding_all,
                 randn, use_input_decoding_order=False, decoding_order=None):
        h_V, h_E, E_idx = self._encode(X, mask, residue_idx, chain_encoding_all)

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(mx.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask
        if not use_input_decoding_order:
            decoding_order, mask_bw, mask_fw = self._build_masks(
                chain_M, mask, randn, E_idx)
        else:
            _, mask_bw, mask_fw = self._build_masks(chain_M, mask, randn, E_idx)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return log_probs

    def sample(self, X, randn, S_true, chain_mask, chain_encoding_all,
               residue_idx, mask=None, temperature=1.0,
               omit_AAs_np=None, bias_AAs_np=None, chain_M_pos=None,
               omit_AA_mask=None, pssm_coef=None, pssm_bias=None,
               pssm_multi=None, pssm_log_odds_flag=None,
               pssm_log_odds_mask=None, pssm_bias_flag=None,
               bias_by_res=None):
        """Autoregressive sequence sampling."""
        h_V, h_E, E_idx = self._encode(X, mask, residue_idx, chain_encoding_all)

        chain_mask = chain_mask * chain_M_pos * mask
        decoding_order, mask_bw, mask_fw = self._build_masks(
            chain_mask, mask, randn, E_idx)

        N_batch, N_nodes = X.shape[0], X.shape[1]
        all_probs = mx.zeros((N_batch, N_nodes, 21))
        h_S = mx.zeros_like(h_V)
        S = mx.zeros((N_batch, N_nodes), dtype=mx.int32)
        h_V_stack = [h_V] + [mx.zeros_like(h_V) for _ in range(len(self.decoder_layers))]

        constant = mx.array(omit_AAs_np) if omit_AAs_np is not None else mx.zeros(21)
        constant_bias = mx.array(bias_AAs_np) if bias_AAs_np is not None else mx.zeros(21)
        omit_AA_mask_flag = omit_AA_mask is not None

        h_EX_encoder = cat_neighbors_nodes(mx.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        # Ensure inputs are evaluated before the sequential loop
        mx.eval(h_V, h_E, E_idx, h_EXV_encoder_fw, mask_bw, decoding_order, chain_mask, mask)
        for stack_item in h_V_stack:
            mx.eval(stack_item)

        for t_ in range(N_nodes):
            t = decoding_order[:, t_]  # (B,)
            chain_mask_gathered = chain_mask[mx.arange(N_batch), t][:, None]  # (B,1)
            mask_gathered = mask[mx.arange(N_batch), t][:, None]

            if bias_by_res is not None:
                bias_by_res_gathered = bias_by_res[mx.arange(N_batch), t]  # (B, 21)
            else:
                bias_by_res_gathered = mx.zeros((N_batch, 21))

            all_masked = (mx.max(mask_gathered) == 0)

            if all_masked:
                S_t = S_true[mx.arange(N_batch), t][:, None]
            else:
                # Gather per-position features
                b_idx = mx.arange(N_batch)
                E_idx_t = E_idx[b_idx, t][:, None, :]  # (B, 1, K)
                h_E_t = h_E[b_idx, t][:, None, :, :]   # (B, 1, K, C)
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = h_EXV_encoder_fw[b_idx, t][:, None, :, :]
                mask_t = mask[b_idx, t][:, None]

                for l, layer in enumerate(self.decoder_layers):
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = h_V_stack[l][b_idx, t][:, None, :]
                    mask_bw_t = mask_bw[b_idx, t][:, None, :, :]
                    h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_t
                    new_h = layer(h_V_t, h_ESV_t, mask_V=mask_t)
                    # scatter: h_V_stack[l+1][b, t[b]] = new_h[b, 0]
                    old_h = h_V_stack[l + 1][b_idx, t]
                    h_V_stack[l + 1] = h_V_stack[l + 1].at[b_idx, t].add(new_h[:, 0, :] - old_h)

                h_V_t = h_V_stack[-1][b_idx, t]  # (B, C)
                logits = self.W_out(h_V_t) / temperature
                probs = mx.softmax(
                    logits - constant[None, :] * 1e8
                    + constant_bias[None, :] / temperature
                    + bias_by_res_gathered / temperature,
                    axis=-1)

                if pssm_bias_flag and pssm_coef is not None:
                    pssm_coef_gathered = pssm_coef[b_idx, t][:, None]
                    pssm_bias_gathered = pssm_bias[b_idx, t]
                    probs = ((1 - pssm_multi * pssm_coef_gathered) * probs
                             + pssm_multi * pssm_coef_gathered * pssm_bias_gathered)
                if pssm_log_odds_flag and pssm_log_odds_mask is not None:
                    plom = pssm_log_odds_mask[b_idx, t]
                    probs_masked = probs * plom + probs * 0.001
                    probs = probs_masked / mx.sum(probs_masked, axis=-1, keepdims=True)
                if omit_AA_mask_flag:
                    oam = omit_AA_mask[b_idx, t]
                    probs_masked = probs * (1.0 - oam)
                    probs = probs_masked / mx.sum(probs_masked, axis=-1, keepdims=True)

                S_t = mx.random.categorical(mx.log(probs + 1e-8))[:, None]  # (B, 1)

                new_probs = chain_mask_gathered * probs  # (B,1) * (B,21) → (B,21)
                all_probs = all_probs.at[b_idx, t].add(new_probs - all_probs[b_idx, t])

            S_true_gathered = S_true[mx.arange(N_batch), t][:, None]
            S_t = (S_t * chain_mask_gathered + S_true_gathered * (1.0 - chain_mask_gathered)).astype(mx.int32)
            temp1 = self.W_s(S_t.squeeze(-1) if S_t.ndim > 1 else S_t)
            if temp1.ndim == 2:
                temp1 = temp1  # (B, C)
            new_h_S = temp1 if temp1.ndim == 2 else temp1[:, 0]
            h_S = h_S.at[mx.arange(N_batch), t].add(new_h_S - h_S[mx.arange(N_batch), t])
            new_S = S_t.squeeze(-1) if S_t.ndim > 1 else S_t
            S = S.at[mx.arange(N_batch), t].add(new_S - S[mx.arange(N_batch), t])

            # Eval every step to keep memory bounded
            mx.eval(h_S, S, all_probs)
            for stack_item in h_V_stack:
                mx.eval(stack_item)

        return {"S": S, "probs": all_probs, "decoding_order": decoding_order}
