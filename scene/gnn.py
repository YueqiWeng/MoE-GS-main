#
# Anchor GNN for MoE+GNN dual-router architecture.
# Spatial k-NN graph → attention-weighted spatial context for the GNN router only.
# Experts always keep raw fv (no feature enrichment).
#

import torch
from torch import nn
from scipy.spatial import cKDTree


def knn_indices_and_dists(pos: torch.Tensor, k: int):
    """
    k nearest neighbors excluding self + Euclidean distances.
    CPU cKDTree keeps GPU memory safe at large N.
    """
    n = pos.shape[0]
    device = pos.device
    if n <= 1:
        empty_i = torch.zeros(n, 0, dtype=torch.long, device=device)
        empty_d = torch.zeros(n, 0, dtype=pos.dtype, device=device)
        return empty_i, empty_d

    k = min(int(k), n - 1)
    pos_cpu = pos.detach().float().cpu().numpy()
    tree = cKDTree(pos_cpu)
    dists, idx = tree.query(pos_cpu, k=k + 1, workers=-1)
    idx = torch.as_tensor(idx, dtype=torch.long, device=device)
    dists = torch.as_tensor(dists, dtype=pos.dtype, device=device)
    if idx.ndim == 1:
        idx = idx.unsqueeze(0)
        dists = dists.unsqueeze(0)
    return idx[:, 1:], dists[:, 1:]


class NeighborAttention(nn.Module):
    """
    Memory-efficient learned attention over kNN neighbors (chunked over nodes).
    score_ij = (Wq h_i · Wk h_j) / sqrt(d) + w_d * dist_ij
    w_ij = softmax_j(score_ij);  neighbor_i = sum_j w_ij h_j
    """

    def __init__(self, feat_dim: int, attn_dim: int = 16):
        super().__init__()
        self.attn_dim = attn_dim
        self.wq = nn.Linear(feat_dim, attn_dim, bias=False)
        self.wk = nn.Linear(feat_dim, attn_dim, bias=False)
        self.wd = nn.Linear(1, 1, bias=False)
        self.scale = attn_dim ** -0.5

    def forward(
        self,
        h: torch.Tensor,
        idx: torch.Tensor,
        dists: torch.Tensor,
        chunk_size: int = 2048,
    ) -> torch.Tensor:
        n, k, c = h.shape[0], idx.shape[1], h.shape[1]
        q = self.wq(h)  # [N, d]
        k_all = self.wk(h)  # [N, d]
        out = h.new_empty(n, c)
        chunk_size = max(1, int(chunk_size))

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            qi = q[start:end].unsqueeze(1)  # [B, 1, d]
            kj = k_all[idx[start:end]]  # [B, K, d]
            logits = (qi * kj).sum(dim=-1) * self.scale  # [B, K]
            logits = logits + self.wd(dists[start:end].unsqueeze(-1)).squeeze(-1)
            w = torch.softmax(logits, dim=-1)
            hj = h[idx[start:end]]  # [B, K, C]
            out[start:end] = (hj * w.unsqueeze(-1)).sum(dim=1)
        return out


class AnchorGNN(nn.Module):
    """
    Attention-weighted message passing on spatial k-NN → h_gnn [N, feat_dim].
    Feeds only the GNN router branch (logits B).
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        knn_k: int = 16,
        rebuild_every: int = 100,
        attn_chunk: int = 2048,
        attn_dim: int = 16,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.knn_k = knn_k
        self.rebuild_every = max(1, int(rebuild_every))
        self.attn_chunk = max(1, int(attn_chunk))

        self.attn_layers = nn.ModuleList(
            [NeighborAttention(feat_dim, attn_dim) for _ in range(num_layers)]
        )
        self.msg_layers = nn.ModuleList(
            [nn.Linear(feat_dim * 2, hidden_dim) for _ in range(num_layers)]
        )
        self.upd_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, feat_dim) for _ in range(num_layers)]
        )
        for upd in self.upd_layers:
            nn.init.normal_(upd.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(upd.bias)

        self._cached_n = -1
        self._cached_idx = None
        self._cached_dists = None
        self._steps = 0

    def invalidate_cache(self):
        self._cached_n = -1
        self._cached_idx = None
        self._cached_dists = None

    def _neighbor_graph(self, pos: torch.Tensor):
        n = pos.shape[0]
        self._steps += 1
        need = (
            self._cached_idx is None
            or self._cached_n != n
            or (self.training and (self._steps % self.rebuild_every == 0))
        )
        if not need:
            return self._cached_idx, self._cached_dists

        with torch.no_grad():
            idx, dists = knn_indices_and_dists(pos.detach(), self.knn_k)

        self._cached_n = n
        self._cached_idx = idx
        self._cached_dists = dists
        return idx, dists

    def forward(self, pos: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if feat.shape[0] <= 1 or self.knn_k <= 0:
            return feat

        idx, dists = self._neighbor_graph(pos)
        h = feat
        for attn, msg_lin, upd_lin in zip(self.attn_layers, self.msg_layers, self.upd_layers):
            neighbor = attn(h, idx, dists, chunk_size=self.attn_chunk)
            h = h + upd_lin(torch.relu(msg_lin(torch.cat([h, neighbor], dim=-1))))
        return h
