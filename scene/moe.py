#
# Mixture-of-Experts decoder for MoE-GS (Scene Experts).
# Each attribute head (opacity / cov / color) gets its own AttributeMoE.
#
# Dual-router (MoE+GNN):
#   z_base = BaseRouter(fv)           # isolated point-wise
#   z_gnn  = GnnRouterHead(h_gnn)     # spatial context from AnchorGNN
#   z      = z_base + γ * z_gnn
#   Top-k(softmax(z)) → mix experts on [fv, view, ...]
#

import torch
from torch import nn
from typing import Optional, List


def build_expert(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    out_activation: Optional[nn.Module] = None,
    depth: int = 2,
) -> nn.Sequential:
    """Scaffold-GS-style attribute MLP.

    ``depth`` is the number of ``Linear`` layers (default 2, as in Scaffold-GS /
    Scene Experts). ``depth>=3`` inserts extra hidden Linear+ReLU blocks for the
    naive depth-scaling analysis.
    """
    depth = max(2, int(depth))
    layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU(True)]
    for _ in range(depth - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True)])
    layers.append(nn.Linear(hidden_dim, out_dim))
    if out_activation is not None:
        layers.append(out_activation)
    return nn.Sequential(*layers)


def _new_activation(out_activation: Optional[nn.Module]) -> Optional[nn.Module]:
    """Create a fresh activation instance (avoid sharing one module across experts)."""
    if out_activation is None:
        return None
    return type(out_activation)()


class AttributeMoE(nn.Module):
    """
    Sparsely-gated MoE for one Gaussian attribute.

    Base router: fv (point-wise).
    Optional GNN router head: spatial context h_gnn → logits fused as
        z = z_base + γ * z_gnn
    Experts always take Scaffold expert input (raw fv + view [+ dist] [+ appearance]).
    """

    def __init__(
        self,
        router_dim: int,
        expert_in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        out_activation: Optional[nn.Module] = None,
        expert_noise_std: float = 2e-2,
        router_init_std: float = 0.1,
        use_gnn_router: bool = False,
        gnn_gamma_init: float = 0.5,
        mlp_depth: int = 2,
    ):
        super().__init__()
        assert num_experts >= 1
        assert 1 <= top_k <= num_experts

        self.router_dim = router_dim
        self.expert_in_dim = expert_in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.sparse = False
        self.expert_noise_std = expert_noise_std
        self.router_init_std = router_init_std
        self.use_gnn_router = bool(use_gnn_router) and num_experts > 1
        self.mlp_depth = max(2, int(mlp_depth))

        # Base router (isolated point-wise on fv) → logits A
        self.router = nn.Linear(router_dim, num_experts)

        # GNN router head on [fv ‖ h_gnn] → logits B (spatial+local, distinct from base)
        if self.use_gnn_router:
            self.gnn_router = nn.Linear(router_dim * 2, num_experts)
            self.gnn_gamma = nn.Parameter(torch.tensor(float(gnn_gamma_init)))
        else:
            self.gnn_router = None
            self.gnn_gamma = None

        self.experts = nn.ModuleList(
            [
                build_expert(
                    expert_in_dim,
                    hidden_dim,
                    out_dim,
                    _new_activation(out_activation),
                    depth=self.mlp_depth,
                )
                for _ in range(num_experts)
            ]
        )

        if self.num_experts > 1:
            self._curriculum_init()

        self._last_gates: Optional[torch.Tensor] = None
        self._last_selected: Optional[torch.Tensor] = None

    def _curriculum_init(self):
        with torch.no_grad():
            ref_state = self.experts[0].state_dict()
            for expert in self.experts[1:]:
                expert.load_state_dict(ref_state)
                if self.expert_noise_std > 0:
                    for p in expert.parameters():
                        p.add_(self.expert_noise_std * torch.randn_like(p))
            nn.init.normal_(self.router.weight, mean=0.0, std=self.router_init_std)
            nn.init.zeros_(self.router.bias)
            if self.gnn_router is not None:
                nn.init.normal_(self.gnn_router.weight, mean=0.0, std=self.router_init_std)
                nn.init.zeros_(self.gnn_router.bias)

    @property
    def use_moe(self) -> bool:
        return self.num_experts > 1

    def set_sparse(self, sparse: bool):
        self.sparse = bool(sparse) and self.num_experts > 1

    def forward(
        self,
        feat: torch.Tensor,
        expert_input: torch.Tensor,
        spatial_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            feat: [N, router_dim] raw anchor features fv (base router + expert path)
            expert_input: [N, expert_in_dim] view-conditioned decoder input
            spatial_feat: [N, router_dim] optional GNN spatial context h_gnn
        """
        if self.num_experts == 1:
            out = self.experts[0](expert_input)
            if self.training:
                gates = torch.ones(feat.shape[0], 1, device=feat.device, dtype=feat.dtype)
                self._last_gates = gates
                self._last_selected = gates
            return out

        # Logits A: base (point-wise)
        z_base = self.router(feat)

        # Logits B + fusion: z = z_base + γ * z_gnn
        if self.use_gnn_router and spatial_feat is not None and self.gnn_router is not None:
            z_gnn = self.gnn_router(torch.cat([feat, spatial_feat], dim=-1))
            logits = z_base + self.gnn_gamma * z_gnn
        else:
            logits = z_base

        if self.training and (not self.sparse):
            logits = logits + 0.05 * torch.randn_like(logits)

        gates = torch.softmax(logits, dim=-1)

        if self.sparse and self.top_k < self.num_experts:
            topk_vals, topk_idx = torch.topk(gates, self.top_k, dim=-1)
            topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            weights = torch.zeros_like(gates)
            weights.scatter_(1, topk_idx, topk_vals)
            selected = torch.zeros_like(gates)
            selected.scatter_(1, topk_idx, 1.0)
        else:
            weights = gates
            selected = torch.ones_like(gates)

        expert_outs = torch.stack(
            [expert(expert_input) for expert in self.experts], dim=1
        )
        out = (weights.unsqueeze(-1) * expert_outs).sum(dim=1)

        if self.training:
            self._last_gates = gates
            self._last_selected = selected

        return out

    def load_balancing_loss(self) -> torch.Tensor:
        if self._last_gates is None or self.num_experts <= 1:
            device = next(self.parameters()).device
            return torch.zeros((), device=device)

        gates = self._last_gates
        n_exp = float(self.num_experts)
        P = gates.mean(dim=0)
        if self.sparse and self._last_selected is not None:
            f = self._last_selected.mean(dim=0)
        else:
            f = P
        return n_exp * (P * f).sum()

    def router_entropy_loss(self) -> torch.Tensor:
        if self._last_gates is None or self.num_experts <= 1:
            device = next(self.parameters()).device
            return torch.zeros((), device=device)
        p = self._last_gates.clamp_min(1e-8)
        return -(p * p.log()).sum(dim=-1).mean()
