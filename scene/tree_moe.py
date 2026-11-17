#
# Hierarchical Tree MoE (--moe_arch dynamic_tree; methods: moe_tree, tree_gnn).
# Canonical trainers: baseline | moe | moe_gnn (flat MoE + dual GNN).
#
#   Base router at each INTERNAL node: z = z_base(fv) [+ γ z_gnn if enabled]
#   Leaf experts decode raw view-conditioned input
#   Selective growth replaces busy leaves with router + children
#

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from scene.moe import build_expert, _new_activation


class DualRouter(nn.Module):
    """Point-wise + optional GNN dual router over `num_children` children."""

    def __init__(
        self,
        router_dim: int,
        num_children: int,
        use_gnn_router: bool = False,
        gnn_gamma_init: float = 0.5,
        router_init_std: float = 0.1,
    ):
        super().__init__()
        self.router_dim = router_dim
        self.num_children = num_children
        self.use_gnn_router = bool(use_gnn_router) and num_children > 1
        self.router = nn.Linear(router_dim, num_children)
        if self.use_gnn_router:
            self.gnn_router = nn.Linear(router_dim * 2, num_children)
            self.gnn_gamma = nn.Parameter(torch.tensor(float(gnn_gamma_init)))
        else:
            self.gnn_router = None
            self.gnn_gamma = None
        nn.init.normal_(self.router.weight, mean=0.0, std=router_init_std)
        nn.init.zeros_(self.router.bias)
        if self.gnn_router is not None:
            nn.init.normal_(self.gnn_router.weight, mean=0.0, std=router_init_std)
            nn.init.zeros_(self.gnn_router.bias)

    def forward(
        self,
        feat: torch.Tensor,
        spatial_feat: Optional[torch.Tensor] = None,
        add_noise: bool = False,
    ) -> torch.Tensor:
        z = self.router(feat)
        if self.use_gnn_router and spatial_feat is not None and self.gnn_router is not None:
            z = z + self.gnn_gamma * self.gnn_router(torch.cat([feat, spatial_feat], dim=-1))
        if add_noise:
            z = z + 0.05 * torch.randn_like(z)
        return z


class TreeNode(nn.Module):
    """
    Leaf: stores one expert MLP.
    Internal: DualRouter + ModuleList of child nodes.
    """

    def __init__(
        self,
        node_id: str,
        depth: int,
        is_leaf: bool,
        *,
        expert: Optional[nn.Module] = None,
        router: Optional[DualRouter] = None,
        child_nodes: Optional[nn.ModuleList] = None,
    ):
        super().__init__()
        self.node_id = node_id
        self.depth = depth
        self.is_leaf = is_leaf
        self.expert = expert
        self.router = router
        self.child_nodes = child_nodes if child_nodes is not None else nn.ModuleList()

    def leaf_nodes(self) -> List["TreeNode"]:
        if self.is_leaf:
            return [self]
        out: List[TreeNode] = []
        for c in self.child_nodes:
            out.extend(c.leaf_nodes())
        return out

    def describe(self) -> str:
        if self.is_leaf:
            return f"Leaf({self.node_id},d={self.depth})"
        kids = ", ".join(c.describe() for c in self.child_nodes)
        return f"Node({self.node_id},d={self.depth})[{kids}]"


class AttributeTreeMoE(nn.Module):
    """
    Dynamic hierarchical MoE (same call signature as AttributeMoE):

        forward(feat, expert_input, spatial_feat=None) -> [N, out_dim]

    Start with branch_factor root leaves; grow high-traffic / heterogeneous leaves
    into grow_factor children. Dual-router at internal nodes; experts use raw u.
    """

    def __init__(
        self,
        router_dim: int,
        expert_in_dim: int,
        hidden_dim: int,
        out_dim: int,
        branch_factor: int = 4,
        tree_depth: int = 2,
        top_k: int = 2,
        out_activation: Optional[nn.Module] = None,
        expert_noise_std: float = 2e-2,
        router_init_std: float = 0.1,
        use_gnn_router: bool = False,
        gnn_gamma_init: float = 0.5,
        dynamic: bool = False,
        max_leaves: int = 64,
        grow_factor: int = 0,
        max_total_grows: int = 64,
    ):
        super().__init__()
        assert branch_factor >= 2
        assert tree_depth >= 1
        grow_factor = int(grow_factor) if int(grow_factor) >= 2 else int(branch_factor)
        assert grow_factor >= 2
        assert 1 <= top_k <= max(branch_factor, grow_factor)

        self.router_dim = router_dim
        self.expert_in_dim = expert_in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.branch_factor = int(branch_factor)
        self.grow_factor = grow_factor
        self.tree_depth = int(tree_depth)
        self.max_depth = int(tree_depth)
        self.top_k = int(top_k)
        self.sparse = False
        self.expert_noise_std = expert_noise_std
        self.router_init_std = router_init_std
        self.use_gnn_router = bool(use_gnn_router)
        self.gnn_gamma_init = float(gnn_gamma_init)
        self.dynamic = bool(dynamic)
        self.max_leaves = int(max_leaves)
        self.max_total_grows = int(max_total_grows)
        self._out_activation_proto = out_activation

        self._node_counter = 0
        if self.dynamic:
            # Root is an internal router over `branch_factor` leaf experts.
            self.root = self._build_depth1_root()
        else:
            self.root = self._build_full_tree(depth=0, node_id="r")

        self._last_gates: Optional[torch.Tensor] = None  # soft leaf path probs [N, L]
        self._last_traffic_gates: Optional[torch.Tensor] = None  # Top-k mix path [N, L]
        self._last_selected: Optional[torch.Tensor] = None
        self._last_router_gates: List[torch.Tensor] = []  # soft gates per router
        self._last_router_selected: List[torch.Tensor] = []
        self._last_feat: Optional[torch.Tensor] = None

        # Dynamic growth stats (CPU scalars, updated during training)
        self._leaf_mass_sum: Dict[str, float] = {}
        self._leaf_loss_mass: Dict[str, float] = {}
        self._leaf_feat_m2: Dict[str, float] = {}  # weighted sum of per-dim feat variance
        self._leaf_steps: Dict[str, int] = {}
        self._growth_history: List[str] = []
        self._last_grow_debug: str = ""

        self._curriculum_init_leaves()

    # ------------------------------------------------------------------ build
    def _next_id(self, prefix: str) -> str:
        self._node_counter += 1
        return f"{prefix}{self._node_counter}"

    def _make_expert(self) -> nn.Sequential:
        return build_expert(
            self.expert_in_dim,
            self.hidden_dim,
            self.out_dim,
            _new_activation(self._out_activation_proto),
        )

    def _make_router(
        self,
        num_children: Optional[int] = None,
        use_gnn_router: Optional[bool] = None,
    ) -> DualRouter:
        n = self.branch_factor if num_children is None else int(num_children)
        use_gnn = self.use_gnn_router if use_gnn_router is None else bool(use_gnn_router)
        return DualRouter(
            self.router_dim,
            n,
            use_gnn_router=use_gnn,
            gnn_gamma_init=self.gnn_gamma_init,
            router_init_std=self.router_init_std,
        )

    def _build_depth1_root(self) -> TreeNode:
        children = nn.ModuleList()
        for i in range(self.branch_factor):
            lid = self._next_id(f"L{i}_")
            children.append(
                TreeNode(lid, depth=1, is_leaf=True, expert=self._make_expert())
            )
        # Root is an internal router over `branch_factor` leaf experts.
        # GNN enabled for dynamic v1 when use_gnn_router=True.
        return TreeNode(
            node_id="root",
            depth=0,
            is_leaf=False,
            router=self._make_router(),
            child_nodes=children,
        )

    def _build_full_tree(self, depth: int, node_id: str) -> TreeNode:
        if depth >= self.tree_depth:
            return TreeNode(node_id, depth=depth, is_leaf=True, expert=self._make_expert())
        children = nn.ModuleList()
        for i in range(self.branch_factor):
            cid = self._next_id(f"{node_id}_{i}_")
            children.append(self._build_full_tree(depth + 1, cid))
        return TreeNode(
            node_id=node_id,
            depth=depth,
            is_leaf=False,
            router=self._make_router(),
            child_nodes=children,
        )

    def _curriculum_init_leaves(self):
        leaves = self.leaf_nodes()
        if len(leaves) <= 1:
            return
        with torch.no_grad():
            ref = leaves[0].expert.state_dict()
            for leaf in leaves[1:]:
                leaf.expert.load_state_dict(ref)
                if self.expert_noise_std > 0:
                    for p in leaf.expert.parameters():
                        p.add_(self.expert_noise_std * torch.randn_like(p))

    # ----------------------------------------------------------------- helpers
    @property
    def use_moe(self) -> bool:
        return self.num_experts > 1

    @property
    def num_experts(self) -> int:
        return len(self.leaf_nodes())

    def leaf_nodes(self) -> List[TreeNode]:
        return self.root.leaf_nodes()

    @property
    def experts(self) -> nn.ModuleList:
        """Compatibility with AttributeMoE / JIT export (leaf experts only)."""
        return nn.ModuleList([leaf.expert for leaf in self.leaf_nodes()])

    def set_sparse(self, sparse: bool):
        self.sparse = bool(sparse) and max(self.branch_factor, self.grow_factor) > 1

    def describe(self) -> str:
        mode = "dynamic" if self.dynamic else "static"
        return f"TreeMoE[{mode}] leaves={self.num_experts} :: {self.root.describe()}"

    # ----------------------------------------------------------------- forward
    def _route_weights(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mix_weights, selected, soft_gates) over children, all [N, B].

        soft_gates = full softmax (matches flat AttributeMoE `_last_gates`).
        mix_weights = Top-k renormalized weights when sparse, else soft_gates.
        """
        soft_gates = torch.softmax(logits, dim=-1)
        n_children = soft_gates.shape[1]
        k = min(self.top_k, n_children)
        if self.sparse and k < n_children:
            topk_vals, topk_idx = torch.topk(soft_gates, k, dim=-1)
            topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            weights = torch.zeros_like(soft_gates)
            weights.scatter_(1, topk_idx, topk_vals)
            selected = torch.zeros_like(soft_gates)
            selected.scatter_(1, topk_idx, 1.0)
        else:
            weights = soft_gates
            selected = torch.ones_like(soft_gates)
        return weights, selected, soft_gates

    def _forward_node(
        self,
        node: TreeNode,
        feat: torch.Tensor,
        expert_input: torch.Tensor,
        spatial_feat: Optional[torch.Tensor],
        path_prob: torch.Tensor,
        soft_path_prob: torch.Tensor,
        leaf_probs: Dict[str, torch.Tensor],
        soft_leaf_probs: Dict[str, torch.Tensor],
        router_soft_acc: List[torch.Tensor],
        router_selected_acc: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        path_prob / soft_path_prob: [N] probability of reaching this node
        (Top-k mix path vs full-softmax path). Returns mixed expert output.
        """
        if node.is_leaf:
            assert node.expert is not None
            leaf_probs[node.node_id] = path_prob
            soft_leaf_probs[node.node_id] = soft_path_prob
            return node.expert(expert_input)

        assert node.router is not None
        add_noise = self.training and (not self.sparse)
        logits = node.router(feat, spatial_feat, add_noise=add_noise)
        weights, selected, soft_gates = self._route_weights(logits)
        router_soft_acc.append(soft_gates)
        router_selected_acc.append(selected)

        # Soft mix over children (Top-k zeros unused children when sparse)
        out = None
        for c_idx, child in enumerate(node.child_nodes):
            child_path = path_prob * weights[:, c_idx]
            child_soft_path = soft_path_prob * soft_gates[:, c_idx]
            child_out = self._forward_node(
                child,
                feat,
                expert_input,
                spatial_feat,
                child_path,
                child_soft_path,
                leaf_probs,
                soft_leaf_probs,
                router_soft_acc,
                router_selected_acc,
            )
            term = weights[:, c_idx].unsqueeze(-1) * child_out
            out = term if out is None else out + term
        assert out is not None
        return out

    def forward(
        self,
        feat: torch.Tensor,
        expert_input: torch.Tensor,
        spatial_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n = feat.shape[0]
        device = feat.device
        dtype = feat.dtype
        path0 = torch.ones(n, device=device, dtype=dtype)
        leaf_probs: Dict[str, torch.Tensor] = {}
        soft_leaf_probs: Dict[str, torch.Tensor] = {}
        router_soft_acc: List[torch.Tensor] = []
        router_selected_acc: List[torch.Tensor] = []

        out = self._forward_node(
            self.root,
            feat,
            expert_input,
            spatial_feat,
            path0,
            path0,
            leaf_probs,
            soft_leaf_probs,
            router_soft_acc,
            router_selected_acc,
        )

        leaves = self.leaf_nodes()
        if leaves:
            gate_mat = torch.stack(
                [
                    leaf_probs.get(leaf.node_id, torch.zeros(n, device=device, dtype=dtype))
                    for leaf in leaves
                ],
                dim=1,
            )
            soft_gate_mat = torch.stack(
                [
                    soft_leaf_probs.get(
                        leaf.node_id, torch.zeros(n, device=device, dtype=dtype)
                    )
                    for leaf in leaves
                ],
                dim=1,
            )
        else:
            gate_mat = torch.ones(n, 1, device=device, dtype=dtype)
            soft_gate_mat = gate_mat

        if self.training:
            # Match flat AttributeMoE: aux losses use soft gates; traffic stats
            # keep Top-k path mass in gate_mat for growth scoring.
            self._last_gates = soft_gate_mat
            self._last_traffic_gates = gate_mat
            self._last_router_gates = router_soft_acc
            self._last_router_selected = router_selected_acc
            if self.sparse and router_selected_acc:
                # Depth-1: selected over leaves == root selected.
                # Deeper: approximate via soft-path > mean (legacy).
                if len(router_selected_acc) == 1 and router_selected_acc[0].shape[1] == gate_mat.shape[1]:
                    self._last_selected = router_selected_acc[0]
                else:
                    self._last_selected = (
                        soft_gate_mat > (soft_gate_mat.mean(dim=1, keepdim=True) + 1e-8)
                    ).float()
            else:
                self._last_selected = torch.ones_like(soft_gate_mat)
            self._last_feat = feat.detach()
        return out

    # ----------------------------------------------------------------- losses
    def load_balancing_loss(self) -> torch.Tensor:
        """Switch-style load loss matching flat AttributeMoE.

        Uses soft router gates P and (when sparse) hard selection freq f:
            loss = B * sum(P * f)
        Averaged over internal routers when the tree has grown.
        """
        device = next(self.parameters()).device
        soft_list = getattr(self, "_last_router_gates", None)
        sel_list = getattr(self, "_last_router_selected", None)
        if not soft_list:
            return torch.zeros((), device=device)
        losses = []
        for i, soft in enumerate(soft_list):
            b = float(soft.shape[1])
            P = soft.mean(dim=0)
            if self.sparse and sel_list is not None and i < len(sel_list):
                f = sel_list[i].mean(dim=0)
            else:
                f = P
            losses.append(b * (P * f).sum())
        return torch.stack(losses).mean()

    def router_entropy_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        if self._last_gates is None or self.num_experts <= 1:
            return torch.zeros((), device=device)
        p = self._last_gates.clamp_min(1e-8)
        # normalize soft path probs (should already sum ~1, but be safe)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return -(p * p.log()).sum(dim=-1).mean()

    # ---------------------------------------------------------- growth stats
    def record_leaf_stats(self, loss_value: float):
        """
        Accumulate per-leaf traffic + feature heterogeneity.

        Image L1 alone cannot rank leaves (same scalar for all). Instead we track:
          mass_e  = sum_i path_prob(e|x_i)          # how much of the scene this leaf owns
          feat_var_e = weighted variance of fv among anchors routed to e
        Call once per training step after forward.
        """
        if self._last_gates is None or not self.dynamic:
            return
        with torch.no_grad():
            # Prefer Top-k path mass for growth traffic (actual mix); fall back to soft.
            gates = getattr(self, "_last_traffic_gates", None)
            if gates is None:
                gates = self._last_gates
            gates = gates.detach()  # [N, L]
            feat = self._last_feat
            leaves = self.leaf_nodes()
            lv = float(loss_value)
            n_leaves = gates.shape[1]
            for i, leaf in enumerate(leaves[:n_leaves]):
                w = gates[:, i]  # [N]
                mass = float(w.sum().item())
                if mass < 1e-8:
                    continue
                lid = leaf.node_id
                self._leaf_mass_sum[lid] = self._leaf_mass_sum.get(lid, 0.0) + mass
                self._leaf_loss_mass[lid] = self._leaf_loss_mass.get(lid, 0.0) + mass * lv
                self._leaf_steps[lid] = self._leaf_steps.get(lid, 0) + 1

                if feat is not None and feat.shape[0] == gates.shape[0]:
                    ww = w.unsqueeze(1)  # [N, 1]
                    # weighted mean / variance of anchor features
                    mean = (ww * feat).sum(dim=0) / (mass + 1e-8)
                    var = (ww * (feat - mean).pow(2)).sum(dim=0) / (mass + 1e-8)
                    self._leaf_feat_m2[lid] = self._leaf_feat_m2.get(lid, 0.0) + float(var.mean().item())

    def reset_leaf_stats(self):
        self._leaf_mass_sum.clear()
        self._leaf_loss_mass.clear()
        self._leaf_feat_m2.clear()
        self._leaf_steps.clear()

    @staticmethod
    def _median(vals: List[float]) -> float:
        """Even-n median = mean of two middle values (critical for 2-leaf trees)."""
        if not vals:
            return 1e-8
        s = sorted(vals)
        n = len(s)
        if n % 2 == 1:
            return max(s[n // 2], 1e-8)
        return max(0.5 * (s[n // 2 - 1] + s[n // 2]), 1e-8)

    def scene_loss_hetero(self) -> float:
        """
        MoE-hardness proxy H = L_max / median(L).
        Homogeneous leaves (easy for MoE) → H ≈ 1; peaked hard leaf → H > 1.
        """
        details = self.leaf_growth_scores()
        if not details:
            return 1.0
        L_vals = [d["L"] for d in details.values()]
        return float(max(L_vals) / self._median(L_vals))

    def grow_budget(
        self,
        hetero_easy: float = 1.08,
        hetero_hard: float = 1.20,
    ) -> Tuple[int, float]:
        """
        Scene-level expert budget from leaf-loss heterogeneity (Dynamic v4).

          H < hetero_easy  → budget 0  (stay lean)
          H < hetero_hard  → budget 1
          else             → budget max_total_grows

        Returns (remaining_grows, H).
        """
        H = self.scene_loss_hetero()
        if H < float(hetero_easy):
            budget = 0
        elif H < float(hetero_hard):
            budget = 1
        else:
            budget = int(self.max_total_grows)
        remaining = max(0, budget - len(self._growth_history))
        return remaining, H

    def leaf_growth_scores(self) -> Dict[str, Dict[str, float]]:
        """
        Rank leaves by capacity–difficulty mismatch (IT / Dynamic v4):

          R_e = L_e / median_L                         # relative difficulty
          excess = max(0, R_e - 1)                     # only above-median hard
          S_e = mass_frac * (1 + excess)^2 * (1 + 0.75 * feat_var)

        Equal-sized experts ⇒ fixed capacity C ≈ median difficulty.
        """
        leaves = self.leaf_nodes()
        total_mass = sum(self._leaf_mass_sum.get(leaf.node_id, 0.0) for leaf in leaves)
        total_mass = max(total_mass, 1e-8)

        L: Dict[str, float] = {}
        mass_frac: Dict[str, float] = {}
        feat_var: Dict[str, float] = {}
        for leaf in leaves:
            lid = leaf.node_id
            mass = self._leaf_mass_sum.get(lid, 0.0)
            if mass < 1e-8:
                continue
            steps = max(1, self._leaf_steps.get(lid, 1))
            mass_frac[lid] = mass / total_mass
            L[lid] = self._leaf_loss_mass.get(lid, 0.0) / mass
            feat_var[lid] = self._leaf_feat_m2.get(lid, 0.0) / steps

        if not mass_frac:
            return {}
        med_L = self._median(list(L.values()))
        med_fv = self._median(list(feat_var.values()))
        out: Dict[str, Dict[str, float]] = {}
        for lid in mass_frac:
            fv = feat_var[lid]
            rel_L = L[lid] / med_L
            excess = max(0.0, rel_L - 1.0)
            score = mass_frac[lid] * ((1.0 + excess) ** 2) * (1.0 + 0.75 * fv)
            neff = 1.0 + fv / med_fv
            out[lid] = {
                "score": score,
                "mass_frac": mass_frac[lid],
                "feat_var": fv,
                "L": L[lid],
                "rel_L": rel_L,
                "excess": excess,
                "neff": neff,
                "med_L": med_L,
                "med_fv": med_fv,
            }
        return out

    def find_leaf(self, leaf_id: str) -> Optional[TreeNode]:
        for leaf in self.leaf_nodes():
            if leaf.node_id == leaf_id:
                return leaf
        return None

    def _find_parent(self, target_id: str, node: Optional[TreeNode] = None) -> Optional[Tuple[TreeNode, int]]:
        node = node or self.root
        if node.is_leaf:
            return None
        for i, child in enumerate(node.child_nodes):
            if child.node_id == target_id:
                return node, i
            hit = self._find_parent(target_id, child)
            if hit is not None:
                return hit
        return None

    @torch.no_grad()
    def grow_leaf(self, leaf_id: str) -> bool:
        """
        Replace leaf with an internal DualRouter + branch_factor children
        cloned from the parent expert (+ noise). Returns True if grew.
        """
        if not self.dynamic:
            return False
        leaf = self.find_leaf(leaf_id)
        if leaf is None or not leaf.is_leaf:
            return False
        if leaf.depth >= self.max_depth:
            return False
        if len(self._growth_history) >= self.max_total_grows:
            return False
        if self.num_experts + (self.grow_factor - 1) > self.max_leaves:
            return False

        parent_info = self._find_parent(leaf_id)
        if parent_info is None:
            return False
        parent_node, child_idx = parent_info

        device = next(leaf.expert.parameters()).device
        children = nn.ModuleList()
        ref_state = leaf.expert.state_dict()
        for i in range(self.grow_factor):
            cid = self._next_id(f"{leaf_id}_c{i}_")
            expert = self._make_expert().to(device)
            expert.load_state_dict(ref_state)
            if self.expert_noise_std > 0:
                for p in expert.parameters():
                    p.add_(self.expert_noise_std * torch.randn_like(p))
            children.append(TreeNode(cid, depth=leaf.depth + 1, is_leaf=True, expert=expert))

        new_internal = TreeNode(
            node_id=self._next_id(f"{leaf_id}_R_"),
            depth=leaf.depth,
            is_leaf=False,
            router=self._make_router(
                self.grow_factor, use_gnn_router=self.use_gnn_router
            ).to(device),
            child_nodes=children,
        )
        parent_node.child_nodes[child_idx] = new_internal
        self._growth_history.append(leaf_id)
        for d in (
            self._leaf_mass_sum,
            self._leaf_loss_mass,
            self._leaf_feat_m2,
            self._leaf_steps,
        ):
            d.pop(leaf_id, None)
        return True

    def select_grow_candidates(
        self,
        min_score_ratio: float = 1.20,
        min_neff: float = 1.2,
        force_grow: bool = False,
        min_mass_frac: float = 0.15,
        bypass_neff: bool = False,
        near_miss_ratio: float = 0.98,
        min_loss_ratio: float = 1.05,
        hetero_easy: float = 1.08,
        hetero_hard: float = 1.20,
        require_feat_var_median: bool = True,
    ) -> List[Tuple[float, str]]:
        """
        Rank growable leaves without mutating the tree (IT v4 / expert budget).

        Scene budget from H = L_max / median_L (easy → 0 grows, hard → more).
        Leaf gates:
          - mass_frac >= min_mass_frac
          - rel_L = L/med_L >= min_loss_ratio
          - feat_var >= median feat_var (ΔI proxy; blocks fake outdoor grows)
          - neff >= min_neff (unless bypass)
          - score >= median(score) * min_score_ratio * near_miss
        """
        if not self.dynamic:
            self._last_grow_debug = "skip: not dynamic"
            return []
        remaining, hetero = self.grow_budget(
            hetero_easy=hetero_easy, hetero_hard=hetero_hard
        )
        if remaining <= 0:
            self._last_grow_debug = (
                f"skip: budget=0 H={hetero:.3f} "
                f"(easy<{hetero_easy} or exhausted; "
                f"grows={len(self._growth_history)}, leaves={self.num_experts})"
            )
            return []
        if len(self._growth_history) >= self.max_total_grows:
            self._last_grow_debug = (
                f"skip: hit max_total_grows={self.max_total_grows} "
                f"(leaves={self.num_experts})"
            )
            return []
        details = self.leaf_growth_scores()
        growable = []
        for lid, d in details.items():
            leaf = self.find_leaf(lid)
            if leaf is None or leaf.depth >= self.max_depth:
                continue
            if self.num_experts + (self.grow_factor - 1) > self.max_leaves:
                continue
            growable.append((d["score"], d["mass_frac"], lid, d))

        if not growable:
            self._last_grow_debug = (
                f"skip: no growable leaves (leaves={self.num_experts}, "
                f"stats={len(details)}, depth_cap={self.max_depth}, "
                f"grows={len(self._growth_history)} rem={remaining}, H={hetero:.3f})"
            )
            return []

        growable.sort(reverse=True)
        scores_only = [g[0] for g in growable]
        med = self._median(scores_only)
        threshold = med * min_score_ratio
        effective_thr = threshold * float(near_miss_ratio)

        candidates: List[Tuple[float, str]] = []
        neff_blocked = 0
        loss_blocked = 0
        fv_blocked = 0
        for sc, mass_frac, lid, d in growable:
            if mass_frac < min_mass_frac:
                continue
            rel_L = float(d.get("rel_L", d["L"] / max(d.get("med_L", 1e-8), 1e-8)))
            if rel_L < float(min_loss_ratio):
                loss_blocked += 1
                continue
            # ΔI proxy: only split leaves with above-median feature diversity.
            if require_feat_var_median:
                med_fv = float(d.get("med_fv", 0.0))
                if float(d.get("feat_var", 0.0)) < med_fv:
                    fv_blocked += 1
                    continue
            neff = float(d.get("neff", 1.0))
            if (not bypass_neff) and neff < min_neff:
                neff_blocked += 1
                continue
            if sc >= effective_thr:
                candidates.append((sc, lid))

        if not candidates and force_grow:
            sc, mass_frac, lid, d = growable[0]
            rel_L = float(d.get("rel_L", 0.0))
            if mass_frac >= min_mass_frac * 0.5 and rel_L >= min_loss_ratio * 0.98:
                candidates = [(sc, lid)]
                self._last_grow_debug = (
                    f"force-cand {lid}: score={sc:.4f} mass={mass_frac:.3f} "
                    f"rel_L={rel_L:.3f} feat_var={d['feat_var']:.4f} "
                    f"L={d['L']:.4f} H={hetero:.3f} rem={remaining}"
                )
            else:
                self._last_grow_debug = (
                    f"skip force: top mass={growable[0][1]:.4f} "
                    f"rel_L={float(growable[0][3].get('rel_L', 0)):.3f} "
                    f"H={hetero:.3f}"
                )
                return []
        elif candidates:
            top = details[candidates[0][1]]
            self._last_grow_debug = (
                f"cand {candidates[0][1]}: score={candidates[0][0]:.4f} "
                f"mass={top['mass_frac']:.3f} rel_L={top.get('rel_L', 0):.3f} "
                f"excess={top.get('excess', 0):.3f} fv={top.get('feat_var', 0):.3f} "
                f"H={hetero:.3f} rem={remaining} "
                f"(thr={threshold:.4f} eff={effective_thr:.4f}, n_cand={len(candidates)})"
            )
        else:
            self._last_grow_debug = (
                f"skip: no leaf above thr={threshold:.4f} eff={effective_thr:.4f} "
                f"(top={growable[0][0]:.4f} mass={growable[0][1]:.3f}, "
                f"loss_blocked={loss_blocked}, fv_blocked={fv_blocked}, "
                f"neff_blocked={neff_blocked}, H={hetero:.3f} rem={remaining})"
            )
            return []
        return candidates

    def maybe_grow(
        self,
        min_score_ratio: float = 1.20,
        min_neff: float = 1.2,
        max_grows: int = 1,
        force_grow: bool = False,
        min_mass_frac: float = 0.15,
        bypass_neff: bool = False,
        near_miss_ratio: float = 0.98,
        min_loss_ratio: float = 1.05,
        hetero_easy: float = 1.08,
        hetero_hard: float = 1.20,
        require_feat_var_median: bool = True,
    ) -> List[str]:
        """
        Grow up to `max_grows` highest-scoring growable leaves (IT v4).
        """
        candidates = self.select_grow_candidates(
            min_score_ratio=min_score_ratio,
            min_neff=min_neff,
            force_grow=force_grow,
            min_mass_frac=min_mass_frac,
            bypass_neff=bypass_neff,
            near_miss_ratio=near_miss_ratio,
            min_loss_ratio=min_loss_ratio,
            hetero_easy=hetero_easy,
            hetero_hard=hetero_hard,
            require_feat_var_median=require_feat_var_median,
        )
        if not candidates:
            return []

        grown = []
        for _, lid in candidates[:max_grows]:
            if self.grow_leaf(lid):
                grown.append(lid)
        if grown:
            self.reset_leaf_stats()
            # Rewrite debug from cand → grow for logs.
            if self._last_grow_debug.startswith("cand "):
                self._last_grow_debug = "grow " + self._last_grow_debug[len("cand "):]
            elif self._last_grow_debug.startswith("force-cand "):
                self._last_grow_debug = "force-grow " + self._last_grow_debug[len("force-cand "):]
        return grown

    # ------------------------------------------------------- checkpoint helpers
    def export_topology(self) -> dict:
        def rec(node: TreeNode) -> dict:
            if node.is_leaf:
                return {"id": node.node_id, "depth": node.depth, "leaf": True}
            return {
                "id": node.node_id,
                "depth": node.depth,
                "leaf": False,
                "children": [rec(c) for c in node.child_nodes],
            }

        return {
            "dynamic": self.dynamic,
            "branch_factor": self.branch_factor,
            "grow_factor": self.grow_factor,
            "tree_depth": self.tree_depth,
            "max_depth": self.max_depth,
            "max_total_grows": self.max_total_grows,
            "node_counter": self._node_counter,
            "growth_history": list(self._growth_history),
            "root": rec(self.root),
        }

    def _build_from_topo_node(self, spec: dict) -> TreeNode:
        if spec["leaf"]:
            return TreeNode(
                node_id=spec["id"],
                depth=spec["depth"],
                is_leaf=True,
                expert=self._make_expert(),
            )
        children = nn.ModuleList(
            [self._build_from_topo_node(c) for c in spec["children"]]
        )
        # All internal routers (incl. root) use dual GNN routing when enabled.
        return TreeNode(
            node_id=spec["id"],
            depth=spec["depth"],
            is_leaf=False,
            router=self._make_router(
                num_children=len(children),
                use_gnn_router=self.use_gnn_router,
            ),
            child_nodes=children,
        )

    def load_topology(self, topo: dict):
        """Rebuild empty tree matching saved topology (call before load_state_dict)."""
        self.dynamic = bool(topo.get("dynamic", self.dynamic))
        self.branch_factor = int(topo.get("branch_factor", self.branch_factor))
        self.grow_factor = int(topo.get("grow_factor", self.grow_factor))
        self.tree_depth = int(topo.get("tree_depth", self.tree_depth))
        self.max_depth = int(topo.get("max_depth", self.max_depth))
        self.max_total_grows = int(topo.get("max_total_grows", self.max_total_grows))
        self._node_counter = int(topo.get("node_counter", 0))
        self._growth_history = list(topo.get("growth_history", []))
        self.root = self._build_from_topo_node(topo["root"])
