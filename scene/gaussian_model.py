#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
from scene.moe import AttributeMoE
from scene.tree_moe import AttributeTreeMoE
from scene.gnn import AnchorGNN

    
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 num_experts: int = 1,
                 moe_top_k: int = 2,
                 use_gnn: bool = False,
                 gnn_hidden_dim: int = 32,
                 gnn_layers: int = 2,
                 gnn_k: int = 16,
                 moe_arch: str = "flat",
                 moe_branch_factor: int = 4,
                 moe_tree_depth: int = 2,
                 moe_max_leaves: int = 10,
                 moe_grow_factor: int = 0,
                 moe_max_total_grows: int = 64,
                 gnn_gamma_init: float = 0.5,
                 mlp_depth: int = 2,
                 ):

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        self.moe_arch = str(moe_arch).lower().strip()
        if self.moe_arch not in ("flat", "dynamic_tree"):
            raise ValueError(f"Unknown moe_arch={moe_arch!r}; expected flat|dynamic_tree")
        self.moe_branch_factor = max(2, int(moe_branch_factor))
        self.moe_tree_depth = max(1, int(moe_tree_depth))
        self.moe_max_leaves = max(self.moe_branch_factor, int(moe_max_leaves))
        self.moe_grow_factor = int(moe_grow_factor) if int(moe_grow_factor) >= 2 else self.moe_branch_factor
        self.moe_max_total_grows = max(0, int(moe_max_total_grows))
        self.gnn_gamma_init = float(gnn_gamma_init)
        self.mlp_depth = max(2, int(mlp_depth))
        self.use_tree_moe = self.moe_arch == "dynamic_tree"
        self._tree_dense_until = 0

        self.num_experts = max(1, int(num_experts))
        self.moe_top_k = min(
            max(1, int(moe_top_k)),
            max(self.moe_branch_factor, self.moe_grow_factor) if self.use_tree_moe else self.num_experts,
        )

        # GNN allowed for flat MoE+GNN (moe_gnn) and legacy dynamic_tree checkpoints.
        self.use_gnn = bool(use_gnn)
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_layers = gnn_layers
        self.gnn_k = max(1, int(gnn_k))
        self.anchor_gnn = None
        if self.use_gnn:
            self.anchor_gnn = AnchorGNN(
                feat_dim=feat_dim,
                hidden_dim=gnn_hidden_dim,
                num_layers=gnn_layers,
                knn_k=self.gnn_k,
            ).cuda()

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        
        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.color_dist_dim = 1 if self.add_color_dist else 0

        if self.use_tree_moe:
            tree_kw = dict(
                branch_factor=self.moe_branch_factor,
                tree_depth=self.moe_tree_depth,
                top_k=self.moe_top_k,
                use_gnn_router=self.use_gnn,
                gnn_gamma_init=self.gnn_gamma_init,
                dynamic=True,
                max_leaves=self.moe_max_leaves,
                grow_factor=self.moe_grow_factor,
                max_total_grows=self.moe_max_total_grows,
            )
            self.mlp_opacity = AttributeTreeMoE(
                router_dim=feat_dim,
                expert_in_dim=feat_dim + 3 + self.opacity_dist_dim,
                hidden_dim=feat_dim,
                out_dim=n_offsets,
                out_activation=nn.Tanh(),
                **tree_kw,
            ).cuda()

            self.mlp_cov = AttributeTreeMoE(
                router_dim=feat_dim,
                expert_in_dim=feat_dim + 3 + self.cov_dist_dim,
                hidden_dim=feat_dim,
                out_dim=7 * self.n_offsets,
                out_activation=None,
                **tree_kw,
            ).cuda()

            self.mlp_color = AttributeTreeMoE(
                router_dim=feat_dim,
                expert_in_dim=feat_dim + 3 + self.color_dist_dim + self.appearance_dim,
                hidden_dim=feat_dim,
                out_dim=3 * self.n_offsets,
                out_activation=nn.Sigmoid(),
                **tree_kw,
            ).cuda()
            self.num_experts = self.mlp_opacity.num_experts
        else:
            moe_kw = dict(
                num_experts=self.num_experts,
                top_k=self.moe_top_k,
                use_gnn_router=self.use_gnn,
                gnn_gamma_init=self.gnn_gamma_init,
                mlp_depth=self.mlp_depth,
            )
            self.mlp_opacity = AttributeMoE(
                router_dim=feat_dim,
                expert_in_dim=feat_dim + 3 + self.opacity_dist_dim,
                hidden_dim=feat_dim,
                out_dim=n_offsets,
                out_activation=nn.Tanh(),
                **moe_kw,
            ).cuda()

            self.mlp_cov = AttributeMoE(
                router_dim=feat_dim,
                expert_in_dim=feat_dim + 3 + self.cov_dist_dim,
                hidden_dim=feat_dim,
                out_dim=7 * self.n_offsets,
                out_activation=None,
                **moe_kw,
            ).cuda()

            self.mlp_color = AttributeMoE(
                router_dim=feat_dim,
                expert_in_dim=feat_dim + 3 + self.color_dist_dim + self.appearance_dim,
                hidden_dim=feat_dim,
                out_dim=3 * self.n_offsets,
                out_activation=nn.Sigmoid(),
                **moe_kw,
            ).cuda()

        self.use_moe = (
            (self.mlp_opacity.num_experts > 1)
            if hasattr(self.mlp_opacity, "num_experts")
            else self.num_experts > 1
        )

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.anchor_gnn is not None:
            self.anchor_gnn.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.anchor_gnn is not None:
            self.anchor_gnn.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def spatial_context(self, anchor_pos, anchor_feat):
        """GNN spatial context h_gnn for dual-router MoE (None if GNN disabled)."""
        if self.anchor_gnn is None:
            return None
        return self.anchor_gnn(anchor_pos, anchor_feat)

    def moe_routing_snapshot(self):
        """Per-head gate stats for routing_analysis.json (works for 1 expert too)."""
        heads = {}
        for name, moe in (
            ("opacity", self.mlp_opacity),
            ("cov", self.mlp_cov),
            ("color", self.mlp_color),
        ):
            n = int(getattr(moe, "num_experts", 1))
            rec = {"n_experts": n}
            gates = getattr(moe, "_last_gates", None)
            selected = getattr(moe, "_last_selected", None)
            if gates is not None and gates.numel() > 0:
                util = gates.detach().float().mean(dim=0).cpu()
                rec["util"] = [float(x) for x in util]
                p = util.clamp_min(1e-8)
                p = p / p.sum()
                h = float(-(p * p.log()).sum())
                rec["util_entropy"] = h
                rec["util_entropy_norm"] = h / max(float(np.log(max(n, 1))), 1e-8)
                mean_u = float(util.mean().clamp_min(1e-8))
                rec["cv"] = float(util.std(unbiased=False) / mean_u) if n > 1 else 0.0
                rec["max_util"] = float(util.max())
                rec["min_util"] = float(util.min())
                g = gates.detach().float().clamp_min(1e-8)
                rec["sample_entropy"] = float(-(g * g.log()).sum(dim=-1).mean())
                if selected is not None:
                    rec["select_frac"] = [
                        float(x) for x in selected.detach().float().mean(dim=0).cpu()
                    ]
            if hasattr(moe, "describe"):
                rec["tree"] = moe.describe()
            hist = getattr(moe, "_growth_history", None)
            if hist is not None:
                rec["growth_history"] = list(hist)
            heads[name] = rec
        return {
            "arch": getattr(self, "moe_arch", "flat"),
            "use_moe": bool(getattr(self, "use_moe", False)),
            "use_gnn": bool(getattr(self, "use_gnn", False)),
            "sparse": bool(getattr(self.mlp_opacity, "sparse", False)),
            "heads": heads,
        }

    def moe_expert_utilization(self):
        """Mean gate per expert/leaf, averaged over opacity/cov/color MoEs."""
        if not self.use_moe:
            return None
        stats = []
        for moe in (self.mlp_opacity, self.mlp_cov, self.mlp_color):
            if moe._last_gates is not None:
                stats.append(moe._last_gates.mean(dim=0))
        if not stats:
            return None
        # Tree MoEs may have different leaf counts after asymmetric growth; use min length
        min_e = min(s.shape[0] for s in stats)
        stats = [s[:min_e] for s in stats]
        return torch.stack(stats, dim=0).mean(dim=0)

    def set_moe_sparse(self, sparse: bool):
        self.mlp_opacity.set_sparse(sparse)
        self.mlp_cov.set_sparse(sparse)
        self.mlp_color.set_sparse(sparse)

    def moe_load_balancing_loss(self):
        # Average over the 3 attribute MoEs so λ_load matches the paper scale
        # (one load term), not 3× that magnitude.
        if not self.use_moe:
            return torch.zeros((), device="cuda")
        return (
            self.mlp_opacity.load_balancing_loss()
            + self.mlp_cov.load_balancing_loss()
            + self.mlp_color.load_balancing_loss()
        ) / 3.0

    def moe_router_entropy_loss(self):
        """Mean per-sample gate entropy across opacity/cov/color MoEs."""
        if not self.use_moe:
            return torch.zeros((), device="cuda")
        return (
            self.mlp_opacity.router_entropy_loss()
            + self.mlp_cov.router_entropy_loss()
            + self.mlp_color.router_entropy_loss()
        ) / 3.0

    def moe_record_leaf_stats(self, loss_value: float):
        """Attribute scalar loss to dynamic-tree leaves (no-op for flat/static)."""
        if not self.use_tree_moe or self.moe_arch != "dynamic_tree":
            return
        for moe in (self.mlp_opacity, self.mlp_cov, self.mlp_color):
            if isinstance(moe, AttributeTreeMoE):
                moe.record_leaf_stats(loss_value)

    def refresh_mlp_optimizer_params(self, training_args):
        """
        After dynamic tree growth, keep Adam state for surviving params and
        register newly created router/expert parameters.
        """
        if self.optimizer is None:
            return
        name_to_module = {
            "mlp_opacity": self.mlp_opacity,
            "mlp_cov": self.mlp_cov,
            "mlp_color": self.mlp_color,
        }
        # Snapshot old Adam state keyed by parameter id
        old_state_by_id = {}
        for p, st in self.optimizer.state.items():
            old_state_by_id[id(p)] = st

        new_groups = []
        for group in self.optimizer.param_groups:
            name = group.get("name", "")
            if name in name_to_module:
                module = name_to_module[name]
                new_groups.append(
                    {
                        "params": list(module.parameters()),
                        "lr": group["lr"],
                        "betas": group.get("betas", (0.9, 0.999)),
                        "eps": group.get("eps", 1e-15),
                        "weight_decay": group.get("weight_decay", 0.0),
                        "amsgrad": group.get("amsgrad", False),
                        "name": name,
                    }
                )
            else:
                new_groups.append(group)

        self.optimizer = torch.optim.Adam(new_groups, lr=0.0, eps=1e-15)
        # Restore momentum for params that still exist
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                st = old_state_by_id.get(id(p))
                if st is not None:
                    self.optimizer.state[p] = st

        self.num_experts = self.mlp_opacity.num_experts
        self.use_moe = self.mlp_opacity.num_experts > 1

    def maybe_grow_tree_moe(self, training_args, logger=None):
        """
        Grow high-need leaves.

        Prefer color when it lags (appearance is usually hardest). Looser
        loss/mass gates for lagging heads so growth is not stuck on opacity/cov.
        """
        if not self.use_tree_moe or self.moe_arch != "dynamic_tree":
            return []
        heads = []
        for name, moe in (
            ("opacity", self.mlp_opacity),
            ("cov", self.mlp_cov),
            ("color", self.mlp_color),
        ):
            if isinstance(moe, AttributeTreeMoE):
                heads.append((name, moe))
        if not heads:
            return []

        leaf_counts = [moe.num_experts for _, moe in heads]
        max_leaves = max(leaf_counts)
        min_leaves = min(leaf_counts)
        base_ratio = float(getattr(training_args, "tree_grow_min_score_ratio", 1.08))
        base_mass = float(getattr(training_args, "tree_grow_min_mass_frac", 0.12))
        base_neff = float(getattr(training_args, "tree_grow_min_neff", 1.1))
        base_loss = float(getattr(training_args, "tree_grow_min_loss_ratio", 0.95))
        hetero_easy = float(getattr(training_args, "tree_grow_hetero_easy", 1.0))
        hetero_hard = float(getattr(training_args, "tree_grow_hetero_hard", 1.0))
        require_fv = bool(getattr(training_args, "tree_grow_require_feat_var_median", False))
        max_grows = int(getattr(training_args, "tree_grow_max_per_event", 1))
        max_heads = int(getattr(training_args, "tree_grow_max_heads_per_event", 2))
        force_grow = bool(getattr(training_args, "tree_grow_force", False))
        # Color first, then opacity, then cov.
        head_priority = {"color": 3, "opacity": 2, "cov": 1}

        proposals = []
        debug_msgs = []
        for name, moe in heads:
            lagging = moe.num_experts <= (max_leaves - 1)
            # Extra boost when this head is the smallest (often color never grew).
            starved = moe.num_experts == min_leaves and max_leaves > min_leaves
            color_boost = name == "color" and (lagging or starved)
            min_score_ratio = base_ratio * (0.88 if color_boost else (0.92 if lagging else 1.0))
            min_mass_frac = base_mass * (0.65 if color_boost else (0.75 if lagging else 1.0))
            min_loss_ratio = base_loss * (0.90 if color_boost else (0.95 if lagging else 1.0))
            # Force at least one candidate if color is starved and others already grew.
            head_force = force_grow or (color_boost and starved and max_leaves >= min_leaves + 1)
            cands = moe.select_grow_candidates(
                min_score_ratio=min_score_ratio,
                min_neff=base_neff,
                force_grow=head_force,
                min_mass_frac=min_mass_frac,
                bypass_neff=lagging or color_boost,
                min_loss_ratio=min_loss_ratio,
                hetero_easy=hetero_easy,
                hetero_hard=hetero_hard,
                require_feat_var_median=require_fv,
            )
            dbg = getattr(moe, "_last_grow_debug", "")
            tag = "lag" if lagging else "ok"
            debug_msgs.append(
                f"{name}[{tag} r={min_score_ratio:.3f} m={min_mass_frac:.3f}]: {dbg}"
            )
            if cands:
                sc, lid = cands[0]
                proposals.append(
                    (
                        1 if lagging else 0,
                        head_priority.get(name, 0),
                        sc,
                        name,
                        moe,
                        lid,
                        min_score_ratio,
                        min_mass_frac,
                        min_loss_ratio,
                    )
                )

        if debug_msgs:
            msg = "[TreeMoE] " + " | ".join(debug_msgs)
            print(msg)
            if logger is not None:
                logger.info(msg)

        if not proposals:
            return []

        proposals.sort(reverse=True)
        chosen = proposals[: max(1, max_heads)]
        all_grown = []
        for _lag, _prio, sc, name, moe, lid, _, _, min_loss_ratio in chosen:
            if moe.grow_leaf(lid):
                moe.reset_leaf_stats()
                all_grown.append(f"{name}:{lid}")
            for _extra in range(max_grows - 1):
                more = moe.select_grow_candidates(
                    min_score_ratio=base_ratio,
                    min_neff=base_neff,
                    force_grow=False,
                    min_mass_frac=base_mass,
                    bypass_neff=False,
                    min_loss_ratio=min_loss_ratio,
                    hetero_easy=hetero_easy,
                    hetero_hard=hetero_hard,
                    require_feat_var_median=require_fv,
                )
                if not more:
                    break
                if moe.grow_leaf(more[0][1]):
                    moe.reset_leaf_stats()
                    all_grown.append(f"{name}:{more[0][1]}")
        if all_grown:
            self.refresh_mlp_optimizer_params(training_args)
            msg = (
                f"[TreeMoE] grew {all_grown} (heads_budget={max_heads}); "
                f"leaves opacity/cov/color="
                f"{self.mlp_opacity.num_experts}/"
                f"{self.mlp_cov.num_experts}/"
                f"{self.mlp_color.num_experts}"
            )
            print(msg)
            if logger is not None:
                logger.info(msg)
        return all_grown

    def tree_moe_summary(self) -> str:
        if not self.use_tree_moe:
            return f"flat MoE experts={self.num_experts}"
        return (
            f"{self.moe_arch} "
            f"opacity[{self.mlp_opacity.describe()}] "
            f"cov[{self.mlp_cov.describe()}] "
            f"color[{self.mlp_color.describe()}]"
        )
    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._anchor, 
        self._offset,
        self._local,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
        return data

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        
        
        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]

        if self.anchor_gnn is not None:
            l.append({'params': self.anchor_gnn.parameters(), 'lr': training_args.gnn_lr_init, "name": "anchor_gnn"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)
        if self.anchor_gnn is not None:
            self.gnn_scheduler_args = get_expon_lr_func(lr_init=training_args.gnn_lr_init,
                                                        lr_final=training_args.gnn_lr_final,
                                                        lr_delay_mult=training_args.gnn_lr_delay_mult,
                                                        max_steps=training_args.gnn_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.anchor_gnn is not None and param_group["name"] == "anchor_gnn":
                lr = self.gnn_scheduler_args(iteration)
                param_group['lr'] = lr
            
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name'] or \
                'gnn' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

        

        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name'] or \
                'gnn' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    
    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                if self.anchor_gnn is not None:
                    self.anchor_gnn.invalidate_cache()
                


    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(path)
        # MoE has two forward inputs; always use state_dict unite format.
        if self.use_moe:
            mode = 'unite'

        if mode == 'split' and not self.use_moe:
            # Legacy JIT path only for single-expert modules (trace one expert MLP)
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(
                self.mlp_opacity.experts[0],
                (torch.rand(1, self.feat_dim + 3 + self.opacity_dist_dim).cuda()),
            )
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(
                self.mlp_cov.experts[0],
                (torch.rand(1, self.feat_dim + 3 + self.cov_dist_dim).cuda()),
            )
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(
                self.mlp_color.experts[0],
                (torch.rand(1, self.feat_dim + 3 + self.color_dist_dim + self.appearance_dim).cuda()),
            )
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            ckpt = {
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                'num_experts': self.num_experts,
                'moe_top_k': self.moe_top_k,
                'moe_arch': self.moe_arch,
                'moe_branch_factor': self.moe_branch_factor,
                'moe_tree_depth': self.moe_tree_depth,
            }
            if self.use_tree_moe:
                ckpt['opacity_tree_topo'] = self.mlp_opacity.export_topology()
                ckpt['cov_tree_topo'] = self.mlp_cov.export_topology()
                ckpt['color_tree_topo'] = self.mlp_color.export_topology()
            if self.anchor_gnn is not None:
                ckpt['anchor_gnn'] = self.anchor_gnn.state_dict()
            if self.use_feat_bank:
                ckpt['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
            if self.appearance_dim > 0:
                ckpt['appearance'] = self.embedding_appearance.state_dict()
            torch.save(ckpt, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        unite_path = os.path.join(path, 'checkpoints.pth')
        # Prefer unite checkpoints (required for MoE)
        if self.use_moe or os.path.isfile(unite_path):
            mode = 'unite'

        if mode == 'split':
            # Load legacy JIT into the single expert slot
            first_op = self.mlp_opacity.experts[0] if hasattr(self.mlp_opacity, 'experts') else None
            self.mlp_opacity.experts[0] = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov.experts[0] = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color.experts[0] = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
            _ = first_op
        elif mode == 'unite':
            checkpoint = torch.load(unite_path, map_location='cuda')
            if self.use_tree_moe:
                if 'opacity_tree_topo' in checkpoint:
                    self.mlp_opacity.load_topology(checkpoint['opacity_tree_topo'])
                    self.mlp_opacity.cuda()
                if 'cov_tree_topo' in checkpoint:
                    self.mlp_cov.load_topology(checkpoint['cov_tree_topo'])
                    self.mlp_cov.cuda()
                if 'color_tree_topo' in checkpoint:
                    self.mlp_color.load_topology(checkpoint['color_tree_topo'])
                    self.mlp_color.cuda()
            # strict=False tolerates root GNN key drift across recipe versions
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'], strict=False)
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'], strict=False)
            self.mlp_color.load_state_dict(checkpoint['color_mlp'], strict=False)
            self.num_experts = self.mlp_opacity.num_experts
            self.use_moe = self.mlp_opacity.num_experts > 1
            if self.anchor_gnn is not None and 'anchor_gnn' in checkpoint:
                self.anchor_gnn.load_state_dict(checkpoint['anchor_gnn'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError
