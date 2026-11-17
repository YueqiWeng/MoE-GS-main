#
# Helpers for MoE warm-start from Scaffold-GS baseline / flat-MoE checkpoints.
#

import os
import torch
from torch import nn

from scene.tree_moe import AttributeTreeMoE, DualRouter


def _expert0_state_dict(moe_state: dict) -> dict:
    prefix = "experts.0."
    return {k[len(prefix):]: v for k, v in moe_state.items() if k.startswith(prefix)}


def _collect_expert_state_dicts(state_dict: dict):
    """Extract experts.i.* blocks from a flat AttributeMoE state_dict."""
    expert_sds = []
    i = 0
    while True:
        prefix = f"experts.{i}."
        sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if not sd:
            break
        expert_sds.append(sd)
        i += 1
    return expert_sds


def _iter_leaf_experts(moe):
    if isinstance(moe, AttributeTreeMoE):
        return [leaf.expert for leaf in moe.leaf_nodes()]
    return list(moe.experts)


def _zero_routers(moe):
    if isinstance(moe, AttributeTreeMoE):
        for module in moe.modules():
            if isinstance(module, DualRouter):
                nn.init.zeros_(module.router.weight)
                nn.init.zeros_(module.router.bias)
                if module.gnn_router is not None:
                    nn.init.zeros_(module.gnn_router.weight)
                    nn.init.zeros_(module.gnn_router.bias)
        return
    if hasattr(moe, "router"):
        nn.init.zeros_(moe.router.weight)
        nn.init.zeros_(moe.router.bias)
    if getattr(moe, "gnn_router", None) is not None:
        nn.init.zeros_(moe.gnn_router.weight)
        nn.init.zeros_(moe.gnn_router.bias)


def _spread_expert_indices(n_flat: int, n_leaves: int):
    """
    Map flat E experts onto L tree leaves with spread indices.

    E=8, L=2 → [0, 4]; E=8, L=4 → [0, 2, 4, 6]; E==L → range(E).
    """
    if n_leaves <= 0:
        return []
    if n_flat <= 0:
        return list(range(n_leaves))
    if n_leaves == 1:
        return [0]
    if n_flat == n_leaves:
        return list(range(n_flat))
    # Contiguous-group starts: E=8,L=2 → [0,4]; E=8,L=4 → [0,2,4,6].
    return [min(n_flat - 1, int(i * n_flat / n_leaves)) for i in range(n_leaves)]


def _copy_flat_router_into_flat(moe, state_dict: dict) -> bool:
    """
    Copy flat MoE base router 1:1 into AttributeMoE; zero-init gnn_router if present.
    """
    if isinstance(moe, AttributeTreeMoE) or not hasattr(moe, "router"):
        return False
    n_exp = int(getattr(moe, "num_experts", 0))
    for prefix in ("router.", "base_router."):
        w_key, b_key = f"{prefix}weight", f"{prefix}bias"
        if w_key not in state_dict or b_key not in state_dict:
            continue
        flat_w, flat_b = state_dict[w_key], state_dict[b_key]
        if flat_w.ndim != 2 or flat_w.shape[0] != n_exp:
            continue
        with torch.no_grad():
            moe.router.weight.copy_(flat_w)
            moe.router.bias.copy_(flat_b)
            if getattr(moe, "gnn_router", None) is not None:
                nn.init.zeros_(moe.gnn_router.weight)
                nn.init.zeros_(moe.gnn_router.bias)
            print(
                f"[MoE] Copied flat router → flat MoE "
                f"(E={n_exp}, prefix={prefix[:-1]}; gnn_router zero-init)"
            )
            return True
    return False


def _copy_flat_router_into_tree_root(moe: AttributeTreeMoE, state_dict: dict) -> bool:
    """
    Copy / pool flat MoE router into the tree root DualRouter.

    - If E == L: copy router columns 1:1.
    - If E > L and E % L == 0: sum contiguous groups of size E/L (preserve prior).
    - Else: fail (caller zeros routers).
    """
    if not isinstance(moe, AttributeTreeMoE):
        return False
    root = moe.root
    if root is None or root.is_leaf or root.router is None:
        return False
    n_leaves = len(root.child_nodes)
    n_flat = len(_collect_expert_state_dicts(state_dict))
    if n_flat <= 0 or n_leaves <= 0:
        return False

    for prefix in ("router.", "base_router."):
        w_key, b_key = f"{prefix}weight", f"{prefix}bias"
        if w_key not in state_dict or b_key not in state_dict:
            continue
        flat_w = state_dict[w_key]
        flat_b = state_dict[b_key]
        if flat_w.ndim != 2 or flat_w.shape[0] != n_flat:
            continue
        with torch.no_grad():
            if n_flat == n_leaves:
                root.router.router.weight.copy_(flat_w)
                root.router.router.bias.copy_(flat_b)
                mode = "1:1"
            elif n_flat % n_leaves == 0:
                group = n_flat // n_leaves
                # Pool contiguous expert logits: [E, D] → [L, D]
                w = flat_w.reshape(n_leaves, group, -1).sum(dim=1)
                b = flat_b.reshape(n_leaves, group).sum(dim=1)
                root.router.router.weight.copy_(w)
                root.router.router.bias.copy_(b)
                mode = f"pool/{group}"
            else:
                continue
            if root.router.gnn_router is not None:
                nn.init.zeros_(root.router.gnn_router.weight)
                nn.init.zeros_(root.router.gnn_router.bias)
            print(
                f"[MoE] Copied flat router → tree root "
                f"({n_flat}→{n_leaves} leaves, {mode}, prefix={prefix[:-1]})"
            )
            return True
    return False


def load_sequential_into_moe(moe, state_dict: dict):
    """
    Copy baseline Sequential or flat-MoE expert weights into leaf experts.

    Flat MoE with E experts → assign onto L tree leaves with spread indices
    (E=8,L=2 → experts 0 and 4). Pool flat router into root when possible;
    otherwise zero-init routers.
    """
    leaves = _iter_leaf_experts(moe)
    expert_sds = _collect_expert_state_dicts(state_dict)

    if expert_sds:
        idxs = _spread_expert_indices(len(expert_sds), len(leaves))
        for j, expert in enumerate(leaves):
            expert.load_state_dict(expert_sds[idxs[j]])
        if idxs != list(range(len(leaves))) or len(expert_sds) != len(leaves):
            print(
                f"[MoE] Warm-start experts spread {len(expert_sds)}→{len(leaves)} "
                f"via indices {idxs}"
            )
        if isinstance(moe, AttributeTreeMoE):
            if not _copy_flat_router_into_tree_root(moe, state_dict):
                _zero_routers(moe)
        elif not _copy_flat_router_into_flat(moe, state_dict):
            _zero_routers(moe)
        return

    if len(leaves) <= 1:
        leaves[0].load_state_dict(state_dict)
        return

    for expert in leaves:
        expert.load_state_dict(state_dict)
    _zero_routers(moe)


def _load_jit_sequential(path: str) -> dict:
    return torch.jit.load(path, map_location="cuda").state_dict()


def load_baseline_checkpoints_into_model(gaussians, checkpoint_path: str):
    """
    Warm-start MoE / TreeMoE experts from Scaffold-GS or flat-MoE weights.

    Accepts:
      - checkpoints.pth (unite state_dict)
      - directory containing checkpoints.pth or opacity_mlp.pt / cov_mlp.pt / color_mlp.pt
    """
    if os.path.isdir(checkpoint_path):
        base = checkpoint_path
        unite = os.path.join(base, "checkpoints.pth")
        if os.path.isfile(unite):
            ckpt = torch.load(unite, map_location="cuda")
        else:
            ckpt = {
                "opacity_mlp": _load_jit_sequential(os.path.join(base, "opacity_mlp.pt")),
                "cov_mlp": _load_jit_sequential(os.path.join(base, "cov_mlp.pt")),
                "color_mlp": _load_jit_sequential(os.path.join(base, "color_mlp.pt")),
            }
    else:
        ckpt = torch.load(checkpoint_path, map_location="cuda")

    for key, moe in (
        ("opacity_mlp", gaussians.mlp_opacity),
        ("cov_mlp", gaussians.mlp_cov),
        ("color_mlp", gaussians.mlp_color),
    ):
        if key not in ckpt:
            raise KeyError(f"Missing {key} in {checkpoint_path}")
        load_sequential_into_moe(moe, ckpt[key])
    print(f"[MoE] Warm-started experts from: {checkpoint_path}")
