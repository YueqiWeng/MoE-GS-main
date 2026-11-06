#!/usr/bin/env bash
# =============================================================================
# Methods:
#
#   3dgs       Official 3D Gaussian Splatting (Kerbl et al., TOG 2023).
#              External code: third_party/gaussian-splatting (cloned on first run).
#   baseline   Scaffold-GS (shared MLP)
#   moe        Flat MoE (8 experts, Top-2). No GNN.
#   moe_gnn    MoE + GNN: warm-start from finished `moe`, keep MoE points,
#              add dual GNN router, train another 30k.
#   moe_tree   Growing Tree MoE (no GNN): warm-start from `moe`, keep MoE
#              points, start 8 leaves (1:1 MoE), grow hard leaves (color-first).
#   tree_gnn   Growing Tree MoE + GNN: same grow recipe as moe_tree, with GNN.
#
# Usage:
#   bash train_method.sh --method 3dgs     -d mipnerf360/kitchen -g 0
#   bash train_method.sh --method baseline -d mipnerf360/kitchen -g 0
#   bash train_method.sh --method moe      -d mipnerf360/kitchen -g 0
#   bash train_method.sh --method moe_gnn  -d mipnerf360/kitchen -g 0
#   bash train_method.sh --method moe_tree -d mipnerf360/kitchen -g 0
#   bash train_method.sh --method tree_gnn -d mipnerf360/kitchen -g 0
#
# Ablation knobs (mostly for moe / moe_tree):
#   --moe_warmup 0              # w/o curriculum (Top-k from start)
#   --lambda_load 0             # w/o L_load
#   --lambda_opacity 0          # w/o L_opacity
#   --lambda_router_entropy 0   # w/o entropy regularizer
#   --no_warm_start             # moe_tree without flat-MoE init
#   --num_experts N --moe_top_k K
# =============================================================================

set -euo pipefail

rand() {
    local min=$1 max=$(($2 - $1 + 1)) num
    num=$(date +%s%N)
    echo $((num % max + min))
}

method=""
port=$(rand 10000 30000)
lod=0
iterations=30_000
appearance_dim=0
voxel_size=0.001
update_init_factor=16
ratio=1
gpu=0
data=""
logdir=""
update_until=""

num_experts=8
moe_top_k=2
moe_warmup=3000
lambda_load=0.005
lambda_router_entropy=0.002
lambda_opacity=0.01

gnn_layers=2
gnn_hidden_dim=32
gnn_k=16
gnn_gamma_init=0.5

# Tree MoE + GNN (growing) — v2 recipe:
# start 8 leaves (= flat MoE capacity, 1:1 warm-start), then grow hard leaves.
moe_branch_factor=8
moe_grow_factor=2
moe_tree_depth=3
moe_max_leaves=14
moe_max_total_grows=4
tree_grow_start=6000
tree_grow_interval=4000
tree_grow_stop=20000
tree_grow_max_per_event=1
tree_grow_max_heads_per_event=2
tree_grow_min_score_ratio=1.08
tree_grow_min_neff=1.1
tree_grow_min_mass_frac=0.12
tree_grow_min_loss_ratio=0.95
tree_grow_dense_iters=1500

start_baseline_mlp=""
start_gaussian_ply=""
warm_start_mode="auto"

add_opacity_dist=""
add_cov_dist=""
add_color_dist=""

usage() {
    sed -n '2,24p' "$0"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --method) method="$2"; shift ;;
        -d|--data) data="$2"; shift ;;
        -g|--gpu) gpu="$2"; shift ;;
        -l|--logdir) logdir="$2"; shift ;;
        --lod) lod="$2"; shift ;;
        --iterations) iterations="$2"; shift ;;
        --appearance_dim) appearance_dim="$2"; shift ;;
        --voxel_size) voxel_size="$2"; shift ;;
        --update_init_factor) update_init_factor="$2"; shift ;;
        --ratio) ratio="$2"; shift ;;
        --update_until) update_until="$2"; shift ;;
        --start_baseline_mlp)
            start_baseline_mlp="$2"
            if [[ -z "$2" ]]; then warm_start_mode="off"; else warm_start_mode="path"; fi
            shift ;;
        --start_gaussian_ply)
            start_gaussian_ply="$2"
            warm_start_mode="path"
            shift ;;
        --no_warm_start) warm_start_mode="off"; start_baseline_mlp=""; start_gaussian_ply="" ;;
        --add_opacity_dist) add_opacity_dist="--add_opacity_dist" ;;
        --add_cov_dist) add_cov_dist="--add_cov_dist" ;;
        --add_color_dist) add_color_dist="--add_color_dist" ;;
        --num_experts) num_experts="$2"; shift ;;
        --moe_top_k) moe_top_k="$2"; shift ;;
        --moe_warmup) moe_warmup="$2"; shift ;;
        --lambda_load) lambda_load="$2"; shift ;;
        --lambda_opacity) lambda_opacity="$2"; shift ;;
        --lambda_router_entropy) lambda_router_entropy="$2"; shift ;;
        --moe_max_total_grows) moe_max_total_grows="$2"; shift ;;
        --tree_grow_start) tree_grow_start="$2"; shift ;;
        --tree_grow_interval) tree_grow_interval="$2"; shift ;;
        --tree_grow_stop) tree_grow_stop="$2"; shift ;;
        --tree_grow_dense_iters) tree_grow_dense_iters="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

case "${method}" in
    3dgs|3d-gs|3d_gs|gaussian_splatting|gaussian-splatting) method=3dgs ;;
    baseline|scaffold|scaffold-gs|scaffold_gs) method=baseline ;;
    moe|flat_moe|flat-moe) method=moe ;;
    moe_gnn|moe+gnn|moe-gnn|moe_then_gnn|moe_gnn_frozen|gnn_frozen) method=moe_gnn ;;
    moe_tree|tree|tree_moe|growing_tree) method=moe_tree ;;
    tree_gnn|dynamic_gnn|dynamic_tree_moe_gnn|tree+gnn) method=tree_gnn ;;
    "")
        echo "[error] --method is required"
        usage ;;
    *)
        echo "[error] unknown method: ${method}"
        echo "        expected: 3dgs | baseline | moe | moe_gnn | moe_tree | tree_gnn"
        exit 1 ;;
esac

if [[ -z "${data}" ]]; then
    echo "Usage: bash train_method.sh --method <3dgs|baseline|moe|moe_gnn|moe_tree|tree_gnn> -d <dataset> -g <gpu>"
    exit 1
fi

if [[ -z "${logdir}" ]]; then
    logdir="${method}"
fi

time=$(date "+%Y-%m-%d_%H:%M:%S")
outdir="outputs/${data}/${logdir}/${time}"

common_args=(
    --eval
    -s "data/${data}"
    --lod "${lod}"
    --gpu "${gpu}"
    --voxel_size "${voxel_size}"
    --update_init_factor "${update_init_factor}"
    --appearance_dim "${appearance_dim}"
    --ratio "${ratio}"
    --iterations "${iterations}"
    --port "${port}"
    -m "${outdir}"
)
[[ -n "${add_opacity_dist}" ]] && common_args+=("${add_opacity_dist}")
[[ -n "${add_cov_dist}" ]] && common_args+=("${add_cov_dist}")
[[ -n "${add_color_dist}" ]] && common_args+=("${add_color_dist}")
[[ -n "${update_until}" ]] && common_args+=(--update_until "${update_until}")

moe_loss_args=(
    --moe_top_k "${moe_top_k}"
    --moe_warmup "${moe_warmup}"
    --lambda_opacity "${lambda_opacity}"
    --lambda_load "${lambda_load}"
    --lambda_router_entropy "${lambda_router_entropy}"
)

gnn_args=(
    --use_gnn
    --gnn_layers "${gnn_layers}"
    --gnn_hidden_dim "${gnn_hidden_dim}"
    --gnn_k "${gnn_k}"
    --gnn_gamma_init "${gnn_gamma_init}"
)

resolve_moe_ckpt() {
    local who="$1"
    if [[ "${warm_start_mode}" == "auto" && -z "${start_baseline_mlp}" ]]; then
        start_baseline_mlp="$(ls -td "outputs/${data}/moe/"*/point_cloud/iteration_30000/checkpoints.pth 2>/dev/null | head -1 || true)"
    fi
    if [[ -z "${start_baseline_mlp}" || ! -e "${start_baseline_mlp}" ]]; then
        echo "[error] ${who} needs a finished MoE run first: --method moe"
        exit 1
    fi
}

resolve_moe_ply() {
    local who="$1"
    if [[ -n "${start_gaussian_ply}" && -e "${start_gaussian_ply}" ]]; then
        return 0
    fi
    local ckpt_dir
    if [[ -n "${start_baseline_mlp}" && -e "${start_baseline_mlp}" ]]; then
        ckpt_dir="$(dirname "${start_baseline_mlp}")"
        if [[ -f "${ckpt_dir}/point_cloud.ply" ]]; then
            start_gaussian_ply="${ckpt_dir}/point_cloud.ply"
            return 0
        fi
    fi
    start_gaussian_ply="$(ls -td "outputs/${data}/moe/"*/point_cloud/iteration_30000/point_cloud.ply 2>/dev/null | head -1 || true)"
    if [[ -z "${start_gaussian_ply}" || ! -e "${start_gaussian_ply}" ]]; then
        echo "[error] ${who} needs MoE point_cloud.ply"
        exit 1
    fi
}

# Same geometry recipe as moe_gnn: keep MoE points (no point rebuild).
keep_moe_points() {
    if [[ -z "${update_until}" ]]; then
        update_until=0
        common_args+=(--update_until 0)
    fi
}

case "${method}" in
    3dgs)
        echo "[method] 3dgs = official 3D Gaussian Splatting (Kerbl et al., TOG 2023)"
        echo "[note]   external trainer; Scaffold-GS voxel/lod/appearance flags are ignored"
        python scripts/train_3dgs.py -d "${data}" -g "${gpu}" -l "${logdir}" --iterations "${iterations}"
        ;;

    baseline)
        echo "[method] baseline = Scaffold-GS (shared MLP)"
        python train.py "${common_args[@]}" --num_experts 1
        ;;

    moe)
        echo "[method] moe = flat MoE (8 experts, no GNN)"
        python train.py "${common_args[@]}" \
            --num_experts "${num_experts}" \
            "${moe_loss_args[@]}"
        ;;

    moe_gnn)
        resolve_moe_ckpt moe_gnn
        resolve_moe_ply moe_gnn
        keep_moe_points
        echo "[method] moe_gnn = MoE + GNN (warm-start from moe)"
        echo "[from] MLPs   ${start_baseline_mlp}"
        echo "[from] points ${start_gaussian_ply}"
        python train.py "${common_args[@]}" \
            --moe_arch flat \
            --num_experts "${num_experts}" \
            "${moe_loss_args[@]}" \
            "${gnn_args[@]}" \
            --start_baseline_mlp "${start_baseline_mlp}" \
            --start_gaussian_ply "${start_gaussian_ply}"
        ;;

    moe_tree)
        tree_extra=()
        if [[ "${warm_start_mode}" != "off" ]]; then
            resolve_moe_ckpt moe_tree
            resolve_moe_ply moe_tree
            keep_moe_points
            tree_extra+=(--start_baseline_mlp "${start_baseline_mlp}" --start_gaussian_ply "${start_gaussian_ply}")
            echo "[method] moe_tree = growing Tree MoE, no GNN (start ${moe_branch_factor}, max_grows=${moe_max_total_grows}, max_leaves=${moe_max_leaves})"
            echo "[from] MLPs   ${start_baseline_mlp}"
            echo "[from] points ${start_gaussian_ply}"
        else
            echo "[method] moe_tree = growing Tree MoE from scratch (no warm-start)"
        fi
        python train.py "${common_args[@]}" \
            --moe_arch dynamic_tree \
            --moe_branch_factor "${moe_branch_factor}" \
            --moe_grow_factor "${moe_grow_factor}" \
            --moe_tree_depth "${moe_tree_depth}" \
            --moe_max_leaves "${moe_max_leaves}" \
            --moe_max_total_grows "${moe_max_total_grows}" \
            "${moe_loss_args[@]}" \
            --tree_grow_start "${tree_grow_start}" \
            --tree_grow_interval "${tree_grow_interval}" \
            --tree_grow_stop "${tree_grow_stop}" \
            --tree_grow_max_per_event "${tree_grow_max_per_event}" \
            --tree_grow_max_heads_per_event "${tree_grow_max_heads_per_event}" \
            --tree_grow_min_score_ratio "${tree_grow_min_score_ratio}" \
            --tree_grow_min_neff "${tree_grow_min_neff}" \
            --tree_grow_min_mass_frac "${tree_grow_min_mass_frac}" \
            --tree_grow_min_loss_ratio "${tree_grow_min_loss_ratio}" \
            --tree_grow_dense_iters "${tree_grow_dense_iters}" \
            ${tree_extra[@]+"${tree_extra[@]}"}
        ;;

    tree_gnn)
        resolve_moe_ckpt tree_gnn
        resolve_moe_ply tree_gnn
        keep_moe_points
        echo "[method] tree_gnn = growing Tree MoE + GNN (start ${moe_branch_factor}, max_grows=${moe_max_total_grows}, max_leaves=${moe_max_leaves})"
        echo "[from] MLPs   ${start_baseline_mlp}"
        echo "[from] points ${start_gaussian_ply}"
        python train.py "${common_args[@]}" \
            --moe_arch dynamic_tree \
            --moe_branch_factor "${moe_branch_factor}" \
            --moe_grow_factor "${moe_grow_factor}" \
            --moe_tree_depth "${moe_tree_depth}" \
            --moe_max_leaves "${moe_max_leaves}" \
            --moe_max_total_grows "${moe_max_total_grows}" \
            "${moe_loss_args[@]}" \
            "${gnn_args[@]}" \
            --tree_grow_start "${tree_grow_start}" \
            --tree_grow_interval "${tree_grow_interval}" \
            --tree_grow_stop "${tree_grow_stop}" \
            --tree_grow_max_per_event "${tree_grow_max_per_event}" \
            --tree_grow_max_heads_per_event "${tree_grow_max_heads_per_event}" \
            --tree_grow_min_score_ratio "${tree_grow_min_score_ratio}" \
            --tree_grow_min_neff "${tree_grow_min_neff}" \
            --tree_grow_min_mass_frac "${tree_grow_min_mass_frac}" \
            --tree_grow_min_loss_ratio "${tree_grow_min_loss_ratio}" \
            --tree_grow_dense_iters "${tree_grow_dense_iters}" \
            --start_baseline_mlp "${start_baseline_mlp}" \
            --start_gaussian_ply "${start_gaussian_ply}"
        ;;
esac
