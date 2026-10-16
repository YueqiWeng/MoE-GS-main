#!/usr/bin/env bash
# Sweep the three canonical methods across dataset suites.
#
# Methods:
#   baseline  — Scaffold-GS shared MLP
#   moe       — flat MoE
#   moe_gnn   — MoE + GNN (needs moe ckpt)
#   moe_tree  — growing Tree MoE, no GNN (needs moe ckpt)
#   tree_gnn  — growing Tree MoE + GNN (needs moe ckpt)
#
# Usage:
#   bash sweep.sh --suite mip360 -g 0
#   bash sweep.sh --suite mip360 --methods moe,moe_tree,moe_gnn,tree_gnn -g 0
#   bash sweep.sh --suite mip360 --skip-done -g 0
#   bash sweep.sh --suite ablate-tree --tree-grows 0,2,4 -g 0
#   bash sweep.sh --suite ablate-components --ablate-components nocl,noload,nowarm -g 0
#   bash sweep.sh --suite ablate-components-quick --ablate-components nocl,noload,nowarm -g 0
#
# Component ablations (moe_tree full model, no GNN):
#   nocl    — moe_warmup=0
#   noload  — lambda_load=0
#   nowarm  — --no_warm_start
#   noopac  — lambda_opacity=0  (optional)
#
# Logs: sweep_logs/sweep_<timestamp>.log

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

GPU="${GPU:-0}"
SUITE="mip360"
SCENES_CSV=""
METHODS_CSV="baseline,moe,moe_gnn,moe_tree,tree_gnn"
SKIP_DONE=0
DRY_RUN=0
CONTINUE_ON_ERROR=1
ITERATIONS=""
TREE_GROWS_CSV=""
ABLATE_COMPONENTS_CSV=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -g|--gpu) GPU="$2"; shift ;;
        --suite) SUITE="$2"; shift ;;
        --scenes) SCENES_CSV="$2"; shift ;;
        --methods) METHODS_CSV="$2"; shift ;;
        --tree-grows) TREE_GROWS_CSV="$2"; shift ;;
        --ablate-components) ABLATE_COMPONENTS_CSV="$2"; shift ;;
        --skip-done) SKIP_DONE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --stop-on-error) CONTINUE_ON_ERROR=0 ;;
        --iterations) ITERATIONS="$2"; shift ;;
        -h|--help)
            sed -n '2,22p' "$0"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# conda
if [[ "${CONDA_DEFAULT_ENV:-}" != "scaffold_gs" ]]; then
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate scaffold_gs
    set -u
fi
unset NVCC_PREPEND_FLAGS NVCC_PREPEND_FLAGS_BACKUP 2>/dev/null || true
export CUDA_HOME="${CONDA_PREFIX:-}"
export PATH="${CONDA_PREFIX:-}/bin:${PATH}"

mkdir -p sweep_logs
STAMP="$(date '+%Y-%m-%d_%H:%M:%S')"
LOG="sweep_logs/sweep_${STAMP}.log"
SUMMARY="sweep_logs/summary_${STAMP}.txt"

log() { echo "$@" | tee -a "$LOG"; }

scene_exists() {
    local s="$1"
    [[ -d "data/$s/sparse" || -f "data/$s/transforms_train.json" || -d "data/$s/images" ]]
}

scene_hparams() {
    local s="$1"
    case "$s" in
        mipnerf360/*)      echo "0.001 16 0" ;;
        tandt/*)           echo "0.01 16 0" ;;
        db/*|blending/*)   echo "0.005 16 0" ;;
        bungeenerf/*)      echo "0 128 30" ;;
        nerfsynthetic/*)   echo "0.001 4 0" ;;
        *)                 echo "0.001 16 0" ;;
    esac
}

expand_suite() {
    local suite="$1"
    case "$suite" in
        mip360|mipnerf360)
            echo "mipnerf360/bicycle mipnerf360/bonsai mipnerf360/counter mipnerf360/garden mipnerf360/kitchen mipnerf360/room mipnerf360/stump"
            ;;
        tandt|tnt)
            echo "tandt/train tandt/truck"
            ;;
        db|blending)
            echo "db/drjohnson db/playroom"
            ;;
        bungee|bungeenerf)
            echo "bungeenerf/amsterdam bungeenerf/bilbao bungeenerf/hollywood bungeenerf/pompidou bungeenerf/quebec bungeenerf/rome bungeenerf/barcelona bungeenerf/chicago"
            ;;
        nerf|nerfsynthetic)
            echo "nerfsynthetic/chair nerfsynthetic/drums nerfsynthetic/ficus nerfsynthetic/hotdog nerfsynthetic/lego nerfsynthetic/materials nerfsynthetic/mic nerfsynthetic/ship"
            ;;
        paper|all)
            echo "$(expand_suite mip360) $(expand_suite tandt) $(expand_suite db) $(expand_suite bungee) $(expand_suite nerf)"
            ;;
        quick)
            echo "mipnerf360/kitchen tandt/truck db/playroom nerfsynthetic/lego"
            ;;
        ablate-tree|tree-ablate)
            echo "mipnerf360/kitchen mipnerf360/bonsai mipnerf360/stump"
            ;;
        ablate-components|ablate-comp)
            # Full component ablation: all scenes with flat-MoE + tree baselines
            echo "$(expand_suite mip360) $(expand_suite tandt) $(expand_suite db) $(expand_suite nerf)"
            ;;
        ablate-components-quick|ablate-comp-quick)
            # Small pilot (DB + T&T only)
            echo "db/drjohnson db/playroom tandt/train tandt/truck"
            ;;
        *)
            echo "Unknown suite: $suite" >&2
            exit 1
            ;;
    esac
}

SCENES=()
if [[ -n "$SCENES_CSV" ]]; then
    IFS=',' read -r -a SCENES <<< "$SCENES_CSV"
else
    IFS=',' read -r -a SUITES <<< "$SUITE"
    for su in "${SUITES[@]}"; do
        # shellcheck disable=SC2207
        SCENES+=($(expand_suite "$su"))
    done
fi

IFS=',' read -r -a METHODS <<< "$METHODS_CSV"

method_done() {
    local scene="$1" method="$2"
    # Canonical folders + legacy aliases for the winning MoE+GNN recipe.
    local folders=("${method}")
    if [[ "${method}" == "moe_gnn" ]]; then
        folders+=("moe_then_gnn" "moe_gnn_frozen")
    fi

    local dir f cfg
    for f in "${folders[@]}"; do
        dir="outputs/${scene}/${f}"
        [[ -d "$dir" ]] || continue
        while IFS= read -r -d '' cfg; do
            # Require results.json next to cfg
            [[ -f "$(dirname "$cfg")/results.json" ]] || continue
            if [[ "${method}" == "moe_gnn" ]]; then
                # Current moe_gnn = warm-start MoE + load ply + GNN.
                # Skip old from-scratch / no-ply moe_gnn folders.
                if grep -q "start_gaussian_ply='[^']" "$cfg" 2>/dev/null; then
                    return 0
                fi
                if grep -q 'start_gaussian_ply="[^"]' "$cfg" 2>/dev/null; then
                    return 0
                fi
                # moe_then_gnn / moe_gnn_frozen folders are always the ply recipe
                if [[ "$f" == "moe_then_gnn" || "$f" == "moe_gnn_frozen" ]]; then
                    return 0
                fi
            else
                return 0
            fi
        done < <(find "$dir" -name cfg_args -print0 2>/dev/null)
        # Fallback: any results.json for non-moe_gnn
        if [[ "${method}" != "moe_gnn" ]]; then
            if find "$dir" -name results.json 2>/dev/null | grep -q .; then
                return 0
            fi
        fi
    done
    return 1
}

latest_moe_ckpt() {
    local scene="$1"
    ls -td "outputs/${scene}/moe/"*/point_cloud/iteration_30000/checkpoints.pth 2>/dev/null | head -1 || true
}

ablate_grows_done() {
    local scene="$1" grows="$2"
    local dir="outputs/${scene}/ablate_grows${grows}"
    [[ -d "$dir" ]] || return 1
    find "$dir" -name results.json 2>/dev/null | grep -q .
}

run_tree_grow_ablate() {
    local scene="$1" grows="$2"
    local hp vsize init_factor lod ckpt
    hp="$(scene_hparams "$scene")"
    read -r vsize init_factor lod <<< "$hp"

    ckpt="$(latest_moe_ckpt "$scene")"
    if [[ -z "$ckpt" ]]; then
        log "[error] no flat MoE ckpt for $scene; run --method moe first"
        return 1
    fi

    if [[ "$SKIP_DONE" -eq 1 ]] && ablate_grows_done "$scene" "$grows"; then
        log "[skip-done] $scene / ablate_grows${grows}"
        return 0
    fi

    local extra=(
        -d "$scene" -g "$GPU"
        --voxel_size "$vsize" --update_init_factor "$init_factor" --lod "$lod"
        -l "ablate_grows${grows}"
        --moe_max_total_grows "$grows"
        --start_baseline_mlp "$ckpt"
    )
    if [[ -n "$ITERATIONS" ]]; then
        extra+=( --iterations "$ITERATIONS" )
    fi

    log ""
    log "========================================"
    log "[$(date '+%F %T')] moe_tree ablate  grows=${grows}  on  ${scene}"
    log "[from] ${ckpt}"
    log "========================================"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "DRY: bash train_method.sh --method moe_tree ${extra[*]}"
        return 0
    fi

    set +e
    bash train_method.sh --method moe_tree "${extra[@]}" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    set -e

    if [[ "$rc" -ne 0 ]]; then
        log "[FAIL] ablate_grows${grows} on $scene  exit=$rc"
        if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
            return "$rc"
        fi
    else
        log "[OK] ablate_grows${grows} on $scene"
    fi
    return 0
}

ablate_component_done() {
    local scene="$1" name="$2"
    local dir="outputs/${scene}/ablate_${name}"
    [[ -d "$dir" ]] || return 1
    find "$dir" -name results.json 2>/dev/null | grep -q .
}

# Component knockout on top of moe_tree (full model). No GNN.
run_component_ablate() {
    local scene="$1" name="$2"
    local hp vsize init_factor lod ckpt
    hp="$(scene_hparams "$scene")"
    read -r vsize init_factor lod <<< "$hp"

    if [[ "$SKIP_DONE" -eq 1 ]] && ablate_component_done "$scene" "$name"; then
        log "[skip-done] $scene / ablate_${name}"
        return 0
    fi

    local extra=(
        -d "$scene" -g "$GPU"
        --voxel_size "$vsize" --update_init_factor "$init_factor" --lod "$lod"
        -l "ablate_${name}"
    )
    if [[ -n "$ITERATIONS" ]]; then
        extra+=( --iterations "$ITERATIONS" )
    fi

    case "$name" in
        nocl)
            extra+=( --moe_warmup 0 )
            ckpt="$(latest_moe_ckpt "$scene")"
            if [[ -z "$ckpt" ]]; then
                log "[error] no flat MoE ckpt for $scene; run --method moe first"
                return 1
            fi
            extra+=( --start_baseline_mlp "$ckpt" )
            ;;
        noload)
            extra+=( --lambda_load 0 )
            ckpt="$(latest_moe_ckpt "$scene")"
            if [[ -z "$ckpt" ]]; then
                log "[error] no flat MoE ckpt for $scene; run --method moe first"
                return 1
            fi
            extra+=( --start_baseline_mlp "$ckpt" )
            ;;
        noopac)
            extra+=( --lambda_opacity 0 )
            ckpt="$(latest_moe_ckpt "$scene")"
            if [[ -z "$ckpt" ]]; then
                log "[error] no flat MoE ckpt for $scene; run --method moe first"
                return 1
            fi
            extra+=( --start_baseline_mlp "$ckpt" )
            ;;
        nowarm)
            extra+=( --no_warm_start )
            ;;
        *)
            log "[error] unknown component ablation: $name (use: nocl,noload,nowarm,noopac)"
            return 1
            ;;
    esac

    log ""
    log "========================================"
    log "[$(date '+%F %T')] moe_tree ablate  component=${name}  on  ${scene}"
    log "========================================"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "DRY: bash train_method.sh --method moe_tree ${extra[*]}"
        return 0
    fi

    set +e
    bash train_method.sh --method moe_tree "${extra[@]}" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    set -e

    if [[ "$rc" -ne 0 ]]; then
        log "[FAIL] ablate_${name} on $scene  exit=$rc"
        if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
            return "$rc"
        fi
    else
        log "[OK] ablate_${name} on $scene"
    fi
    return 0
}

run_one() {
    local scene="$1" method="$2"
    local hp vsize init_factor lod
    hp="$(scene_hparams "$scene")"
    read -r vsize init_factor lod <<< "$hp"

    local extra=( -d "$scene" -g "$GPU" --voxel_size "$vsize" --update_init_factor "$init_factor" --lod "$lod" )
    if [[ -n "$ITERATIONS" ]]; then
        extra+=( --iterations "$ITERATIONS" )
    fi

    local method_flag=""
    case "$method" in
        baseline|scaffold|scaffold-gs)
            method_flag="baseline"; method="baseline" ;;
        moe|flat_moe|flat-moe)
            method_flag="moe"; method="moe" ;;
        moe_gnn|moe-gnn|moe+gnn|moe_then_gnn)
            method_flag="moe_gnn"
            method="moe_gnn"
            local ckpt
            ckpt="$(latest_moe_ckpt "$scene")"
            if [[ -n "$ckpt" ]]; then
                extra+=( --start_baseline_mlp "$ckpt" )
            else
                log "[error] no flat MoE ckpt for $scene; moe_gnn needs --method moe first"
                return 1
            fi
            ;;
        moe_tree|tree|tree_moe|growing_tree)
            method_flag="moe_tree"
            method="moe_tree"
            local ckpt
            ckpt="$(latest_moe_ckpt "$scene")"
            if [[ -n "$ckpt" ]]; then
                extra+=( --start_baseline_mlp "$ckpt" )
            else
                log "[error] no flat MoE ckpt for $scene; moe_tree needs --method moe first"
                return 1
            fi
            ;;
        tree_gnn|dynamic_gnn|tree+gnn)
            method_flag="tree_gnn"
            method="tree_gnn"
            local ckpt
            ckpt="$(latest_moe_ckpt "$scene")"
            if [[ -n "$ckpt" ]]; then
                extra+=( --start_baseline_mlp "$ckpt" )
            else
                log "[error] no flat MoE ckpt for $scene; tree_gnn needs --method moe first"
                return 1
            fi
            ;;
        *)
            log "[error] unknown method: $method (use: baseline,moe,moe_gnn,moe_tree,tree_gnn)"
            return 1
            ;;
    esac

    if [[ "$SKIP_DONE" -eq 1 ]] && method_done "$scene" "$method"; then
        log "[skip-done] $scene / $method"
        return 0
    fi

    log ""
    log "========================================"
    log "[$(date '+%F %T')] $method  on  $scene  (voxel=$vsize init=$init_factor lod=$lod)"
    log "========================================"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "DRY: bash train_method.sh --method $method_flag ${extra[*]}"
        return 0
    fi

    set +e
    bash train_method.sh --method "$method_flag" "${extra[@]}" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    set -e

    if [[ "$rc" -ne 0 ]]; then
        log "[FAIL] $method on $scene  exit=$rc"
        if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
            return "$rc"
        fi
    else
        log "[OK] $method on $scene"
    fi
    return 0
}

write_summary() {
    python3 - "$SUMMARY" <<'PY'
import glob, json, os, sys
out_path = sys.argv[1]
methods = ["baseline", "moe", "moe_gnn", "moe_tree", "tree_gnn"]
method_aliases = {
    "baseline": "baseline",
    "moe": "moe",
    "moe_gnn": "MoE+GNN",
    "moe_tree": "Tree",
    "tree_gnn": "Tree+GNN",
}
scenes = sorted({
    "/".join(p.split("/")[1:3])
    for p in glob.glob("outputs/*/*/*/results.json")
})
rows = []
header = ["scene"] + [method_aliases.get(m, m) for m in methods]
rows.append(header)
for scene in scenes:
    row = [scene]
    for m in methods:
        best = None
        folders = [m]
        if m == "moe_gnn":
            folders += ["moe_then_gnn", "moe_gnn_frozen"]
        if m == "tree_gnn":
            folders += ["dynamic_tree_moe_gnn"]
        for folder in folders:
            for p in glob.glob(f"outputs/{scene}/{folder}/*/results.json"):
                try:
                    with open(p) as f:
                        d = json.load(f)
                    # prefer ours_30000
                    key = "ours_30000" if "ours_30000" in d else next(iter(d), None)
                    if key is None:
                        continue
                    psnr = float(d[key].get("PSNR", float("nan")))
                    if best is None or psnr > best:
                        best = psnr
                except Exception:
                    pass
        row.append(f"{best:.3f}" if best is not None else "-")
    rows.append(row)

widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
lines = []
for r in rows:
    lines.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
text = "\n".join(lines) + "\n"
print(text)
with open(out_path, "w") as f:
    f.write(text)
PY
}

trap 'log "[trap] writing summary..."; write_summary || true' EXIT INT TERM

if [[ -n "$TREE_GROWS_CSV" ]]; then
    IFS=',' read -r -a TREE_GROWS <<< "$TREE_GROWS_CSV"
    log "Sweep start  tree-grows ablation  grows=${TREE_GROWS_CSV}  gpu=$GPU"
elif [[ -n "$ABLATE_COMPONENTS_CSV" ]]; then
    IFS=',' read -r -a ABLATE_COMPONENTS <<< "$ABLATE_COMPONENTS_CSV"
    log "Sweep start  component ablation  comps=${ABLATE_COMPONENTS_CSV}  gpu=$GPU"
else
    log "Sweep start  suite=$SUITE  methods=${METHODS_CSV}  gpu=$GPU"
fi
log "Scenes: ${SCENES[*]}"
log "Log: $LOG"

if [[ -n "$TREE_GROWS_CSV" ]]; then
    for scene in "${SCENES[@]}"; do
        if ! scene_exists "$scene"; then
            log "[skip] missing data: $scene"
            continue
        fi
        for grows in "${TREE_GROWS[@]}"; do
            if ! run_tree_grow_ablate "$scene" "$grows"; then
                log "[fail] $scene / ablate_grows${grows}"
                if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
                    exit 1
                fi
            fi
        done
    done
elif [[ -n "$ABLATE_COMPONENTS_CSV" ]]; then
    for scene in "${SCENES[@]}"; do
        if ! scene_exists "$scene"; then
            log "[skip] missing data: $scene"
            continue
        fi
        for comp in "${ABLATE_COMPONENTS[@]}"; do
            if ! run_component_ablate "$scene" "$comp"; then
                log "[fail] $scene / ablate_${comp}"
                if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
                    exit 1
                fi
            fi
        done
    done
else
    for scene in "${SCENES[@]}"; do
        if ! scene_exists "$scene"; then
            log "[skip] missing data: $scene"
            continue
        fi
        for method in "${METHODS[@]}"; do
            if ! run_one "$scene" "$method"; then
                log "[fail] $scene / $method"
                if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
                    exit 1
                fi
            fi
        done
    done
fi

log "Sweep done."
write_summary
log "Summary: $SUMMARY"
