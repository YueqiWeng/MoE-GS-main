#!/usr/bin/env bash
# Train Scaffold-GS baseline at n_offsets k=5,10,20, then plot Fig. 9 analogue.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != "scaffold_gs" ]]; then
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate scaffold_gs
    set -u
fi
unset NVCC_PREPEND_FLAGS NVCC_PREPEND_FLAGS_BACKUP 2>/dev/null || true
export CUDA_HOME="${CONDA_PREFIX:-}"
export PATH="${CONDA_PREFIX:-}/bin:${PATH}"

SCENE="${SCENE:-db/playroom}"
GPU="${GPU:-0}"
ITERATIONS="${ITERATIONS:-30000}"
VOXEL="${VOXEL:-0.005}"
INIT="${INIT:-16}"
LOD="${LOD:-0}"
LOG="sweep_logs/k_ablation_$(date '+%Y-%m-%d_%H:%M:%S').log"
mkdir -p sweep_logs
exec > >(tee -a "$LOG") 2>&1

rand_port() {
    echo $((10000 + RANDOM % 20000))
}

echo "[k-ablate] scene=${SCENE} gpu=${GPU} log=${LOG}"

for k in 5 10 20; do
    stamp="$(date '+%Y-%m-%d_%H:%M:%S')"
    outdir="outputs/${SCENE}/k_ablate_k${k}/${stamp}"
    echo ""
    echo "========================================"
    echo "[$(date '+%F %T')] n_offsets=${k}  ->  ${outdir}"
    echo "========================================"
    python train.py \
        --eval \
        -s "data/${SCENE}" \
        --lod "${LOD}" \
        --gpu "${GPU}" \
        --voxel_size "${VOXEL}" \
        --update_init_factor "${INIT}" \
        --appearance_dim 0 \
        --ratio 1 \
        --iterations "${ITERATIONS}" \
        --port "$(rand_port)" \
        -m "${outdir}" \
        --n_offsets "${k}" \
        --num_experts 1
done

python scripts/plot_k_ablation.py
echo "ALL_K_ABLATION_DONE"
