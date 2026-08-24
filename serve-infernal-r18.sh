#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ROOT_DIR}/.venv-infernal-r18"
LAUNCHER="${ROOT_DIR}/third_party/vllm-infernal/serve-ds4-flash.sh"
API_KEY_FILE="${API_KEY_FILE:-${ROOT_DIR}/.api_key}"

[[ -x "${VENV}/bin/vllm" ]] || {
  printf 'Infernal r18 environment is missing. Run ./setup-infernal-r18.sh first.\n' >&2
  exit 1
}
[[ -x "${LAUNCHER}" ]] || {
  printf 'Infernal r18 launcher is missing: %s\n' "${LAUNCHER}" >&2
  exit 1
}
if [[ -z "${VLLM_API_KEY:-}" ]]; then
  [[ -r "${API_KEY_FILE}" ]] || {
    printf 'API key file is missing: %s\n' "${API_KEY_FILE}" >&2
    exit 1
  }
  IFS= read -r VLLM_API_KEY < "${API_KEY_FILE}"
  export VLLM_API_KEY
fi
[[ -n "${VLLM_API_KEY}" ]] || {
  printf 'VLLM_API_KEY must not be empty.\n' >&2
  exit 1
}

export PATH="${VENV}/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8000}"
export MODEL_PATH="${MODEL_PATH:-/data1/models/DeepSeek-V4-Flash-0731}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dsv4-flash}"
export MODE="${MODE:-dspark}"
export BACKEND="${BACKEND:-b12x-a8}"
export TP_SIZE="${TP_SIZE:-4}"
export DCP_SIZE="${DCP_SIZE:-1}"
export ALLREDUCE_MODE="${ALLREDUCE_MODE:-b12x}"
export DSPARK_TOKENS="${DSPARK_TOKENS:-5}"
export DSPARK_DEPTH_MODE="${DSPARK_DEPTH_MODE:-fixed}"
export DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-probabilistic}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export GRAPH="${GRAPH:-auto}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.975}"
export LOAD_FORMAT="${LOAD_FORMAT:-safetensors}"
export PREFIX_CACHE="${PREFIX_CACHE:-1}"
export ENABLE_FLASHINFER_AUTOTUNE="${ENABLE_FLASHINFER_AUTOTUNE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data1/uv-cache/vllm-dsv4-infernal-r18}"
export HF_HOME="${HF_HOME:-/data1/uv-cache/huggingface}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export VLLM_DISABLED_KERNELS="${VLLM_DISABLED_KERNELS:-MarlinFP8ScaledMMLinearKernel}"
NUMA_NODE="${NUMA_NODE:-0}"

[[ "${NUMA_NODE}" =~ ^[0-9]+$ ]] || {
  printf 'NUMA_NODE must be a non-negative integer; got %s\n' "${NUMA_NODE}" >&2
  exit 2
}
command -v numactl >/dev/null 2>&1 || {
  printf 'numactl is required for NUMA-local serving.\n' >&2
  exit 1
}

# The qualified launcher defaults to reasoning_effort=high. Keep reasoning on,
# but preserve this deployment's existing max-effort API default.
export EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-} --default-chat-template-kwargs.reasoning_effort=max"

printf 'NUMA binding: node=%s; OMP_NUM_THREADS=%s\n' \
  "${NUMA_NODE}" "${OMP_NUM_THREADS}" >&2
exec numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
  "${LAUNCHER}" "$@"
