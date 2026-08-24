#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${UV_BIN:-}" ]]; then
  UV_BIN="$(command -v uv 2>/dev/null || true)"
  UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"
fi
VENV="${ROOT_DIR}/.venv-infernal-r18"
BUILD_DIR="${ROOT_DIR}/.build-infernal-r18"
VLLM_SRC="${ROOT_DIR}/third_party/vllm-infernal"
B12X_SRC="${ROOT_DIR}/third_party/b12x-infernal"
FLASHINFER_SRC="${ROOT_DIR}/third_party/flashinfer-infernal"
XGRAMMAR_SRC="${ROOT_DIR}/third_party/xgrammar-infernal"
BLACKWELL_SRC="${ROOT_DIR}/third_party/blackwell-llm-docker"
RELEASE_ROOT="${ROOT_DIR}/third_party/blackwell-llm-docker/patches/releases/infernal-invocation-r18"
UV_CACHE_DIR="${UV_CACHE_DIR:-/data1/uv-cache}"

BLACKWELL_REPO="https://github.com/local-inference-lab/blackwell-llm-docker.git"
BLACKWELL_COMMIT="bb3a4f954cfcc831dd4520a883a402eb09e66e62"
VLLM_REPO="https://github.com/local-inference-lab/vllm.git"
VLLM_BASE="6dc2f516688fe6f84c6994dcd20fddf296853a6c"
VLLM_TREE="f0fa1cefc1865d316c2478525f550e7646addc40"
B12X_REPO="https://github.com/local-inference-lab/b12x.git"
B12X_BASE="c25cdba2c1df7a69b2d7771e4243e12a8fbf19d5"
B12X_TREE="75787c7a7431b3bea414d2ebf5f2b8671b23eb33"
FLASHINFER_REPO="https://github.com/voipmonitor/flashinfer.git"
FLASHINFER_COMMIT="1ac6942776b383c6b03c7a5805a22e72a3e3349f"
FLASHINFER_VERSION="0.6.18+cu133"
XGRAMMAR_REPO="https://github.com/mlc-ai/xgrammar.git"
XGRAMMAR_COMMIT="2ea71da4ccb997a06928c9fb69b99f330da56697"
XGRAMMAR_VERSION="0.2.5"
VLLM_VERSION="0.26.1rc0+infernal.invocation.cu133.r18.vllmf0fa1ce.b12x75787c7"
TRITON_KERNELS_REPO="https://github.com/triton-lang/triton.git"
TRITON_KERNELS_COMMIT="0add68262ab0a2e33b84524346cb27cbb2787356"

export UV_CACHE_DIR
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="12.0a"
export CMAKE_CUDA_ARCHITECTURES="120a"
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
export FLASHINFER_LOCAL_VERSION="cu133"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export NVCC_THREADS="${NVCC_THREADS:-4}"
export MAX_JOBS="${MAX_JOBS:-32}"

if [[ -n "${PROXY_URL:-}" && -z "${HTTP_PROXY:-}" && -z "${http_proxy:-}" ]]; then
  export HTTP_PROXY="${PROXY_URL}"
  export HTTPS_PROXY="${PROXY_URL}"
  export http_proxy="${PROXY_URL}"
  export https_proxy="${PROXY_URL}"
fi

[[ -x "${UV_BIN}" ]] || {
  printf 'uv not found: %s\n' "${UV_BIN}" >&2
  exit 1
}
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || {
  printf 'CUDA 13.3 nvcc not found: %s/bin/nvcc\n' "${CUDA_HOME}" >&2
  exit 1
}

mkdir -p "${BUILD_DIR}" "${UV_CACHE_DIR}" "${ROOT_DIR}/third_party"

if [[ ! -d "${BLACKWELL_SRC}/.git" ]]; then
  git clone --filter=blob:none --no-checkout \
    "${BLACKWELL_REPO}" "${BLACKWELL_SRC}"
  git -C "${BLACKWELL_SRC}" fetch --depth=1 origin "${BLACKWELL_COMMIT}"
  git -C "${BLACKWELL_SRC}" checkout --detach "${BLACKWELL_COMMIT}"
elif [[ "$(git -C "${BLACKWELL_SRC}" rev-parse HEAD 2>/dev/null || true)" != "${BLACKWELL_COMMIT}" ]]; then
  [[ -z "$(git -C "${BLACKWELL_SRC}" status --porcelain)" ]] || {
    printf 'blackwell-llm-docker has local changes: %s\n' "${BLACKWELL_SRC}" >&2
    exit 1
  }
  git -C "${BLACKWELL_SRC}" fetch --depth=1 origin "${BLACKWELL_COMMIT}"
  git -C "${BLACKWELL_SRC}" checkout --detach "${BLACKWELL_COMMIT}"
fi
[[ -f "${RELEASE_ROOT}/vllm/integration.patch" \
  && -f "${RELEASE_ROOT}/b12x/integration.patch" ]] || {
  printf 'Infernal Invocation r18 patch set is incomplete: %s\n' \
    "${RELEASE_ROOT}" >&2
  exit 1
}

prepare_composed_source() {
  local name=$1 repo=$2 base=$3 expected_tree=$4 patch_file=$5 destination=$6
  local actual_tree

  if [[ ! -d "${destination}/.git" ]]; then
    git clone --filter=blob:none --no-checkout "${repo}" "${destination}"
  fi

  actual_tree="$(git -C "${destination}" write-tree 2>/dev/null || true)"
  if [[ "${actual_tree}" == "${expected_tree}" ]]; then
    printf '%s source tree already matches %s\n' "${name}" "${expected_tree}"
    return
  fi

  if [[ -n "$(git -C "${destination}" status --porcelain)" ]]; then
    printf '%s has local changes and does not match the r18 tree: %s\n' \
      "${name}" "${destination}" >&2
    exit 1
  fi

  git -C "${destination}" fetch --depth=1 origin "${base}"
  git -C "${destination}" checkout --detach "${base}"
  git -C "${destination}" apply "${patch_file}"
  git -C "${destination}" add -A
  actual_tree="$(git -C "${destination}" write-tree)"
  [[ "${actual_tree}" == "${expected_tree}" ]] || {
    printf '%s source tree mismatch: got %s, expected %s\n' \
      "${name}" "${actual_tree}" "${expected_tree}" >&2
    exit 1
  }
}

prepare_composed_source \
  vLLM "${VLLM_REPO}" "${VLLM_BASE}" "${VLLM_TREE}" \
  "${RELEASE_ROOT}/vllm/integration.patch" "${VLLM_SRC}"
prepare_composed_source \
  B12X "${B12X_REPO}" "${B12X_BASE}" "${B12X_TREE}" \
  "${RELEASE_ROOT}/b12x/integration.patch" "${B12X_SRC}"

if [[ ! -d "${FLASHINFER_SRC}/.git" ]]; then
  git clone --filter=blob:none --no-checkout "${FLASHINFER_REPO}" "${FLASHINFER_SRC}"
fi
if [[ "$(git -C "${FLASHINFER_SRC}" rev-parse HEAD 2>/dev/null || true)" != "${FLASHINFER_COMMIT}" ]]; then
  # A --no-checkout clone reports the whole index as deleted even though its
  # worktree is intentionally empty. Only reject changes once files exist.
  if find "${FLASHINFER_SRC}" -mindepth 1 -maxdepth 1 ! -name .git \
      -print -quit | grep -q .; then
    [[ -z "$(git -C "${FLASHINFER_SRC}" status --porcelain 2>/dev/null || true)" ]] || {
      printf 'FlashInfer has local changes: %s\n' "${FLASHINFER_SRC}" >&2
      exit 1
    }
  fi
  git -C "${FLASHINFER_SRC}" fetch --depth=1 origin "${FLASHINFER_COMMIT}"
  git -C "${FLASHINFER_SRC}" checkout --detach "${FLASHINFER_COMMIT}"
  git -C "${FLASHINFER_SRC}" submodule update --init --recursive
fi

if [[ ! -d "${XGRAMMAR_SRC}/.git" ]]; then
  git clone --filter=blob:none --no-checkout "${XGRAMMAR_REPO}" "${XGRAMMAR_SRC}"
fi
if [[ "$(git -C "${XGRAMMAR_SRC}" rev-parse HEAD 2>/dev/null || true)" != "${XGRAMMAR_COMMIT}" ]]; then
  [[ -z "$(git -C "${XGRAMMAR_SRC}" status --porcelain 2>/dev/null || true)" ]] || {
    printf 'XGrammar has local changes: %s\n' "${XGRAMMAR_SRC}" >&2
    exit 1
  }
  git -C "${XGRAMMAR_SRC}" fetch --depth=1 origin "${XGRAMMAR_COMMIT}"
  git -C "${XGRAMMAR_SRC}" checkout --detach "${XGRAMMAR_COMMIT}"
fi
git -C "${XGRAMMAR_SRC}" submodule update --init --recursive

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${UV_BIN}" venv --python 3.12 "${VENV}"
fi
PYTHON="${VENV}/bin/python"

"${UV_BIN}" pip install --python "${PYTHON}" \
  pip \
  -r "${VLLM_SRC}/requirements/build/cuda.txt" \
  -r "${VLLM_SRC}/requirements/common.txt"

"${UV_BIN}" pip install --python "${PYTHON}" \
  'numba==0.65.0' \
  'torchvision==0.28.0' \
  'fastsafetensors>=0.3.3' \
  'nvidia-cudnn-frontend>=1.19.1' \
  'nvidia-cutlass-dsl[cu13]==4.6.2' \
  'cupy-cuda13x==13.6.0' \
  'sortedcontainers==2.4.0' \
  'aiofile==3.11.1' \
  'aiofiles==25.1.0' \
  'apache-tvm-ffi==0.1.11' \
  'torch-c-dlpack-ext==0.1.5' \
  'z3-solver==4.15.4.0' \
  'tokenspeed-triton==3.8.10.post20260721' \
  'nvidia-ml-py' \
  'cuda-python' \
  'protoc-wheel-0==30.2' \
  'scikit-build-core>=0.10.0' \
  'rich>=13'
"${UV_BIN}" pip install --python "${PYTHON}" --no-deps \
  'quack-kernels==0.6.4' \
  'humming-kernels[cu13]==0.1.12' \
  'tilelang==0.1.12' \
  'tokenspeed-mla==0.1.8' \
  'nvidia-cutlass-dsl-libs-cu13==4.6.2'

FLASHINFER_WHEEL_DIR="${BUILD_DIR}/flashinfer-wheels"
mkdir -p "${FLASHINFER_WHEEL_DIR}"
if ! compgen -G "${FLASHINFER_WHEEL_DIR}/flashinfer_python-${FLASHINFER_VERSION}*.whl" >/dev/null; then
  (
    cd "${FLASHINFER_SRC}"
    FLASHINFER_ENABLE_SM90=0 BUILD_NVEP=0 BUILD_NCCL_EP=0 BUILD_NIXL_EP=0 \
      "${PYTHON}" -m pip wheel --no-build-isolation --no-deps \
      -w "${FLASHINFER_WHEEL_DIR}" .
  )
fi
if ! compgen -G "${FLASHINFER_WHEEL_DIR}/flashinfer_jit_cache-${FLASHINFER_VERSION}*.whl" >/dev/null; then
  mkdir -p "${FLASHINFER_WHEEL_DIR}"
  (
    cd "${FLASHINFER_SRC}"
    "${PYTHON}" -m pip wheel --no-build-isolation --no-deps \
      -w "${FLASHINFER_WHEEL_DIR}" ./flashinfer-jit-cache
  )
fi
"${PYTHON}" -m pip install --no-deps --force-reinstall \
  "${FLASHINFER_WHEEL_DIR}"/flashinfer_python-*.whl \
  "${FLASHINFER_WHEEL_DIR}"/flashinfer_jit_cache-*.whl

XGRAMMAR_WHEEL_DIR="${BUILD_DIR}/xgrammar-wheels"
mkdir -p "${XGRAMMAR_WHEEL_DIR}"
if ! compgen -G "${XGRAMMAR_WHEEL_DIR}/xgrammar-${XGRAMMAR_VERSION}-*.whl" >/dev/null; then
  SKBUILD_BUILD_DIR="${BUILD_DIR}/xgrammar-build" \
  CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}" \
    "${PYTHON}" -m pip wheel --no-build-isolation --no-deps \
      -w "${XGRAMMAR_WHEEL_DIR}" "${XGRAMMAR_SRC}"
fi
"${PYTHON}" -m pip install --no-deps --force-reinstall \
  "${XGRAMMAR_WHEEL_DIR}"/xgrammar-${XGRAMMAR_VERSION}-*.whl

"${PYTHON}" -m pip install --no-build-isolation --no-deps --force-reinstall \
  "${B12X_SRC}"

VLLM_BUILD_DIR="${BUILD_DIR}/vllm-extensions"
if [[ ! -f "${VLLM_BUILD_DIR}/_C_stable_libtorch.abi3.so" ]]; then
  cmake -S "${VLLM_SRC}" -B "${VLLM_BUILD_DIR}" -G Ninja \
    -U MARLIN_GEN_SCRIPT_HASH_AND_ARCH \
    -U MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DVLLM_TARGET_DEVICE=cuda \
    -DVLLM_PYTHON_EXECUTABLE="${PYTHON}" \
    -DNVCC_THREADS="${NVCC_THREADS}"
  cmake --build "${VLLM_BUILD_DIR}" --parallel "${MAX_JOBS}"
fi

FA4_INSTALL_DIR="${BUILD_DIR}/vllm-flash-attn-python"
cmake --install "${VLLM_BUILD_DIR}" \
  --prefix "${FA4_INSTALL_DIR}" --component _vllm_fa4_cutedsl_C
cp -a "${FA4_INSTALL_DIR}/vllm/vllm_flash_attn/cute" \
  "${VLLM_SRC}/vllm/vllm_flash_attn/"
while IFS= read -r extension; do
  case "${extension}" in
    */vllm-flash-attn/*)
      destination="${VLLM_SRC}/vllm/vllm_flash_attn"
      ;;
    */_deps/*)
      continue
      ;;
    *)
      destination="${VLLM_SRC}/vllm"
      ;;
  esac
  cp "${extension}" "${destination}/$(basename "${extension}")"
done < <(find "${VLLM_BUILD_DIR}" -name '*.abi3.so')

TRITON_KERNELS_SRC="${ROOT_DIR}/third_party/triton-kernels-infernal"
if [[ ! -d "${TRITON_KERNELS_SRC}/.git" ]]; then
  git clone --filter=blob:none --no-checkout \
    "${TRITON_KERNELS_REPO}" "${TRITON_KERNELS_SRC}"
fi
if [[ "$(git -C "${TRITON_KERNELS_SRC}" rev-parse HEAD 2>/dev/null || true)" != "${TRITON_KERNELS_COMMIT}" ]]; then
  git -C "${TRITON_KERNELS_SRC}" fetch --depth=1 origin "${TRITON_KERNELS_COMMIT}"
  git -C "${TRITON_KERNELS_SRC}" checkout --detach "${TRITON_KERNELS_COMMIT}"
fi
mkdir -p "${VLLM_SRC}/vllm/third_party/triton_kernels"
cp -a "${TRITON_KERNELS_SRC}/python/triton_kernels/triton_kernels/." \
  "${VLLM_SRC}/vllm/third_party/triton_kernels/"

export CARGO_HOME="${BUILD_DIR}/cargo"
export RUSTUP_HOME="${BUILD_DIR}/rustup"
export PATH="${CARGO_HOME}/bin:${PATH}"
if [[ ! -x "${CARGO_HOME}/bin/rustup" ]]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile minimal --default-toolchain none --no-modify-path
fi
RUST_TOOLCHAIN="$(sed -n 's/^channel = "\([^"]*\)"/\1/p' "${VLLM_SRC}/rust-toolchain.toml")"
rustup toolchain install "${RUST_TOOLCHAIN}"
rustup default "${RUST_TOOLCHAIN}"
export PROTOC="${PROTOC:-${VENV}/bin/protoc}"
[[ -x "${PROTOC}" ]] || {
  printf 'protoc not found: %s\n' "${PROTOC}" >&2
  exit 1
}
(
  cd "${VLLM_SRC}"
  "${PYTHON}" tools/build_rust.py --release
  VLLM_TARGET_DEVICE=empty \
  VLLM_VERSION_OVERRIDE="${VLLM_VERSION}" \
  VLLM_REQUIRE_RUST_FRONTEND=1 \
    "${PYTHON}" -m pip install --no-build-isolation --no-deps --force-reinstall .
)

SITE_PACKAGES="$(${PYTHON} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
INSTALLED_VLLM="${SITE_PACKAGES}/vllm"
find "${VLLM_SRC}/vllm" -maxdepth 1 -type f -name '*.abi3.so' \
  -exec cp -a -t "${INSTALLED_VLLM}" {} +
rm -rf "${INSTALLED_VLLM}/vllm_flash_attn"
cp -a "${VLLM_SRC}/vllm/vllm_flash_attn" "${INSTALLED_VLLM}/vllm_flash_attn"
rm -rf "${INSTALLED_VLLM}/third_party/triton_kernels"
mkdir -p "${INSTALLED_VLLM}/third_party/triton_kernels"
cp -a "${VLLM_SRC}/vllm/third_party/triton_kernels/." \
  "${INSTALLED_VLLM}/third_party/triton_kernels/"

"${PYTHON}" - <<PY
import importlib.metadata as metadata
import torch

assert metadata.version("vllm") == "${VLLM_VERSION}"
assert metadata.version("b12x") == "1.2.3"
assert metadata.version("flashinfer-python") == "${FLASHINFER_VERSION}"
assert metadata.version("xgrammar") == "${XGRAMMAR_VERSION}"
assert torch.__version__.startswith("2.13.0")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", metadata.version("vllm"))
print("b12x", metadata.version("b12x"))
print("flashinfer", metadata.version("flashinfer-python"))
print("xgrammar", metadata.version("xgrammar"))
PY

touch "${BUILD_DIR}/ready"
printf 'Infernal Invocation r18 uv environment is ready: %s\n' "${VENV}"
