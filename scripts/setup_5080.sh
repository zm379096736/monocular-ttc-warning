#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTHON_BIN="${PYTHON_BIN:-}"

log() {
  printf '[setup-5080] %s\n' "$*"
}

fail() {
  printf '[setup-5080] ERROR: %s\n' "$*" >&2
  exit 1
}

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found. Install the NVIDIA driver first."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    fail "Python was not found. Install Python 3.11."
  fi
fi

"${PYTHON_BIN}" - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(f"Python 3.10-3.12 is required, found {sys.version.split()[0]}")
print(f"Python: {sys.version.split()[0]}")
PY

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  log "Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
"${VENV_PYTHON}" -m pip install --upgrade pip

if ! "${VENV_PYTHON}" -c "import torch, torchvision; assert torch.cuda.is_available()" >/dev/null 2>&1; then
  log "Installing CUDA-enabled PyTorch from ${TORCH_INDEX_URL}"
  "${VENV_PYTHON}" -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
fi

log "Installing project dependencies"
"${VENV_PYTHON}" -m pip install -e "${PROJECT_ROOT}[dev]"

"${VENV_PYTHON}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in PyTorch. Check the driver and PyTorch CUDA build.")

major, minor = torch.cuda.get_device_capability(0)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {major}.{minor}")
PY

log "Running unit tests"
"${VENV_DIR}/bin/pytest" "${PROJECT_ROOT}/tests"
log "Environment is ready."

