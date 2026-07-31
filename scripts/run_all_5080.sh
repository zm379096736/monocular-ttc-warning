#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KITTI_ROOT="${KITTI_ROOT:-}"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/data}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/base.yaml}"
DEVICE="${DEVICE:-cuda}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
FORCE="${FORCE:-0}"
PREDICT_ALL="${PREDICT_ALL:-1}"
SEQUENCES="${SEQUENCES:-}"

LABEL_ROOT="${KITTI_ROOT}/label_02"
IMAGE_ROOT="${KITTI_ROOT}/image_02"
CACHE_ROOT="${WORK_DIR}/cache"
LABEL_CACHE="${CACHE_ROOT}/labels"
UPSTREAM_CACHE="${CACHE_ROOT}/upstream"
SEQUENCE_DATA="${WORK_DIR}/sequences"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/base"
PREDICTION_ROOT="${OUTPUT_ROOT}/predictions"
LOG_ROOT="${PROJECT_ROOT}/outputs/logs"

log() {
  printf '[run-all-5080] %s\n' "$*"
}

fail() {
  printf '[run-all-5080] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  KITTI_ROOT=/path/to/KITTI/tracking/training ./scripts/run_all_5080.sh

Optional environment variables:
  WORK_DIR=/path/to/work       Cache and generated sequence data directory
  SEQUENCES="0000 0001"       Run only selected sequences (default: all label files)
  FORCE=1                      Recompute completed stages
  PREDICT_ALL=0                Skip per-sequence prediction JSONL files
  VENV_DIR=/path/to/.venv      Override virtual environment location
  TORCH_INDEX_URL=...          Used by setup_5080.sh when PyTorch is missing
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

[[ -n "${KITTI_ROOT}" ]] || {
  usage
  fail "KITTI_ROOT is required."
}
[[ -d "${LABEL_ROOT}" ]] || fail "KITTI label directory not found: ${LABEL_ROOT}"
[[ -d "${IMAGE_ROOT}" ]] || fail "KITTI image directory not found: ${IMAGE_ROOT}"
[[ -f "${CONFIG}" ]] || fail "Configuration not found: ${CONFIG}"

mkdir -p "${LABEL_CACHE}" "${UPSTREAM_CACHE}" "${PREDICTION_ROOT}" "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  log "Virtual environment is missing; running setup_5080.sh"
  VENV_DIR="${VENV_DIR}" "${PROJECT_ROOT}/scripts/setup_5080.sh"
fi
PYTHON="${VENV_DIR}/bin/python"

"${PYTHON}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Run scripts/setup_5080.sh and inspect the driver.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
PY

declare -a sequence_ids=()
if [[ -n "${SEQUENCES}" ]]; then
  read -r -a sequence_ids <<< "${SEQUENCES}"
else
  while IFS= read -r label_file; do
    sequence_ids+=("$(basename "${label_file}" .txt)")
  done < <(find "${LABEL_ROOT}" -maxdepth 1 -type f -name '*.txt' | sort)
fi

[[ ${#sequence_ids[@]} -ge 3 ]] || fail "At least 3 sequences are required for train/validation/test splitting."
log "Sequences (${#sequence_ids[@]}): ${sequence_ids[*]}"

declare -a label_files=()
declare -a upstream_files=()
for sequence_id in "${sequence_ids[@]}"; do
  source_label="${LABEL_ROOT}/${sequence_id}.txt"
  source_images="${IMAGE_ROOT}/${sequence_id}"
  label_output="${LABEL_CACHE}/${sequence_id}.jsonl"
  upstream_output="${UPSTREAM_CACHE}/${sequence_id}.jsonl"
  [[ -f "${source_label}" ]] || fail "Missing label file: ${source_label}"
  [[ -d "${source_images}" ]] || fail "Missing image directory: ${source_images}"

  if [[ "${FORCE}" == "1" || ! -s "${label_output}" ]]; then
    log "[${sequence_id}] Generating TTC labels"
    label_temporary="${label_output}.tmp"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/build_kitti_ttc_labels.py" \
      --label-file "${source_label}" \
      --sequence-id "${sequence_id}" \
      --output "${label_temporary}"
    mv -f "${label_temporary}" "${label_output}"
  else
    log "[${sequence_id}] Reusing TTC labels"
  fi

  if [[ "${FORCE}" == "1" || ! -s "${upstream_output}" ]]; then
    log "[${sequence_id}] Extracting frozen upstream features"
    upstream_temporary="${upstream_output}.tmp"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/extract_upstream.py" \
      --input "${source_images}" \
      --sequence-id "${sequence_id}" \
      --fps 10 \
      --output "${upstream_temporary}" \
      --config "${CONFIG}" \
      --device "${DEVICE}" \
      --half
    mv -f "${upstream_temporary}" "${upstream_output}"
  else
    log "[${sequence_id}] Reusing frozen upstream cache"
  fi
  label_files+=("${label_output}")
  upstream_files+=("${upstream_output}")
done

ALL_LABELS="${CACHE_ROOT}/all_labels.jsonl"
ALL_UPSTREAM="${CACHE_ROOT}/all_upstream.jsonl"
"${PYTHON}" "${PROJECT_ROOT}/scripts/merge_jsonl.py" \
  --inputs "${label_files[@]}" \
  --output "${ALL_LABELS}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/merge_jsonl.py" \
  --inputs "${upstream_files[@]}" \
  --output "${ALL_UPSTREAM}"

if [[ "${FORCE}" == "1" || ! -s "${SEQUENCE_DATA}/train.jsonl" || ! -s "${SEQUENCE_DATA}/validation.jsonl" || ! -s "${SEQUENCE_DATA}/test.jsonl" ]]; then
  log "Preparing leakage-safe sequence splits"
  "${PYTHON}" "${PROJECT_ROOT}/scripts/prepare_sequences.py" \
    --records "${ALL_UPSTREAM}" \
    --labels "${ALL_LABELS}" \
    --output "${SEQUENCE_DATA}" \
    --config "${CONFIG}"
else
  log "Reusing prepared sequence dataset"
fi

if [[ "${FORCE}" == "1" || ! -s "${OUTPUT_ROOT}/best.pt" ]]; then
  log "Training temporal weighting MLP"
  "${PYTHON}" "${PROJECT_ROOT}/scripts/train.py" \
    --data "${SEQUENCE_DATA}" \
    --config "${CONFIG}" \
    --output "${OUTPUT_ROOT}" \
    --device "${DEVICE}"
else
  log "Reusing checkpoint ${OUTPUT_ROOT}/best.pt"
fi

log "Evaluating learned fusion and traditional baselines"
"${PYTHON}" "${PROJECT_ROOT}/scripts/evaluate.py" \
  --data "${SEQUENCE_DATA}" \
  --checkpoint "${OUTPUT_ROOT}/best.pt" \
  --output "${OUTPUT_ROOT}/test_metrics.json" \
  --device "${DEVICE}"

if [[ "${PREDICT_ALL}" == "1" ]]; then
  for sequence_id in "${sequence_ids[@]}"; do
    log "[${sequence_id}] Exporting streaming predictions"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/predict_cache.py" \
      --records "${UPSTREAM_CACHE}/${sequence_id}.jsonl" \
      --checkpoint "${OUTPUT_ROOT}/best.pt" \
      --normalizer "${SEQUENCE_DATA}/normalizer.json" \
      --output "${PREDICTION_ROOT}/${sequence_id}.jsonl" \
      --device "${DEVICE}"
  done
fi

log "Completed successfully."
log "Metrics: ${OUTPUT_ROOT}/test_metrics.json"
log "Checkpoint: ${OUTPUT_ROOT}/best.pt"
log "Predictions: ${PREDICTION_ROOT}"
log "Log: ${LOG_FILE}"

