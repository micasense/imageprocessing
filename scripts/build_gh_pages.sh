#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-${ROOT}/_site}"
EXECUTE="${EXECUTE:-0}"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT}"

nbconvert_args=(
  --to html
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${EXECUTE}" == "1" ]]; then
  nbconvert_args+=(
    --ExecutePreprocessor.timeout=60
    --execute
  )
fi

shopt -s nullglob
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  mapfile -t notebooks < <(git ls-files '*.ipynb')
else
  notebooks=(*.ipynb)
fi
if [[ ${#notebooks[@]} -eq 0 ]]; then
  echo "No notebooks found in ${ROOT}" >&2
  exit 1
fi

jupyter nbconvert "${nbconvert_args[@]}" "${notebooks[@]}"

echo "Built ${#notebooks[@]} notebook(s) in ${OUTPUT_DIR}"
