#!/bin/bash
set -euo pipefail

FRAMEWORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CROSS_SPECIES_CONFIG:-${FRAMEWORK_DIR}/config.sh}"
[ -r "${CONFIG_FILE}" ] || {
    echo "ERROR: copy config.example.sh to config.sh and edit the paths" >&2
    exit 1
}
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

mkdir -p "${MATRIX_OUTPUT_DIR}"
BUILD_DIR="$(mktemp -d "${MATRIX_OUTPUT_DIR}/.matrix_build.XXXXXX")"
trap 'rm -rf -- "${BUILD_DIR}"' EXIT

python "${FRAMEWORK_DIR}/build_full_orthogroup_matrix.py" \
  --orthogroups "${CORE_CROPS_ORTHOGROUPS}" \
  --intensities "${PROTEIN_INTENSITIES}" \
  --ibaq-intensities "${PROTEIN_INTENSITIES}" \
  --tissue-ontology "${TISSUE_ONTOLOGY}" \
  --pnumber-tissues "${PNUMBER_TISSUES}" \
  --metadata "${METADATA}" \
  --skip-raw \
  --output-dir "${BUILD_DIR}"

test -s "${BUILD_DIR}/protein_orthogroup_pnumber_matrix_mcIBAQ_intensities.tsv"
mv -f -- \
  "${BUILD_DIR}/protein_orthogroup_pnumber_matrix_mcIBAQ_intensities.tsv" \
  "${FULL_ORTHOGROUP_MATRIX}"
echo "Rebuilt ${FULL_ORTHOGROUP_MATRIX}"
