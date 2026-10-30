#!/bin/bash
set -euo pipefail

FRAMEWORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CROSS_SPECIES_CONFIG:-${FRAMEWORK_DIR}/config.sh}"
[ -r "${CONFIG_FILE}" ] || {
    echo "ERROR: configuration not found: ${CONFIG_FILE}" >&2
    echo "Copy config.example.sh to config.sh and edit the data paths." >&2
    exit 1
}
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_path() {
    [ -e "$1" ] || die "Required path not found: $1"
}

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_fasta> <output_name> <output_dir> [--query-cover PERCENT] [--plant-workers N] [--overwrite] [--prepare-only]"
    echo "  --max     Maximum protein intensity per orthogroup/tissue (default)"
    echo "  --median  Median nonzero protein intensity per orthogroup/tissue"
    echo "  --sum     Sum of protein intensities per orthogroup/tissue"
    echo "  --overwrite  Replace an existing result directory (required for reruns)"
    echo "  --query-cover  Minimum DIAMOND query coverage percentage (default: 50)"
    echo "  --plant-workers  Concurrent plant-level renderers (default: 4)"
    echo "  --prepare-only  Generate the reusable run script without executing it"
    exit 1
fi

require_cmd realpath
require_cmd python
require_cmd diamond
require_cmd awk
require_cmd sort
require_cmd tee
require_cmd sha256sum
require_cmd stat
require_cmd mktemp

INPUT_FASTA=$(realpath "$1")
OUTPUT_NAME="$2"
OUTPUT_DIR_ARG="$3"
shift 3

INTENSITY_AGGREGATION="max"
DIAMOND_QUERY_COVER=50
PLANT_WORKERS=4
OVERWRITE=false
PREPARE_ONLY=false
SEARCH_WORKERS="${SEARCH_WORKERS:-10}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --max) INTENSITY_AGGREGATION="max" ;;
        --median) INTENSITY_AGGREGATION="median" ;;
        --sum) INTENSITY_AGGREGATION="sum" ;;
        --query-cover)
            [ "$#" -ge 2 ] || die "--query-cover requires a percentage"
            DIAMOND_QUERY_COVER="$2"
            shift
            ;;
        --plant-workers)
            [ "$#" -ge 2 ] || die "--plant-workers requires a positive integer"
            PLANT_WORKERS="$2"
            shift
            ;;
        --overwrite) OVERWRITE=true ;;
        --prepare-only) PREPARE_ONLY=true ;;
        *) die "Unknown option '$1'." ;;
    esac
    shift
done

if [[ ! "${DIAMOND_QUERY_COVER}" =~ ^[0-9]+$ ]] \
   || [ "${DIAMOND_QUERY_COVER}" -lt 1 ] \
   || [ "${DIAMOND_QUERY_COVER}" -gt 100 ]; then
    die "--query-cover must be an integer from 1 to 100"
fi
if [[ ! "${PLANT_WORKERS}" =~ ^[0-9]+$ ]] \
   || [ "${PLANT_WORKERS}" -lt 1 ] \
   || [ "${PLANT_WORKERS}" -gt 10 ]; then
    die "--plant-workers must be an integer from 1 to 10"
fi

[ -f "${INPUT_FASTA}" ] || die "Input FASTA file not found: ${INPUT_FASTA}"
[ -r "${INPUT_FASTA}" ] || die "Input FASTA file is not readable: ${INPUT_FASTA}"

if [[ "${OUTPUT_NAME}" =~ [[:space:]/] ]]; then
    die "output_name must not contain spaces or '/': ${OUTPUT_NAME}"
fi

if [[ ! "${OUTPUT_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    die "output_name contains unsupported characters: ${OUTPUT_NAME} (allowed: A-Z a-z 0-9 . _ -)"
fi

PYTHON_BIN="$(command -v python)"
DIAMOND_BIN="$(command -v diamond)"

PY_SCRIPTS_DIR="${FRAMEWORK_DIR}/scripts"
TREE_HEATMAP_SCRIPT="${PY_SCRIPTS_DIR}/generate_tree_ordered_presence_heatmap.py"
INTERACTIVE_TREE_HEATMAP_SCRIPT="${PY_SCRIPTS_DIR}/generate_interactive_tree_tissue_heatmap.py"
FAMILY_MAPPING_SCRIPT="${PY_SCRIPTS_DIR}/map_queries_by_family.py"
FAMILY_MERGED_HEATMAP_SCRIPT="${PY_SCRIPTS_DIR}/build_family_merged_heatmaps.py"

require_path "${PY_SCRIPTS_DIR}/fix_fasta.py"
require_path "${TREE_HEATMAP_SCRIPT}"
require_path "${INTERACTIVE_TREE_HEATMAP_SCRIPT}"
require_path "${FAMILY_MAPPING_SCRIPT}"
require_path "${FAMILY_MERGED_HEATMAP_SCRIPT}"
require_path "${FAMILY_DIAMOND_DB_DIR}"
require_path "${SPECIES_TREE}"
require_path "${CORE_CROPS_ORTHOGROUPS}"
require_path "$(dirname "${CORE_CROPS_ORTHOGROUPS}")/Orthogroups_UnassignedGenes.tsv"
require_path "${METADATA}"
require_path "${TISSUE_ONTOLOGY}"
require_path "${PNUMBER_TISSUES}"
require_path "${PROTEIN_DESCRIPTIONS}"
require_path "${PROBLEM_REPORT}"
require_path "${UNIPROT_MAPPING}"
require_path "${PROTEIN_INTENSITIES}"
require_path "${FULL_ORTHOGROUP_MATRIX}"

OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR_ARG}")"

if [ "${OUTPUT_DIR}" = "/" ] || [ "$(basename "${OUTPUT_DIR}")" != "${OUTPUT_NAME}" ]; then
    die "output_dir must be a dedicated directory whose basename equals output_name ('${OUTPUT_NAME}'): ${OUTPUT_DIR}"
fi

for protected_dir in "${FRAMEWORK_DIR}" "${DATA_DIR:-}" "${PY_SCRIPTS_DIR}"; do
    [ -n "${protected_dir}" ] || continue
    [ "${OUTPUT_DIR}" != "${protected_dir}" ] || die "Refusing protected output directory: ${OUTPUT_DIR}"
done

if [ -d "${OUTPUT_DIR}" ] && [ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    if [ "${OVERWRITE}" != true ]; then
        die "Output directory is not empty: ${OUTPUT_DIR}. Use --overwrite for an intentional clean rerun."
    fi

    rm -rf -- "${OUTPUT_DIR}"
fi

mkdir -p "${OUTPUT_DIR}/logs"
JOB_SCRIPT="${OUTPUT_DIR}/${OUTPUT_NAME}_run.sh"


cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash
set -euo pipefail
trap 'echo "ERROR at line \${LINENO}: \${BASH_COMMAND}" >&2' ERR

INPUT_FASTA="${INPUT_FASTA}"
OUTPUT_NAME="${OUTPUT_NAME}"
OUTPUT_DIR="${OUTPUT_DIR}"
PY_SCRIPTS_DIR="${PY_SCRIPTS_DIR}"
PYTHON_BIN="${PYTHON_BIN}"
DIAMOND_BIN="${DIAMOND_BIN}"
TREE_HEATMAP_SCRIPT="${TREE_HEATMAP_SCRIPT}"
INTERACTIVE_TREE_HEATMAP_SCRIPT="${INTERACTIVE_TREE_HEATMAP_SCRIPT}"
FAMILY_MAPPING_SCRIPT="${FAMILY_MAPPING_SCRIPT}"
FAMILY_MERGED_HEATMAP_SCRIPT="${FAMILY_MERGED_HEATMAP_SCRIPT}"
FAMILY_DIAMOND_DB_DIR="${FAMILY_DIAMOND_DB_DIR}"
SPECIES_TREE="${SPECIES_TREE}"
INTENSITY_AGGREGATION="${INTENSITY_AGGREGATION}"
PROTEIN_INTENSITIES="${PROTEIN_INTENSITIES}"
FULL_ORTHOGROUP_MATRIX="${FULL_ORTHOGROUP_MATRIX}"

core_crops_orthogroups="${CORE_CROPS_ORTHOGROUPS}"

METADATA="${METADATA}"
TISSUE_ONTOLOGY="${TISSUE_ONTOLOGY}"
PNUMBER_TISSUES="${PNUMBER_TISSUES}"
PROTEIN_DESCRIPTIONS="${PROTEIN_DESCRIPTIONS}"
PROBLEM_REPORT="${PROBLEM_REPORT}"
UNIPROT_MAPPING="${UNIPROT_MAPPING}"
QUERY_PROTEIN_MATRIX="\${OUTPUT_DIR}/query_protein_tissue_mcIBAQ_matrix.tsv"
DIAMOND_QUERY_COVER="${DIAMOND_QUERY_COVER}"
PLANT_WORKERS="${PLANT_WORKERS}"
SEARCH_WORKERS="${SEARCH_WORKERS}"

export LC_ALL=C
export PYTHONHASHSEED=0
export MPLBACKEND=Agg
export MPLCONFIGDIR="\${OUTPUT_DIR}/.mplconfig"
export XDG_CACHE_HOME="\${OUTPUT_DIR}/.mplconfig"
mkdir -p "\${OUTPUT_DIR}/logs" "\${MPLCONFIGDIR}"

{
    echo -e "record_type\tname\tvalue\tdetails"
    echo -e "setting\tstarted_at\t\$(date --iso-8601=seconds)\t"
    echo -e "setting\tinput_fasta\t\${INPUT_FASTA}\t"
    echo -e "setting\toutput_name\t\${OUTPUT_NAME}\t"
    echo -e "setting\taggregation\t\${INTENSITY_AGGREGATION}\t"
    echo -e "setting\tdiamond_query_cover\t\${DIAMOND_QUERY_COVER}\t"
    echo -e "setting\tdiamond_low_complexity_fallback\tsensitive_unmasked_missing_pairs\tNormal hits are retained; fallback uses the same query coverage and E-value thresholds"
    echo -e "setting\tplant_workers\t\${PLANT_WORKERS}\t"
    echo -e "software\tpython\t\$("\${PYTHON_BIN}" --version 2>&1)\t\${PYTHON_BIN}"
    echo -e "software\tdiamond\t\$("\${DIAMOND_BIN}" version 2>&1 | head -n 1)\t\${DIAMOND_BIN}"
    echo -e "software\tpython_packages\t\$("\${PYTHON_BIN}" -c 'from importlib.metadata import version; names=["pandas","numpy","matplotlib","seaborn","biopython","scipy","plotly"]; print(",".join(f"{name}={version(name)}" for name in names))')\t"
    for manifest_file in "\${INPUT_FASTA}" "\${core_crops_orthogroups}" \
                         "\${SPECIES_TREE}" "\${METADATA}" \
                         "\${TISSUE_ONTOLOGY}" "\${PNUMBER_TISSUES}" \
                         "\${PROTEIN_DESCRIPTIONS}" \
                         "\${PROBLEM_REPORT}" "\${UNIPROT_MAPPING}" \
                         "\${FULL_ORTHOGROUP_MATRIX}" \
                         "\${PY_SCRIPTS_DIR}/fix_fasta.py" \
                         "\${FAMILY_MAPPING_SCRIPT}" \
                         "\${FAMILY_MERGED_HEATMAP_SCRIPT}" \
                         "\${TREE_HEATMAP_SCRIPT}" \
                         "\${INTERACTIVE_TREE_HEATMAP_SCRIPT}"; do
        checksum="\$(sha256sum "\${manifest_file}" | awk '{print \$1}')"
        echo -e "input_sha256\t\${manifest_file}\t\${checksum}\t"
    done
    while IFS= read -r intensity_file; do
        checksum="\$(sha256sum "\${intensity_file}" | awk '{print \$1}')"
        echo -e "intensity_sha256\t\$(basename "\${intensity_file}")\t\${checksum}\t\$(stat -c '%s' "\${intensity_file}") bytes"
    done < <(find "\${PROTEIN_INTENSITIES}" -maxdepth 1 -type f -print | sort)
} > "\${OUTPUT_DIR}/run_manifest.tsv"

echo "[STEP 1] Normalizing query FASTA at \$(date)"
"\${PYTHON_BIN}" "\${PY_SCRIPTS_DIR}/fix_fasta.py" \
  "\${INPUT_FASTA}" \
  "\${OUTPUT_DIR}/${OUTPUT_NAME}_fasta.faa"

echo "[STEP 2] Mapping every query independently against each plant family at \$(date)"
"\${PYTHON_BIN}" "\${FAMILY_MAPPING_SCRIPT}" \
    --query-fasta "\${OUTPUT_DIR}/${OUTPUT_NAME}_fasta.faa" \
    --metadata "\${METADATA}" \
    --family-db-dir "\${FAMILY_DIAMOND_DB_DIR}" \
    --orthogroup-dir "\$(dirname "\${core_crops_orthogroups}")" \
    --output "\${OUTPUT_DIR}/other_files/per_family_best_hits.tsv" \
    --hit-dir "\${OUTPUT_DIR}/.family_diamond_hits" \
    --discard-raw-hits \
    --workers "\${SEARCH_WORKERS}" \
    --threads-per-search 1 \
    --query-cover "\${DIAMOND_QUERY_COVER}" \
    2>&1 | tee "\${OUTPUT_DIR}/logs/family_mapping.log"

echo "[STEP 3] Merging family-mapped OGs and generating final heatmaps at \$(date)"
"\${PYTHON_BIN}" "\${FAMILY_MERGED_HEATMAP_SCRIPT}" \
    --results "\${OUTPUT_DIR}" \
    --mappings "\${OUTPUT_DIR}/other_files/per_family_best_hits.tsv" \
    --full-matrix "\${FULL_ORTHOGROUP_MATRIX}" \
    --orthogroups "\${core_crops_orthogroups}" \
    --tree "\${SPECIES_TREE}" \
    --metadata "\${METADATA}" \
    --protein-descriptions "\${PROTEIN_DESCRIPTIONS}" \
    --intensities "\${PROTEIN_INTENSITIES}" \
    --outliers "\${PROBLEM_REPORT}" \
    --tissue-ontology "\${TISSUE_ONTOLOGY}" \
    --pnumber-tissues "\${PNUMBER_TISSUES}" \
    --uniprot-mapping "\${UNIPROT_MAPPING}" \
    2>&1 | tee "\${OUTPUT_DIR}/logs/family_merged_heatmaps.log"

echo "[STEP 4] Organizing outputs at \$(date)"
mkdir -p "\${OUTPUT_DIR}/other_files"
# Matplotlib's generated font cache is runtime-only and is not a result.
rm -rf -- "\${MPLCONFIGDIR}"

# Record checksums of the principal results before moving intermediates.
for result_file in \
    "\${OUTPUT_DIR}/${OUTPUT_NAME}_fasta.faa" \
    "\${QUERY_PROTEIN_MATRIX}" \
    "\${OUTPUT_DIR}/${OUTPUT_NAME}_heatmap_tree_all_tissues_family_OGs_merged.jpg" \
    "\${OUTPUT_DIR}/${OUTPUT_NAME}_heatmap_tree_species.jpg"; do
    checksum="\$(sha256sum "\${result_file}" | awk '{print \$1}')"
    echo -e "result_sha256\t\$(basename "\${result_file}")\t\${checksum}\t" >> "\${OUTPUT_DIR}/run_manifest.tsv"
done
while IFS= read -r interactive_file; do
    checksum="\$(sha256sum "\${interactive_file}" | awk '{print \$1}')"
    echo -e "result_sha256\t\$(basename "\${interactive_file}")\t\${checksum}\t" >> "\${OUTPUT_DIR}/run_manifest.tsv"
done < <(find "\${OUTPUT_DIR}" -maxdepth 1 -type f \
    -name 'interactive_heatmap_*.html' \
    -print | sort)
while IFS= read -r heatmap_file; do
    checksum="\$(sha256sum "\${heatmap_file}" | awk '{print \$1}')"
    relative_name="\${heatmap_file#"\${OUTPUT_DIR}/"}"
    echo -e "result_sha256\t\${relative_name}\t\${checksum}\t" >> "\${OUTPUT_DIR}/run_manifest.tsv"
done < <(find "\${OUTPUT_DIR}/other_heatmaps" -type f \
    \( -name '*.jpg' -o -name '*.png' \) -print | sort)

is_keep_item() {
    local name="\$1"
    case "\${name}" in
        other_files|\
        other_heatmaps|\
        \${OUTPUT_NAME}_fasta.faa|\
        query_protein_tissue_mcIBAQ_matrix.tsv|\
        \${OUTPUT_NAME}_heatmap_tree_all_tissues_family_OGs_merged.jpg|\
        interactive_heatmap_*.html|\
        \${OUTPUT_NAME}_heatmap_tree_species.jpg)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

shopt -s nullglob dotglob
for item in "\${OUTPUT_DIR}"/*; do
    base_name="\$(basename "\${item}")"
    # Defensive guard: never move organizer targets into other_files/.
    if [ "\${item}" -ef "\${OUTPUT_DIR}/other_files" ]; then
        continue
    fi
    if is_keep_item "\${base_name}"; then
        continue
    fi
    mv "\${item}" "\${OUTPUT_DIR}/other_files/"
done
shopt -u nullglob dotglob

echo -e "setting\tfinished_at\t\$(date --iso-8601=seconds)\t" >> "\${OUTPUT_DIR}/other_files/run_manifest.tsv"
echo "DONE at \$(date)"
EOF

chmod +x "${JOB_SCRIPT}"
if [ "${PREPARE_ONLY}" = true ]; then
    echo "Prepared job script: ${JOB_SCRIPT}"
    exit 0
fi
echo "Running workflow in the current terminal."
bash "${JOB_SCRIPT}"
