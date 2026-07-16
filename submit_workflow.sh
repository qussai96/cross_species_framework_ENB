#!/bin/bash
set -euo pipefail

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

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_fasta> <output_name> <output_dir>"
    exit 1
fi

require_cmd realpath
require_cmd sbatch
require_cmd python
require_cmd diamond
require_cmd awk
require_cmd sort
require_cmd tee

INPUT_FASTA=$(realpath "$1")
OUTPUT_NAME="$2"
OUTPUT_DIR_ARG="$3"

[ -f "${INPUT_FASTA}" ] || die "Input FASTA file not found: ${INPUT_FASTA}"
[ -r "${INPUT_FASTA}" ] || die "Input FASTA file is not readable: ${INPUT_FASTA}"

if [[ "${OUTPUT_NAME}" =~ [[:space:]/] ]]; then
    die "output_name must not contain spaces or '/': ${OUTPUT_NAME}"
fi

if [[ ! "${OUTPUT_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    die "output_name contains unsupported characters: ${OUTPUT_NAME} (allowed: A-Z a-z 0-9 . _ -)"
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPTS_DIR="${REPO_DIR}/scripts"

CORE_CROPS_ORTHOGROUPS="/home/students/q.abbas/Scripts/40-OrthoFinder/ENB/Final/Results_Dec28/Orthogroups/Orthogroups.tsv"
DIAMOND_DB="/home/students/q.abbas/+proj-q.abbas/Workflow_orthogroups/Mapping_OG/Diamond/target"
METADATA="/home/students/q.abbas/Fetch_data/all_crops_metadata.tsv"
TISSUE_ONTOLOGY="/home/students/q.abbas/+proj-q.abbas/Workflow_orthogroups/ENB_TissueOntology_long.tsv"
OG_ANNOTATION_TABLE="/home/students/q.abbas/+proj-q.abbas/Workflow_orthogroups/Mapping_OG/OG_functional_annotation.tsv"
PROBLEM_REPORT="/home/students/q.abbas/+proj-q.abbas/Workflow_orthogroups/cross_species_framework_ENB/ProblemReport.tsv"

require_path "${PY_SCRIPTS_DIR}/fix_fasta.py"
require_path "${PY_SCRIPTS_DIR}/map_to_orthogroups.py"
require_path "${PY_SCRIPTS_DIR}/protein_orthogroup_po_matrix.py"
require_path "${PY_SCRIPTS_DIR}/plot_heatmap.py"
require_path "${PY_SCRIPTS_DIR}/generate_pathway_heatmaps.py"
require_path "${PY_SCRIPTS_DIR}/plot_plant_tissue_heatmap.py"
require_path "${CORE_CROPS_ORTHOGROUPS}"
if [ ! -e "${DIAMOND_DB}" ] && [ ! -e "${DIAMOND_DB}.dmnd" ]; then
    die "DIAMOND DB not found at '${DIAMOND_DB}' or '${DIAMOND_DB}.dmnd'"
fi
require_path "${METADATA}"
require_path "${TISSUE_ONTOLOGY}"
require_path "${OG_ANNOTATION_TABLE}"
require_path "${PROBLEM_REPORT}"

WORKFLOW_DIR="${REPO_DIR}"
mkdir -p "${OUTPUT_DIR_ARG}"
OUTPUT_DIR="$(realpath "${OUTPUT_DIR_ARG}")"
JOB_SCRIPT="${OUTPUT_DIR}/${OUTPUT_NAME}_job.sh"

if [ "${OUTPUT_DIR}" = "/" ] || [ "${OUTPUT_DIR}" = "${REPO_DIR}" ]; then
    die "Refusing unsafe output_dir '${OUTPUT_DIR}'. Use a dedicated results subdirectory."
fi

mkdir -p "${OUTPUT_DIR}/logs"

cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=${OUTPUT_NAME}
#SBATCH --output=${OUTPUT_DIR}/${OUTPUT_NAME}.%j.out
#SBATCH --error=${OUTPUT_DIR}/${OUTPUT_NAME}.%j.err
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G

set -euo pipefail
trap 'echo "ERROR at line \${LINENO}: \${BASH_COMMAND}" >&2' ERR

INPUT_FASTA="${INPUT_FASTA}"
OUTPUT_NAME="${OUTPUT_NAME}"
WORKFLOW_DIR="${WORKFLOW_DIR}"
OUTPUT_DIR="${OUTPUT_DIR}"
PY_SCRIPTS_DIR="${PY_SCRIPTS_DIR}"

core_crops_orthogroups="${CORE_CROPS_ORTHOGROUPS}"
PROTEIN_INTENSITIES="${WORKFLOW_DIR}/Combined_ibaq_intensities"
METADATA="${METADATA}"
TISSUE_ONTOLOGY="${TISSUE_ONTOLOGY}"
OG_ANNOTATION_TABLE="${OG_ANNOTATION_TABLE}"
PROBLEM_REPORT="${PROBLEM_REPORT}"

mkdir -p "\${OUTPUT_DIR}/${OUTPUT_NAME}_fasta" "\${OUTPUT_DIR}/logs"

echo "[STEP 1] OrthoFinder assignment"
python "\${PY_SCRIPTS_DIR}/fix_fasta.py" \
  "\${INPUT_FASTA}" \
  "\${OUTPUT_DIR}/${OUTPUT_NAME}_fasta/${OUTPUT_NAME}.faa"

diamond blastp -q "\${OUTPUT_DIR}/${OUTPUT_NAME}_fasta/${OUTPUT_NAME}.faa" \
 -d "${DIAMOND_DB}" -o "\${OUTPUT_DIR}/hits_raw.tsv" --threads 10 --outfmt 6 --max-target-seqs 25 --evalue 1e-6

# Keep only the single best hit per query (highest bitscore) — --max-target-seqs 1
# does NOT guarantee the best hit in DIAMOND (Shah et al. 2018, Bioinformatics).
sort -k1,1 -k12,12rn "\${OUTPUT_DIR}/hits_raw.tsv" | awk '!seen[\$1]++' > "\${OUTPUT_DIR}/hits.tsv"

python "\${PY_SCRIPTS_DIR}/map_to_orthogroups.py" "\${OUTPUT_DIR}/hits.tsv" "\${core_crops_orthogroups}" "\${OUTPUT_DIR}/Assigned_Orthogroups.tsv"

echo "[STEP 2] Writing matched orthogroup tables at \$(date)"
python - <<PY
import pandas as pd

input_fasta = "\${OUTPUT_DIR}/${OUTPUT_NAME}_fasta/${OUTPUT_NAME}.faa"
orthogroups_path = "\${OUTPUT_DIR}/Assigned_Orthogroups.tsv"
out_matched = "\${OUTPUT_DIR}/Orthogroups_matched_to_\${OUTPUT_NAME}.tsv"
out_annot = "\${OUTPUT_DIR}/Orthogroups_matched_with_annotations.tsv"

proteins = set()
with open(input_fasta, "r", encoding="utf-8") as handle:
    for line in handle:
        if line.startswith(">"):
            proteins.add(line[1:].strip().split()[0])

og = pd.read_csv(orthogroups_path, sep="\t", dtype=str).fillna("")
crop_cols = [c for c in og.columns if c != "Orthogroup"]

def row_has_match(row):
    for col in crop_cols:
        cell = row[col]
        if not cell:
            continue
        for token in str(cell).split(","):
            if token.strip() in proteins:
                return True
    return False

matched = og[og.apply(row_has_match, axis=1)].copy()
matched.to_csv(out_matched, sep="\t", index=False)

ann = pd.read_csv("\${OG_ANNOTATION_TABLE}", sep="\t", dtype=str)
ann = ann.rename(columns={"orthogroup_id": "Orthogroup"})
merged = matched[["Orthogroup"]].merge(ann, on="Orthogroup", how="left")
merged.to_csv(out_annot, sep="\t", index=False)

print(f"Matched orthogroups: {len(matched)}")
print(f"Wrote: {out_matched}")
print(f"Wrote: {out_annot}")
PY

echo "[STEP 3] Generating orthogroup_po matrix at \$(date)"
python "\${PY_SCRIPTS_DIR}/protein_orthogroup_po_matrix.py" \
  --input-fasta "\${OUTPUT_DIR}/${OUTPUT_NAME}_fasta/${OUTPUT_NAME}.faa" \
  --orthogroups "\${OUTPUT_DIR}/Assigned_Orthogroups.tsv" \
  --intensities "\${PROTEIN_INTENSITIES}" \
  --tissue-ontology "\${TISSUE_ONTOLOGY}" \
  --metadata "\${METADATA}" \
  --og-annotation-table "\${OG_ANNOTATION_TABLE}" \
        --output "\${OUTPUT_DIR}/protein_orthogroup_po_matrix_mcIBAQ_intensities.csv" \
  2>&1 | tee "\${OUTPUT_DIR}/logs/orthogroup_po_matrix.log"

echo "[STEP 4] Generating heatmaps and orthogroup summary outputs at \$(date)"
python "\${PY_SCRIPTS_DIR}/plot_heatmap.py" \
    --name "\${OUTPUT_NAME}" \
    --protein_orthogroup_po_matrix "\${OUTPUT_DIR}/protein_orthogroup_po_matrix_mcIBAQ_intensities.csv" \
    --orthogroups "\${OUTPUT_DIR}/Assigned_Orthogroups.tsv" \
    --input_fasta "\${OUTPUT_DIR}/${OUTPUT_NAME}_fasta/${OUTPUT_NAME}.faa" \
    --metadata "\${METADATA}" \
    --matched_orthogroups "\${OUTPUT_DIR}/Orthogroups_matched_to_\${OUTPUT_NAME}.tsv" \
    --output_dir "\${OUTPUT_DIR}" \
    2>&1 | tee "\${OUTPUT_DIR}/logs/plot_heatmap.log"

echo "[STEP 4B] Generating pathway matrix heatmaps at \$(date)"
mkdir -p "\${OUTPUT_DIR}/heatmaps"
python "\${PY_SCRIPTS_DIR}/generate_pathway_heatmaps.py" \
    --input "\${OUTPUT_DIR}/protein_orthogroup_po_matrix_mcIBAQ_intensities.csv" \
    --matched-tsv "\${OUTPUT_DIR}/Orthogroups_matched_to_\${OUTPUT_NAME}.tsv" \
    --output-dir "\${OUTPUT_DIR}/heatmaps" \
    2>&1 | tee "\${OUTPUT_DIR}/logs/generate_pathway_heatmaps.log"

echo "[STEP 5] Generating plant-level heatmaps at \$(date)"
mkdir -p "\${OUTPUT_DIR}/heatmaps/plant_level"
python "\${PY_SCRIPTS_DIR}/plot_plant_tissue_heatmap.py" \
    --matrix "\${OUTPUT_DIR}/protein_orthogroup_po_matrix_mcIBAQ_intensities.csv" \
    --orthogroups "\${OUTPUT_DIR}/Assigned_Orthogroups.tsv" \
    --intensities "\${PROTEIN_INTENSITIES}" \
    --input-fasta "\${OUTPUT_DIR}/${OUTPUT_NAME}_fasta/${OUTPUT_NAME}.faa" \
    --tissue-ontology "\${TISSUE_ONTOLOGY}" \
    --metadata "\${METADATA}" \
    --problem-report "\${PROBLEM_REPORT}" \
    --all-plants \
    --output-dir "\${OUTPUT_DIR}/heatmaps/plant_level" \
    2>&1 | tee "\${OUTPUT_DIR}/logs/plot_plant_tissue_heatmap.log"

echo "[STEP 6] Organizing outputs at \$(date)"
mkdir -p "\${OUTPUT_DIR}/files"
mkdir -p "\${OUTPUT_DIR}/heatmaps"

# Move any top-level heatmap image/html artifacts into heatmaps/.
shopt -s nullglob
for hm in "\${OUTPUT_DIR}"/heatmap*.png "\${OUTPUT_DIR}"/heatmap*.html; do
    mv "\${hm}" "\${OUTPUT_DIR}/heatmaps/"
done
shopt -u nullglob

# Ensure legacy top-level plant_level is nested under heatmaps/.
if [ -d "\${OUTPUT_DIR}/plant_level" ]; then
    mkdir -p "\${OUTPUT_DIR}/heatmaps/plant_level"
    shopt -s nullglob dotglob
    for legacy_item in "\${OUTPUT_DIR}/plant_level"/*; do
        mv "\${legacy_item}" "\${OUTPUT_DIR}/heatmaps/plant_level/"
    done
    shopt -u nullglob dotglob
    rmdir "\${OUTPUT_DIR}/plant_level" 2>/dev/null || true
fi

is_keep_item() {
    local name="$1"
    case "\${name}" in
        files|heatmaps|\
        protein_orthogroup_po_matrix_mcIBAQ_intensities.csv|\
        orthogroup_summary_barplot.png|\
        Orthogroups_matched_to_\${OUTPUT_NAME}.tsv|\
        orthogroups_enriched_thresholds.tsv|\
        orthogroups_broadly_shared_all_species.tsv)
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
    # Defensive guard: never move organizer target directories into files/.
    if [ "\${item}" -ef "\${OUTPUT_DIR}/files" ] || [ "\${item}" -ef "\${OUTPUT_DIR}/heatmaps" ]; then
        continue
    fi
    if is_keep_item "\${base_name}"; then
        continue
    fi
    mv "\${item}" "\${OUTPUT_DIR}/files/"
done
shopt -u nullglob dotglob

echo "DONE at \$(date)"
EOF

chmod +x "${JOB_SCRIPT}"
SBATCH_OUT=$(sbatch "${JOB_SCRIPT}")
echo "${SBATCH_OUT}"
echo "Job script: ${JOB_SCRIPT}"
