#!/bin/bash
set -e

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_fasta> <output_name> <output_dir>"
    exit 1
fi

INPUT_FASTA=$(realpath "$1")
OUTPUT_NAME="$2"
OUTPUT_DIR_ARG="$3"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPTS_DIR="${REPO_DIR}/scripts"

WORKFLOW_DIR="${REPO_DIR}"
mkdir -p "${OUTPUT_DIR_ARG}"
OUTPUT_DIR="$(realpath "${OUTPUT_DIR_ARG}")"
JOB_SCRIPT="${OUTPUT_DIR}/${OUTPUT_NAME}_job.sh"

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

set -e

INPUT_FASTA="${INPUT_FASTA}"
OUTPUT_NAME="${OUTPUT_NAME}"
WORKFLOW_DIR="${WORKFLOW_DIR}"
OUTPUT_DIR="${OUTPUT_DIR}"
PY_SCRIPTS_DIR="${PY_SCRIPTS_DIR}"

core_crops_orthogroups="/home/students/q.abbas/Scripts/40-OrthoFinder/ENB/Final/Results_Dec28/Orthogroups/Orthogroups.tsv"
PROTEIN_INTENSITIES="/home/students/q.abbas/Fetch_data/Proteomics/combined_proteins/"
METADATA="/home/students/q.abbas/Fetch_data/all_crops_metadata.tsv"
TISSUE_ONTOLOGY="/home/students/q.abbas/+proj-q.abbas/Workflow_orthogroups/ENB_TissueOntology_long.tsv"
OG_ANNOTATION_TABLE="/home/students/q.abbas/+proj-q.abbas/Workflow_orthogroups/Mapping_OG/OG_functional_annotation.tsv"

mkdir -p "\${OUTPUT_DIR}/${OUTPUT_NAME}_fasta" "\${OUTPUT_DIR}/logs"

echo "[STEP 1] OrthoFinder assignment"
python "\${PY_SCRIPTS_DIR}/fix_fasta.py" \
  "\${INPUT_FASTA}" \
  "\${OUTPUT_DIR}/${OUTPUT_NAME}_fasta/${OUTPUT_NAME}.faa"

diamond blastp -q "\${OUTPUT_DIR}/${OUTPUT_NAME}_fasta/${OUTPUT_NAME}.faa" \
 -d /home/students/q.abbas/+proj-q.abbas/Workflow_orthogroups/Mapping_OG/Diamond/target -o "\${OUTPUT_DIR}/hits.tsv" --threads 10 --outfmt 6 --max-target-seqs 1 --evalue 1e-5

python "\${PY_SCRIPTS_DIR}/map_to_orthogroups.py" "\${OUTPUT_DIR}/hits.tsv" "\${core_crops_orthogroups}" "\${OUTPUT_DIR}/Assigned_Orthogroups.tsv"

echo "[STEP 2] Writing matched orthogroup tables at $(date)"
python - <<PY
import pandas as pd

input_fasta = "\${OUTPUT_DIR}/${OUTPUT_NAME}_fasta/${OUTPUT_NAME}.faa"
orthogroups_path = "\${OUTPUT_DIR}/Assigned_Orthogroups.tsv"
out_matched = "\${OUTPUT_DIR}/Orthogroups_matched_to_${OUTPUT_NAME}.tsv"
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

echo "[STEP 3] Generating orthogroup_po matrix at $(date)"
python "\${PY_SCRIPTS_DIR}/protein_orthogroup_po_matrix.py" \
  --input-fasta "\${OUTPUT_DIR}/${OUTPUT_NAME}_fasta/${OUTPUT_NAME}.faa" \
  --orthogroups "\${OUTPUT_DIR}/Assigned_Orthogroups.tsv" \
  --intensities "\${PROTEIN_INTENSITIES}" \
  --tissue-ontology "\${TISSUE_ONTOLOGY}" \
  --metadata "\${METADATA}" \
  --og-annotation-table "\${OG_ANNOTATION_TABLE}" \
  --output "\${OUTPUT_DIR}/protein_orthogroup_po_matrix.csv" \
  2>&1 | tee "\${OUTPUT_DIR}/logs/orthogroup_po_matrix.log"

echo "[STEP 4] Generating heatmaps and orthogroup summary outputs at $(date)"
python "\${PY_SCRIPTS_DIR}/plot_heatmap.py" \
    --name "\${OUTPUT_NAME}" \
    --protein_orthogroup_po_matrix "\${OUTPUT_DIR}/protein_orthogroup_po_matrix.csv" \
    --orthogroups "\${OUTPUT_DIR}/Assigned_Orthogroups.tsv" \
    --input_fasta "\${OUTPUT_DIR}/${OUTPUT_NAME}_fasta/${OUTPUT_NAME}.faa" \
    --metadata "\${METADATA}" \
    --matched_orthogroups "\${OUTPUT_DIR}/Orthogroups_matched_to_${OUTPUT_NAME}.tsv" \
    --output_dir "\${OUTPUT_DIR}" \
    2>&1 | tee "\${OUTPUT_DIR}/logs/plot_heatmap.log"

echo "DONE at $(date)"
EOF

chmod +x "${JOB_SCRIPT}"
sbatch "${JOB_SCRIPT}"
