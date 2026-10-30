# Cross-species crop proteomics framework

This framework maps query proteins independently against every plant family,
connects qualifying DIAMOND hits to OrthoFinder orthogroups, and projects the
family-specific orthogroups onto tissue-resolved mcIBAQ proteomics. It produces
static, per-plant, normalized, species-summary, and interactive heatmaps plus
an auditable query-by-sample matrix.

The biological datasets are not stored here. Users must have the reference
FASTAs, OrthoFinder results, FragPipe-derived mcIBAQ tables, and metadata below.

## Software

- Linux or another Unix-like environment with Bash
- Conda/Mamba and DIAMOND
- Python 3.11, pandas, NumPy, Matplotlib, seaborn, SciPy, Plotly, tqdm, Biopython

```bash
conda env create -f environment.yml
conda activate cross_species_framework_enb
```

## Required inputs

Paths are set in `config.sh`; data need not be copied into this repository.

| Variable | Required content |
|---|---|
| `CORE_CROPS_ORTHOGROUPS` | OrthoFinder `Orthogroups.tsv`; first column `Orthogroup`, remaining headers corresponding to reference FASTAs. `Orthogroups_UnassignedGenes.tsv` must be beside it. |
| `SPECIES_TREE` | Rooted OrthoFinder species tree in Newick format with matching terminal names. |
| `FAMILY_DIAMOND_DB_DIR` | One `<Family>.dmnd` database for every metadata family; punctuation becomes `_`. |
| `PROTEIN_INTENSITIES` | One `<Crop>_mcIBAQ.intensities.tsv` per crop, containing `Protein ID` or `Protein`, optional `Indistinguishable Proteins`, and p-number columns such as `P095093_13`. |
| `FULL_ORTHOGROUP_MATRIX` | Reusable OG × crop–p-number matrix built by `build_full_orthogroup_matrix.sh`. |
| `METADATA` | TSV with `Name`, `FASTA`, `Family`, `Order`, `Class`. `Name` matches mcIBAQ filenames; `FASTA` identifies the reference proteome. |
| `TISSUE_ONTOLOGY` | TSV with `pNumber`, `TissueName`. PO fields may exist but never define sample identity. |
| `PNUMBER_TISSUES` | Fallback TSV with `p-number`, `Tissue`. |
| `PROTEIN_DESCRIPTIONS` | TSV with `species`, `fasta_id`, `description`. |
| `PROBLEM_REPORT` | TSV with `Tissue` p-numbers and `Outlier=1`; a header-only table is valid. |
| `UNIPROT_MAPPING` | TSV with `Entry`, `Protein names`, `Gene Names`, `RefSeq`; a header-only table is valid. |

Reference FASTA IDs, OrthoFinder IDs, mcIBAQ protein IDs, and description IDs
must be consistent. Samples are keyed only as `crop_pNumber`; tissue text is a
display annotation.

## Setup

```bash
git clone <repository-url>
cd cross_species_framework_ENB
cp config.example.sh config.sh
```

Edit `config.sh`. It is ignored by Git.

### Family DIAMOND databases

If these are not already available, reference FASTA filenames must match the
metadata `FASTA` column:

```bash
python scripts/build_family_databases.py \
  --metadata /path/to/all_crops_metadata.tsv \
  --fasta-dir /path/to/reference_fastas \
  --output-dir /path/to/family_databases \
  --threads 8
```

### Reusable full matrix

Build once, and rebuild whenever proteomics tables or OrthoFinder results change:

```bash
bash build_full_orthogroup_matrix.sh
```

The builder prefers the master protein. Indistinguishable proteins are used
only when the master is unmapped and all mapped alternatives agree on one OG.
For each OG/sample it retains the maximum positive mcIBAQ value.

## Run one category

The main entry point is `submit_workflow.sh`:

```bash
bash submit_workflow.sh \
  /path/to/Hormones.faa \
  Hormones \
  /path/to/Results/Hormones
```

The command runs in the current terminal and prints progress as it works. The
output basename must equal the category name. Options include
`--query-cover 50`, `--plant-workers 4`, and `--overwrite`.

Each query/family pair is searched normally with E-value ≤ `1e-5` and the
configured coverage. Pairs lacking a hit are retried using DIAMOND
`--sensitive --masking 0` with the same thresholds. A fallback never replaces
a normal hit.

## Outputs

```text
Results/<Category>/
├── <Category>_fasta.faa
├── <Category>_heatmap_tree_all_tissues_family_OGs_merged.jpg
├── <Category>_heatmap_tree_species.jpg
├── interactive_heatmap_<category>.html
├── query_protein_tissue_mcIBAQ_matrix.tsv
├── other_heatmaps/
│   ├── <Category>_all_plants_tissues_min_max.jpg
│   ├── <Category>_all_plants_tissues_z_score.png
│   └── per_plant/<Plant>.jpg
└── other_files/
    ├── per_family_best_hits.tsv
    ├── family_OGs_merged_query_to_orthogroups.tsv
    ├── run_manifest.tsv
    └── logs/
```

Colors are consistent: light gray `#E3E6E8` means OG absent/unassigned, dark
gray `#AEB4BA` means OG present but mcIBAQ missing/zero, and red represents
positive mcIBAQ. Raw heatmaps display shared `log10(1 + mcIBAQ)` values.
