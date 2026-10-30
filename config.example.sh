#!/bin/bash
# Copy this file to config.sh and replace every /path/to/... value.

DATA_DIR="/path/to/cross_species_data"
FAMILY_DIAMOND_DB_DIR="${DATA_DIR}/family_databases"

ORTHOFINDER_DIR="${DATA_DIR}/orthofinder_results"
CORE_CROPS_ORTHOGROUPS="${ORTHOFINDER_DIR}/Orthogroups/Orthogroups.tsv"
SPECIES_TREE="${ORTHOFINDER_DIR}/Species_Tree/SpeciesTree_rooted.txt"

PROTEIN_INTENSITIES="${DATA_DIR}/Combined_mcibaq_intensities"
MATRIX_OUTPUT_DIR="${DATA_DIR}/matrix"
FULL_ORTHOGROUP_MATRIX="${MATRIX_OUTPUT_DIR}/protein_orthogroup_pnumber_matrix_mcIBAQ_intensities.tsv"

METADATA="${DATA_DIR}/all_crops_metadata.tsv"
TISSUE_ONTOLOGY="${DATA_DIR}/ENB_TissueOntology_long.tsv"
PNUMBER_TISSUES="${DATA_DIR}/Plant_tissue_p-number.tsv"
PROTEIN_DESCRIPTIONS="${DATA_DIR}/all_species.fasta_id_to_description.tsv"
PROBLEM_REPORT="${DATA_DIR}/outliers.tsv"
UNIPROT_MAPPING="${DATA_DIR}/uniprot_mapped_Ids.tsv"
