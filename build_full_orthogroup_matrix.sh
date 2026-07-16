#!/bin/bash
#SBATCH --job-name=build_full_orthogroup_matrix
#SBATCH --output=build_full_orthogroup_matrix.%j.out
#SBATCH --error=build_full_orthogroup_matrix.%j.err
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G


set -e

core_crops_orthogroups="/home/students/q.abbas/Scripts/40-OrthoFinder/ENB/Final/Results_Dec28/Orthogroups/Orthogroups.tsv"
PROTEIN_INTENSITIES="/home/students/q.abbas/Fetch_data/Proteomics/combined_proteins/"
IBAQ_INTENSITIES="/home/students/q.abbas/+proj-q.abbas/Workflow_orthogroups/cross_species_framework_ENB/Combined_ibaq_intensities"
METADATA="/home/students/q.abbas/Fetch_data/all_crops_metadata.tsv"
TISSUE_ONTOLOGY="/home/students/q.abbas/+proj-q.abbas/Workflow_orthogroups/ENB_TissueOntology_long.tsv"
OG_ANNOTATION_TABLE="/home/students/q.abbas/+proj-q.abbas/Workflow_orthogroups/Mapping_OG/OG_functional_annotation.tsv"



python /home/students/q.abbas/+proj-q.abbas/Workflow_orthogroups/cross_species_framework_ENB/scripts/build_full_orthogroup_matrix.py \
 --orthogroups "${core_crops_orthogroups}" \
 --intensities "${PROTEIN_INTENSITIES}" \
 --ibaq-intensities "${IBAQ_INTENSITIES}" \
 --tissue-ontology "${TISSUE_ONTOLOGY}" \
 --metadata "${METADATA}" \
 --output-dir .