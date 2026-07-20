


cd /home/students/q.abbas/+proj-q.abbas/Workflow_orthogroups/cross_species_framework_ENB \
 && for d in Results/*; do if [ -d "$d/heatmaps" ]; then category=$(basename "$d"); input="$d/files/protein_orthogroup_po_matrix_mcIBAQ_intensities.csv"; matched="$d/files/Orthogroups_matched_to_${category}.tsv"; \ 
 if [ -f "$input" ] && [ -f "$matched" ]; then echo "=== regenerating $category ==="; rm -f "$d/heatmaps/heatmap_log10_intensity_all_tissues.jpg" "$d/heatmaps/heatmap_minmax_per_row_all_tissues.jpg" "$d/heatmaps/heatmap_log10_minmax_per_row_all_tissues.jpg" "$d/heatmaps/heatmap_zscore_per_row_all_tissues.jpg"; python3 scripts/generate_pathway_heatmaps.py --input "$input" --matched-tsv "$matched" --metadata all_crops_metadata.tsv --output-dir "$d/heatmaps"; else echo "SKIP $category: missing input or matched TSV"; fi; fi; done