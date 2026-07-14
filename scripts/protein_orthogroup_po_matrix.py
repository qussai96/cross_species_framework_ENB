#!/usr/bin/env python3
"""
FAST Orthogroup × Species_PO Intensity Matrix
Vectorized, chunked, no Python row loops
"""
import pandas as pd
import numpy as np
import os
import argparse
import logging
import re
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- utils ----------------

def setup_logging(log=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()] + ([logging.FileHandler(log)] if log else [])
    )


def normalize_pnumber(s):
    if pd.isna(s) or s is None:
        return None
    s = str(s).strip()
    s = re.sub(r'_[0-9]+$', '', s)
    s = re.sub(r'\.v\d+$', '', s)
    return s


def parse_interproscan_with_fasta(interproscan_file, input_fasta):
    """
    Parse InterProScan TSV file and extract first valid description for each protein.
    
    The InterProScan output has the actual protein names in "Protein accession" column,
    which should match the protein names extracted from the FASTA file headers.
    
    Returns a dict: protein_name -> description
    
    Priority:
    1. InterPro annotations description - full functional descriptions
    2. Signature description - domain/signature names
    """
    protein_descriptions = {}
    
    if not interproscan_file or not os.path.exists(interproscan_file):
        logging.warning(f"InterProScan file not found or not provided: {interproscan_file}")
        return protein_descriptions
    
    try:
        # Parse InterProScan file with proper header
        df = pd.read_csv(interproscan_file, sep='\t', dtype=str)
        
        logging.info(f"InterProScan file columns: {df.columns.tolist()}")
        
        # Check if we have the expected columns
        if 'Protein accession' not in df.columns:
            logging.warning(f"InterProScan file missing 'Protein accession' column. Available columns: {df.columns.tolist()}")
            return protein_descriptions
        
        # Group by protein name and get first valid description
        for protein_id in df['Protein accession'].unique():
            if pd.isna(protein_id):
                continue
                
            protein_rows = df[df['Protein accession'] == protein_id]
            
            # Priority 1: Try InterPro annotations description
            # This contains full functional descriptions like "Pyridoxal 5'-phosphate synthase"
            if 'InterPro annotations description' in df.columns:
                interpro_descs = protein_rows['InterPro annotations description'].dropna()
                for desc in interpro_descs:
                    # Skip dashes and GO term strings (which contain "GO:")
                    if (desc and str(desc).strip() and str(desc) != '-' and 'GO:' not in str(desc)):
                        protein_descriptions[protein_id] = str(desc).strip()
                        break
            
            # Priority 2: If no InterPro description, try Signature description
            if protein_id not in protein_descriptions and 'Signature description' in df.columns:
                sig_descs = protein_rows['Signature description'].dropna()
                for desc in sig_descs:
                    # Skip dashes and numeric values (coordinates, not descriptions)
                    if (desc and str(desc).strip() and str(desc) != '-' and not str(desc).isdigit()):
                        protein_descriptions[protein_id] = str(desc).strip()
                        break
        
        logging.info(f"Loaded descriptions for {len(protein_descriptions)} proteins from InterProScan")
    
    except Exception as e:
        logging.warning(f"Error parsing InterProScan file: {e}")
        import traceback
        traceback.print_exc()
    
    return protein_descriptions


def parse_interproscan(interproscan_file):
    """
    Legacy function - kept for backward compatibility.
    Use parse_interproscan_with_fasta instead.
    """
    logging.warning("parse_interproscan() is deprecated. Use parse_interproscan_with_fasta() instead.")
    return {}


def _format_orthogroup_with_desc(orthogroup, protein, protein_descriptions):
    """
    Format orthogroup name with protein description.
    Returns "Orthogroup|Description" if description exists, otherwise just "Orthogroup"
    """
    if protein in protein_descriptions:
        desc = protein_descriptions[protein]
        # Truncate long descriptions to avoid excessively wide output
        if len(desc) > 100:
            desc = desc[:100] + "..."
        return f"{orthogroup}|{desc}"
    return orthogroup


# ---------------- fast matrix ----------------

def build_matrix_fast(
    orthogroups,
    intensities_dir,
    tissue_ontology,
    metadata,
    output_raw,
    output_maxlfq,
    protein_descriptions=None,
    og_desc_path=None,
    og_annotation_table=None,
):

    if protein_descriptions is None:
        protein_descriptions = {}
    
    # ---------- metadata ----------
    meta = pd.read_csv(metadata, sep='\t', dtype=str)
    meta['ortho_col'] = meta['FASTA'].str.replace(
        r'\.(faa|fasta|helixer\.faa)$', '', regex=True
    )
    species_map = dict(zip(meta['Name'], meta['ortho_col']))

    # ---------- orthogroups ----------
    logging.info("Loading orthogroups (building orthogroup→description mapping)...")
    og = pd.read_csv(orthogroups, sep='\t')
    
    # ===== STEP 1: Map descriptions to orthogroups (BEFORE any intensity data) =====
    # Prefer precomputed orthogroup-level annotations when available.
    orthogroup_descriptions = {}
    if og_annotation_table and os.path.exists(og_annotation_table):
        logging.info(f"Using orthogroup annotations from table: {og_annotation_table}")
        ann = pd.read_csv(og_annotation_table, sep='\t', dtype=str)
        required_cols = {'orthogroup_id', 'InterPro_domain_description', 'functional_description'}
        if not required_cols.issubset(set(ann.columns)):
            raise ValueError(
                f"Annotation table must contain columns {sorted(required_cols)}; got {ann.columns.tolist()}"
            )

        ann = ann.dropna(subset=['orthogroup_id'])
        # Prefer InterPro_domain_description, fall back to functional_description
        ann['description_to_use'] = ann['InterPro_domain_description'].where(
            (ann['InterPro_domain_description'].notna()) & 
            (ann['InterPro_domain_description'].str.strip() != '') &
            (ann['InterPro_domain_description'] != 'NA'),
            ann['functional_description']
        )
        ann = ann[ann['description_to_use'].notna()]
        ann = ann[ann['description_to_use'].str.strip() != '']
        ann = ann[ann['description_to_use'].str.lower() != 'unknown function']
        ann = ann.drop_duplicates(subset=['orthogroup_id'], keep='first')

        orthogroup_descriptions = {
            og_id: (desc[:100] + '...') if len(desc) > 100 else desc
            for og_id, desc in zip(ann['orthogroup_id'], ann['description_to_use'])
        }
    else:
        for _, row in og.iterrows():
            og_id = row['Orthogroup']
            # Iterate through all columns (all fasta sources) and find first protein with description
            for col in og.columns[1:]:  # skip 'Orthogroup' column
                if pd.notna(row[col]):
                    proteins = [p.strip() for p in str(row[col]).split(',')]
                    for protein in proteins:
                        if protein in protein_descriptions:
                            desc = protein_descriptions[protein]
                            # Truncate long descriptions to avoid excessively wide output
                            if len(desc) > 100:
                                desc = desc[:100] + "..."
                            orthogroup_descriptions[og_id] = desc
                            break
                    if og_id in orthogroup_descriptions:
                        break
    
    logging.info(f"Mapped descriptions to {len(orthogroup_descriptions)} orthogroups")

    if og_desc_path:
        og_desc_df = pd.DataFrame(
            sorted(orthogroup_descriptions.items()),
            columns=["Orthogroup", "Description"],
        )
        og_desc_df.to_csv(og_desc_path, sep='\t', index=False)
        logging.info(f"Wrote orthogroup descriptions to: {og_desc_path}")
    
    # ===== STEP 2: Build protein→orthogroup mapping with descriptions =====
    logging.info("Loading orthogroups (exploding proteins)...")
    og_long = (
        og
        .melt(id_vars='Orthogroup', var_name='species', value_name='protein')
        .dropna(subset=['protein'])
    )

    og_long['protein'] = og_long['protein'].str.split(',')
    og_long = og_long.explode('protein')
    og_long['protein'] = og_long['protein'].str.strip()
    
    # Add descriptions to each row using pre-computed mapping
    og_long['orthogroup_with_desc'] = og_long['Orthogroup'].map(
        lambda og_id: f"{og_id}|{orthogroup_descriptions[og_id]}" if og_id in orthogroup_descriptions else og_id
    )
    
    og_long = og_long[['protein', 'Orthogroup', 'orthogroup_with_desc', 'species']]

    # ---------- tissue ontology ----------
    tissue = pd.read_csv(tissue_ontology, sep='\t', dtype=str)
    tissue['p_norm'] = tissue['pNumber'].apply(normalize_pnumber)
    p_to_po = dict(zip(tissue['p_norm'], tissue['PO_1']))

    def build_final_matrix(intensity_columns, matrix_label):
        chunks = []

        for fname in tqdm(os.listdir(intensities_dir), desc=f"Species ({matrix_label})"):
            if not fname.endswith('.tsv'):
                continue

            species = fname.replace('.proteins.tsv', '').replace('.tsv', '')
            ortho_col = species_map.get(species)
            if not ortho_col:
                continue

            path = os.path.join(intensities_dir, fname)

            for df in pd.read_csv(path, sep='\t', chunksize=200_000, low_memory=False):

                if 'Protein ID' not in df.columns:
                    continue

                # ---- proteins ----
                prot = df[['Protein ID']].copy()
                prot['Protein ID'] = prot['Protein ID'].astype(str)

                if 'Indistinguishable Proteins' in df.columns:
                    extra = df[['Indistinguishable Proteins']].dropna()
                    extra = extra.rename(columns={'Indistinguishable Proteins': 'Protein ID'})
                    prot = pd.concat([prot, extra])

                prot['Protein ID'] = prot['Protein ID'].str.replace(';', ',')
                prot['Protein ID'] = prot['Protein ID'].str.split(',')
                prot = prot.explode('Protein ID')
                prot['Protein ID'] = prot['Protein ID'].str.strip()

                # ---- intensities ----
                int_cols = intensity_columns(df.columns)
                if not int_cols:
                    continue

                vals = df[int_cols].copy()
                vals['row_id'] = vals.index

                prot['row_id'] = prot.index

                merged = (
                    prot
                    .merge(og_long, left_on='Protein ID', right_on='protein', how='inner')
                    .merge(vals, on='row_id', how='left')
                )

                melted = merged.melt(
                    id_vars='orthogroup_with_desc',
                    value_vars=int_cols,
                    var_name='p',
                    value_name='intensity'
                )

                melted['p'] = (
                    melted['p']
                    .str.replace('MaxLFQ Intensity', '', regex=False)
                    .str.replace('Intensity', '', regex=False)
                    .str.strip()
                    .map(normalize_pnumber)
                )

                melted['PO'] = melted['p'].map(p_to_po)
                melted = melted.dropna(subset=['PO', 'intensity'])

                melted['col'] = ortho_col + "_" + melted['PO']
                chunks.append(
                    melted.groupby(['orthogroup_with_desc', 'col'], as_index=False)['intensity'].sum()
                )

        if not chunks:
            logging.warning(f"No intensity values found for {matrix_label} matrix")
            return pd.DataFrame()

        logging.info(f"Final aggregation for {matrix_label} matrix...")
        final = pd.concat(chunks)
        final = final.groupby(['orthogroup_with_desc', 'col'])['intensity'].sum().unstack(fill_value=0)

        if og_desc_path and os.path.exists(og_desc_path):
            og_desc = pd.read_csv(og_desc_path, sep='\t', dtype=str)
            og_keep = set(og_desc['Orthogroup'].dropna())
            if og_keep:
                final = final.loc[[idx for idx in final.index if idx.split('|')[0] in og_keep]]
            else:
                logging.info("No orthogroups listed in description table; skipping matrix row filtering.")

        return final

    raw_final = build_final_matrix(
        lambda cols: [c for c in cols if c.endswith(' Intensity') and 'MaxLFQ Intensity' not in c],
        matrix_label='raw intensities',
    )
    raw_final.to_csv(output_raw, sep='\t')
    logging.info(f"Saved raw intensity matrix to: {output_raw}")

    maxlfq_final = build_final_matrix(
        lambda cols: [c for c in cols if 'MaxLFQ Intensity' in c],
        matrix_label='MaxLFQ intensities',
    )
    maxlfq_final.to_csv(output_maxlfq, sep='\t')
    logging.info(f"Saved MaxLFQ intensity matrix to: {output_maxlfq}")
    logging.info("DONE")


def generate_orthogroup_heatmap(matrix_file, orthogroups_file, input_fasta, output_file,
                                metadata_file, title="Protein Orthogroups Expression Across Plants"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    # Load metadata to map species names
    meta = pd.read_csv(metadata_file, sep='\t', dtype=str)
    meta['ortho_col'] = meta['FASTA'].str.replace(
        r'\.(faa|fasta|helixer\.faa)$', '', regex=True
    )
    ortho_to_name = dict(zip(meta['ortho_col'], meta['Name']))

    # Load matrix
    df = pd.read_csv(matrix_file, sep='\t', index_col=0)

    # Load orthogroups and map protein -> orthogroup
    og = pd.read_csv(orthogroups_file, sep='\t', low_memory=False)
    protein_to_og = {}
    for _, row in og.iterrows():
        og_id = row['Orthogroup']
        for col in og.columns[1:]:
            if pd.notna(row[col]):
                proteins = [p.strip() for p in str(row[col]).split(',')]
                for p in proteins:
                    protein_to_og[p] = og_id

    # Extract proteins from input FASTA
    proteins_in_fasta = set()
    with open(input_fasta, 'r') as f:
        for line in f:
            if line.startswith('>'):
                proteins_in_fasta.add(line[1:].strip().split()[0])

    # Keep only orthogroups with proteins from FASTA
    ogs_to_keep = set(protein_to_og[p] for p in proteins_in_fasta if p in protein_to_og)
    
    # Filter matrix rows - the matrix index may have "Orthogroup|Description" format
    # Extract the base orthogroup ID from each row for comparison
    df_filtered_indices = []
    for idx in df.index:
        base_og = idx.split('|')[0] if '|' in idx else idx
        if base_og in ogs_to_keep:
            df_filtered_indices.append(idx)
    
    df_filtered = df.loc[df_filtered_indices]
    if df_filtered.empty:
        print("No orthogroups found for input FASTA proteins!")
        return

    # Collapse PO/tissue columns to median per plant
    species_cols = {}
    for col in df_filtered.columns:
        species = col.split('_PO:')[0]
        species_cols.setdefault(species, []).append(col)

    # Initialize median_df with plants as rows and orthogroups as columns
    median_df = pd.DataFrame(index=species_cols.keys(), columns=df_filtered.index, dtype=float)

    for sp, cols in species_cols.items():
        # Calculate median, but keep NaN if all values are 0 or NaN
        sp_data = df_filtered[cols]
        # Median treating 0 as valid, but result is NaN if no valid data
        sp_median = sp_data.replace(0, np.nan).median(axis=1)
        # If median is NaN but original data had zeros, use 0
        has_zeros = (sp_data == 0).any(axis=1)
        sp_median = sp_median.where(~(sp_median.isna() & has_zeros), 0)
        median_df.loc[sp] = sp_median.values

    # Map species names from metadata
    median_df.index = median_df.index.map(lambda x: ortho_to_name.get(x, x))
    
    # Drop species with all NaN
    median_df = median_df.dropna(how='all')

    if median_df.empty:
        print("No data left after collapsing to per-plant medians.")
        return

    # Save raw intensities before normalization (orthogroups as rows, species as columns)
    raw_output_file = output_file.replace('.png', '_raw_intensities.tsv')
    median_df.T.to_csv(raw_output_file, sep='\t')
    print(f"Saved raw intensities (before normalization) to: {raw_output_file}")

    # --------- MIN-MAX SCALE PER COLUMN (orthogroup) ---------
    # Scale only non-NaN values per column
    scaled_df = median_df.copy()
    for col in scaled_df.columns:
        col_data = scaled_df[col]
        valid_data = col_data.dropna()
        if len(valid_data) > 0:
            col_min = valid_data.min()
            col_max = valid_data.max()
            if col_max > col_min:
                scaled_df[col] = (col_data - col_min) / (col_max - col_min)
            else:
                scaled_df[col] = 0  # constant column
    # NaN values remain as NaN
    # ---------------------------------------------------------

    # Custom colormap: white for NaN, yellow to red for 0-1
    colors = ['#FFFF00', '#FF0000']  # yellow to red
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('yellow_red', colors, N=n_bins)
    cmap.set_bad(color='white')  # NaN = white

    if scaled_df.shape[0] < 2:
        print(f"Not enough plants to cluster (rows={scaled_df.shape[0]}). Saving simple heatmap.")
        plt.figure(figsize=(max(4, 0.3 * scaled_df.shape[1]), max(4, 0.5 * scaled_df.shape[0])))
        sns.heatmap(scaled_df, cmap=cmap, xticklabels=False, yticklabels=True, vmin=0, vmax=1)
        plt.title(f"{title}\n(min-max scaled per orthogroup, yellow-red=0-1)")
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        return

    max_width = 20  # in inches
    width = min(max_width, 0.15 * scaled_df.shape[1])
    height = max(6, 0.3 * scaled_df.shape[0])

    g = sns.clustermap(
        scaled_df,
        row_cluster=True,
        col_cluster=False,
        cmap=cmap,
        method='ward',
        metric='euclidean',
        figsize=(width, height),
        yticklabels=True,
        xticklabels=False,
        vmin=0,
        vmax=1
    )

    g.fig.suptitle(
        f"{title}\n({scaled_df.shape[0]} Plants, {scaled_df.shape[1]} Orthogroups)\n(min-max scaled per orthogroup, yellow-red=0-1)",
        fontsize=14, fontweight='bold'
    )
    g.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def generate_orthogroup_z_score_heatmap(matrix_file, orthogroups_file, input_fasta, output_file,
                                metadata_file, title="Protein Orthogroups Expression Across Plants",
                                min_species_presence=2):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import zscore
    from matplotlib.colors import LinearSegmentedColormap

    # Load metadata to map species names
    meta = pd.read_csv(metadata_file, sep='\t', dtype=str)
    meta['ortho_col'] = meta['FASTA'].str.replace(
        r'\.(faa|fasta|helixer\.faa)$', '', regex=True
    )
    ortho_to_name = dict(zip(meta['ortho_col'], meta['Name']))

    # Load matrix (rows = orthogroups, cols = samples/tissues)
    df = pd.read_csv(matrix_file, sep="\t", index_col=0)

    # Load orthogroups and map protein -> orthogroup
    og = pd.read_csv(orthogroups_file, sep="\t", low_memory=False)
    protein_to_og = {}
    for _, row in og.iterrows():
        og_id = row["Orthogroup"]
        for col in og.columns[1:]:
            if pd.notna(row[col]):
                proteins = [p.strip() for p in str(row[col]).split(",")]
                for p in proteins:
                    protein_to_og[p] = og_id

    # Extract proteins from input FASTA
    proteins_in_fasta = set()
    with open(input_fasta, "r") as f:
        for line in f:
            if line.startswith(">"):
                proteins_in_fasta.add(line[1:].strip().split()[0])

    # Keep only orthogroups with proteins from FASTA
    ogs_to_keep = {protein_to_og[p] for p in proteins_in_fasta if p in protein_to_og}
    
    # Filter matrix rows - the matrix index may have "Orthogroup|Description" format
    # Extract the base orthogroup ID from each row for comparison
    df_filtered_indices = []
    for idx in df.index:
        base_og = idx.split('|')[0] if '|' in idx else idx
        if base_og in ogs_to_keep:
            df_filtered_indices.append(idx)
    
    df_filtered = df.loc[df_filtered_indices]
    if df_filtered.empty:
        print("No orthogroups found for input FASTA proteins!")
        return

    # Collapse PO/tissue columns to median per species
    species_cols = {}
    for col in df_filtered.columns:
        species = col.split("_PO:")[0]
        species_cols.setdefault(species, []).append(col)

    # species_intensity: rows = orthogroups, cols = species
    species_intensity = pd.DataFrame(index=df_filtered.index, columns=species_cols.keys(), dtype=float)

    for sp, cols in species_cols.items():
        # Calculate median, preserving distinction between NaN (no proteins) and 0 (protein exists, zero intensity)
        sp_data = df_filtered[cols]
        sp_median = sp_data.replace(0, np.nan).median(axis=1)
        has_zeros = (sp_data == 0).any(axis=1)
        sp_median = sp_median.where(~(sp_median.isna() & has_zeros), 0)
        species_intensity[sp] = sp_median.values

    # Map species names from metadata
    species_intensity.columns = species_intensity.columns.map(lambda x: ortho_to_name.get(x, x))
    
    # Remove empty species (all-NA)
    species_intensity = species_intensity.dropna(axis=1, how="all")

    if species_intensity.empty:
        print("No data left after collapsing to per-species medians.")
        return

    # --------- SAME LOGIC AS YOUR PIPELINE ---------
    # 1) Filter orthogroups present in at least N species (count non-NaN, non-zero)
    present_count = (species_intensity > 0).sum(axis=1)
    species_intensity_filtered = species_intensity.loc[present_count >= min_species_presence]
    if species_intensity_filtered.empty:
        print(f"No orthogroups present in ≥{min_species_presence} species.")
        return

    # 2) log2 transform (only for non-NaN values)
    species_log = species_intensity_filtered.copy()
    mask = species_log.notna()
    species_log[mask] = np.log2(species_log[mask] + 1)

    # 3) z-score per orthogroup (row-wise; each orthogroup across species)
    species_z = species_log.apply(
        lambda x: zscore(x, nan_policy="omit"),
        axis=1,
        result_type="broadcast"
    )
    species_z = species_z.replace([np.inf, -np.inf], 0)
    # NaN remains NaN (no proteins)

    # Plot data: species as rows, orthogroups as columns
    plot_data = species_z.T
    # ------------------------------------------------

    # Custom colormap: white for NaN, yellow-red for positive z-scores, blue for negative
    # For z-scores, we'll use blue-white-yellow-red
    colors = ['#0000FF', '#FFFFFF', '#FFFF00', '#FF0000']  # blue-white-yellow-red
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('diverging_yellow_red', colors, N=n_bins)
    cmap.set_bad(color='white')  # NaN = white

    # Handle too few species for clustering
    if plot_data.shape[0] < 2:
        print(f"Not enough plants to cluster (rows={plot_data.shape[0]}). Saving simple heatmap.")
        plt.figure(figsize=(max(4, 0.3 * plot_data.shape[1]), max(4, 0.5 * plot_data.shape[0])))
        sns.heatmap(plot_data, cmap=cmap, center=0, xticklabels=False, yticklabels=True,
                    cbar_kws={"label": "Z-score (log2 intensity), white=no proteins"})
        plt.title(f"{title}\n({plot_data.shape[0]} Plants, {plot_data.shape[1]} Orthogroups)")
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        return

    # Auto sizing
    max_width = 20
    width = min(max_width, 0.5 * plot_data.shape[1])
    height = max(6, 0.35 * plot_data.shape[0])

    g = sns.clustermap(
        plot_data,
        method="ward",
        metric="euclidean",
        cmap=cmap,
        center=0,
        figsize=(width, height),
        row_cluster=True,   # cluster species
        col_cluster=True,   # cluster orthogroups
        xticklabels=False,
        yticklabels=True,
        cbar_kws={"label": "Z-score per orthogroup (log2 intensity), white=no proteins"},
        dendrogram_ratio=(0.15, 0.15)
    )

    g.fig.suptitle(
        f"{title}\n({plot_data.shape[1]} Orthogroups, {plot_data.shape[0]} Plants)",
        y=1.02, fontsize=14, fontweight="bold"
    )
    g.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()



# ---------------- CLI ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-fasta', required=True)
    p.add_argument('--orthogroups', required=True)
    p.add_argument('--intensities', required=True)
    p.add_argument('--tissue-ontology', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--interproscan', required=False, default=None, help='Path to InterProScan TSV file')
    p.add_argument('--og-annotation-table', required=False, default=None,
                   help='Path to orthogroup annotation table with orthogroup_id and functional_description columns')
    p.add_argument('--output', required=True)
    p.add_argument('--log')
    args = p.parse_args()

    setup_logging(args.log)
    
    # Parse interproscan file if provided
    protein_descriptions = {}
    if args.interproscan:
        protein_descriptions = parse_interproscan_with_fasta(args.interproscan, args.input_fasta)
    
    output_dir = Path(args.output).parent
    raw_output = output_dir / "protein_orthogroup_po_matrix_raw_intensities.csv"
    maxlfq_output = output_dir / "protein_orthogroup_po_matrix_maxLFQ_intensities.csv"
    og_desc_path = output_dir / "OG_Desc.tsv"

    build_matrix_fast(
        args.orthogroups,
        args.intensities,
        args.tissue_ontology,
        args.metadata,
        str(raw_output),
        str(maxlfq_output),
        protein_descriptions=protein_descriptions,
        og_desc_path=str(og_desc_path),
        og_annotation_table=args.og_annotation_table,
    )

    # Generate statistics
    logging.info("Generating statistics...")
    
    # Count proteins in input FASTA
    proteins_in_fasta = set()
    with open(args.input_fasta, 'r') as f:
        for line in f:
            if line.startswith('>'):
                proteins_in_fasta.add(line[1:].strip().split()[0])
    num_proteins = len(proteins_in_fasta)
    
    # Load orthogroups and map protein -> orthogroup
    og = pd.read_csv(args.orthogroups, sep='\t')
    protein_to_og = {}
    for _, row in og.iterrows():
        og_id = row['Orthogroup']
        for col in og.columns[1:]:
            if pd.notna(row[col]):
                proteins = [p.strip() for p in str(row[col]).split(',')]
                for p in proteins:
                    protein_to_og[p] = og_id
    
    # Find orthogroups containing input proteins
    ogs_with_input_proteins = set(protein_to_og[p] for p in proteins_in_fasta if p in protein_to_og)
    
    # Load matrix to count orthogroups with values
    matrix_df = pd.read_csv(maxlfq_output, sep='\t', index_col=0)
    
    # Get the species name from input FASTA filename
    fasta_basename = Path(args.input_fasta).stem
    
    # Find columns matching this species in the matrix
    species_columns = [col for col in matrix_df.columns if col.startswith(fasta_basename)]
    
    if species_columns:
        # Count orthogroups with non-zero values in any of the species columns
        species_data = matrix_df[species_columns]
        ogs_with_values = (species_data.sum(axis=1) > 0).sum()
    else:
        ogs_with_values = 0
        logging.warning(f"No columns found for species '{fasta_basename}' in the matrix")
    
    # Write statistics to file
    stats_file = output_dir / "stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"=== Statistics ===\n\n")
        f.write(f"Number of proteins in input FASTA: {num_proteins}\n")
        f.write(f"Number of orthogroups containing input proteins: {len(ogs_with_input_proteins)}\n")
        f.write(f"Number of orthogroups with values in '{fasta_basename}' columns: {ogs_with_values}\n")
    
    logging.info(f"Statistics written to {stats_file}")

    # generate_orthogroup_heatmap(
    #     matrix_file=args.output,
    #     orthogroups_file=args.orthogroups,
    #     input_fasta=args.input_fasta,
    #     output_file=str(Path(args.output).parent / "heatmap.png"),
    #     metadata_file=args.metadata
    # )

    
    # generate_orthogroup_z_score_heatmap(
    #     matrix_file=args.output,
    #     orthogroups_file=args.orthogroups,
    #     input_fasta=args.input_fasta,
    #     output_file=str(Path(args.output).parent / "heatmap_z_score.png"),
    #     metadata_file=args.metadata
    # )




if __name__ == "__main__":
    main()
