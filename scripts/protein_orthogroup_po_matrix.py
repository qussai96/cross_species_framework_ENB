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

LEGACY_LFQ_RE = re.compile(r"max\s*lfq", re.IGNORECASE)

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


def normalize_label_part(value):
    if pd.isna(value) or value is None:
        return "unknown_tissue"
    text = str(value).strip()
    if not text:
        return "unknown_tissue"
    text = re.sub(r'[^A-Za-z0-9:]+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text or "unknown_tissue"


def strip_legacy_lfq_token(value):
    text = str(value)
    return LEGACY_LFQ_RE.sub("", text)


def is_legacy_lfq_intensity_col(value):
    text = str(value)
    return text.endswith(' Intensity') and bool(LEGACY_LFQ_RE.search(text))


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


def _clean_protein_ids(values):
    cleaned = []
    if values is None:
        return cleaned

    for value in values:
        if value is None:
            continue
        token = str(value).strip()
        if not token or token.lower() in {'none', 'nan', 'na'}:
            continue
        cleaned.extend([part.strip() for part in token.replace(';', ',').split(',') if part.strip()])
    return cleaned


def extract_primary_protein_ids(df):
    protein_col = None
    if 'Protein ID' in df.columns:
        protein_col = 'Protein ID'
    elif 'Protein' in df.columns:
        protein_col = 'Protein'

    if protein_col is None:
        return []

    primary = df[[protein_col]].copy().rename(columns={protein_col: 'Protein ID'})
    primary['Protein ID'] = primary['Protein ID'].replace({pd.NA: None, np.nan: None})
    primary = primary.dropna(subset=['Protein ID'])
    return _clean_protein_ids(primary['Protein ID'].tolist())


def extract_indistinguishable_protein_ids(df):
    if 'Indistinguishable Proteins' not in df.columns:
        return []

    extra = df[['Indistinguishable Proteins']].dropna()
    extra = extra.rename(columns={'Indistinguishable Proteins': 'Protein ID'})
    extra['Protein ID'] = extra['Protein ID'].replace({pd.NA: None, np.nan: None})
    extra = extra.dropna(subset=['Protein ID'])
    return _clean_protein_ids(extra['Protein ID'].tolist())


def extract_protein_ids(df):
    """Extract protein identifiers from the main protein column and the fallback indistinguishable-protein column."""
    protein_ids = extract_primary_protein_ids(df) + extract_indistinguishable_protein_ids(df)

    # Keep only non-empty identifiers and preserve order while avoiding duplicates.
    seen = set()
    cleaned = []
    for protein_id in protein_ids:
        if protein_id in seen:
            continue
        seen.add(protein_id)
        cleaned.append(protein_id)
    return cleaned


def get_fallback_protein_ids(df, known_proteins):
    primary_ids = set(extract_primary_protein_ids(df))
    fallback_ids = extract_indistinguishable_protein_ids(df)
    if not fallback_ids:
        return []

    if primary_ids and any(pid in known_proteins for pid in primary_ids):
        return []

    return [protein_id for protein_id in fallback_ids if protein_id in known_proteins]


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


def write_search_protein_expanded_matrix(
    matrix_path,
    orthogroups_path,
    output_path,
):
    matrix_df = pd.read_csv(matrix_path, sep='\t', dtype={0: str})
    if matrix_df.empty:
        pd.DataFrame(columns=['search_protein_id', 'orthogroup_with_desc']).to_csv(output_path, sep='\t', index=False)
        logging.info(f"Expanded search-protein matrix is empty; wrote placeholder to: {output_path}")
        return 0

    row_col = matrix_df.columns[0]
    matrix_df['Orthogroup'] = matrix_df[row_col].astype(str).str.split('|', n=1).str[0]

    orth_df = pd.read_csv(orthogroups_path, sep='\t', dtype=str).fillna('')
    if 'Orthogroup' not in orth_df.columns or 'search_protein_id' not in orth_df.columns:
        raise ValueError("Assigned orthogroups file must contain 'Orthogroup' and 'search_protein_id' columns")

    expanded_rows = []
    for _, row in orth_df[['Orthogroup', 'search_protein_id']].drop_duplicates().iterrows():
        orthogroup = str(row['Orthogroup']).strip()
        if not orthogroup:
            continue
        search_ids = [token.strip() for token in str(row['search_protein_id']).split(',') if token.strip()]
        for search_id in search_ids:
            expanded_rows.append({'Orthogroup': orthogroup, 'search_protein_id': search_id})

    expanded_df = pd.DataFrame(expanded_rows)
    if expanded_df.empty:
        pd.DataFrame(columns=['search_protein_id', 'orthogroup_with_desc']).to_csv(output_path, sep='\t', index=False)
        logging.info(f"No search proteins available for matrix expansion; wrote placeholder to: {output_path}")
        return 0

    merged = expanded_df.merge(matrix_df, on='Orthogroup', how='inner')
    tissue_cols = [c for c in matrix_df.columns if c not in {row_col, 'Orthogroup'}]
    merged = merged[['search_protein_id', row_col] + tissue_cols].rename(columns={row_col: 'orthogroup_with_desc'})
    merged.to_csv(output_path, sep='\t', index=False)
    logging.info(f"Saved search-protein expanded matrix to: {output_path}")
    return len(merged)


# ---------------- fast matrix ----------------

def build_matrix_fast(
    orthogroups,
    intensities_dir,
    tissue_ontology,
    metadata,
    output_raw,
    ibaq_dir=None,
    output_ibaq=None,
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
    tissue['PO_norm'] = tissue['PO_1'].fillna('PO:0000000').replace('', 'PO:0000000')
    tissue['TissueName_norm'] = tissue['TissueName'].fillna('').replace('', 'unknown_tissue').map(normalize_label_part)
    tissue['po_tissue_label'] = tissue['PO_norm'].astype(str) + '_' + tissue['TissueName_norm'].astype(str)
    p_to_label = dict(zip(tissue['p_norm'], tissue['po_tissue_label']))

    def build_final_matrix(
        intensity_columns,
        matrix_label,
        source_dir=None,
        file_filter=None,
        species_from_filename=None,
        p_column_to_pnumber=None,
    ):
        chunks = []
        all_mapped_cols = set()

        if source_dir is None:
            source_dir = intensities_dir

        if p_column_to_pnumber is None:
            def p_column_to_pnumber(value):
                return (
                    strip_legacy_lfq_token(str(value))
                    .replace('Intensity', '')
                    .strip()
                )

        for fname in tqdm(os.listdir(source_dir), desc=f"Species ({matrix_label})"):
            if file_filter is not None:
                if not file_filter(fname):
                    continue
            elif not fname.endswith('.tsv'):
                continue

            if species_from_filename is not None:
                species = species_from_filename(fname)
            else:
                species = fname.replace('.proteins.tsv', '').replace('.tsv', '')

            ortho_col = species_map.get(species)
            if not ortho_col:
                continue

            path = os.path.join(source_dir, fname)

            # Record all valid tissue columns for this species from file headers,
            # even if no selected proteins map to them in this query run.
            try:
                header_df = pd.read_csv(path, sep='\t', nrows=0)
                header_int_cols = intensity_columns(header_df.columns)
                for hcol in header_int_cols:
                    pnum = normalize_pnumber(p_column_to_pnumber(hcol))
                    po_label = p_to_label.get(pnum)
                    if po_label:
                        all_mapped_cols.add(f"{ortho_col}_{po_label}")
            except Exception as exc:
                logging.warning(f"Could not read headers for {path}: {exc}")

            for df in pd.read_csv(path, sep='\t', chunksize=200_000, low_memory=False):

                protein_col = None
                if 'Protein ID' in df.columns:
                    protein_col = 'Protein ID'
                elif 'Protein' in df.columns:
                    protein_col = 'Protein'

                if protein_col is None:
                    continue

                # ---- proteins ----
                prot = pd.DataFrame({'Protein ID': extract_protein_ids(df)})
                if prot.empty:
                    continue

                known_proteins = set(og_long['protein'].dropna().astype(str).tolist())
                fallback_ids = get_fallback_protein_ids(df, known_proteins)
                if fallback_ids:
                    logging.info(
                        "Indistinguishable-protein fallback matched orthogroups for species=%s file=%s: %s",
                        species,
                        fname,
                        ", ".join(fallback_ids),
                    )

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

                melted['p'] = melted['p'].map(p_column_to_pnumber).map(normalize_pnumber)

                melted['po_tissue_label'] = melted['p'].map(p_to_label)
                melted = melted.dropna(subset=['po_tissue_label', 'intensity'])

                melted['col'] = ortho_col + "_" + melted['po_tissue_label']
                chunks.append(
                    melted.groupby(['orthogroup_with_desc', 'col'], as_index=False)['intensity'].mean()
                )

        if not chunks:
            logging.warning(f"No intensity values found for {matrix_label} matrix")
            return pd.DataFrame()

        logging.info(f"Final aggregation for {matrix_label} matrix...")
        final = pd.concat(chunks)
        final = final.groupby(['orthogroup_with_desc', 'col'])['intensity'].mean().unstack(fill_value=0)

        # Keep all mapped tissue columns (including all-zero undetected columns)
        # so downstream "all_tissues" heatmaps represent the full tissue universe.
        if all_mapped_cols:
            ordered_cols = sorted(all_mapped_cols)
            final = final.reindex(columns=ordered_cols, fill_value=0)

        if og_desc_path and os.path.exists(og_desc_path):
            og_desc = pd.read_csv(og_desc_path, sep='\t', dtype=str)
            og_keep = set(og_desc['Orthogroup'].dropna())
            if og_keep:
                final = final.loc[[idx for idx in final.index if idx.split('|')[0] in og_keep]]
            else:
                logging.info("No orthogroups listed in description table; skipping matrix row filtering.")

        return final

    raw_final = build_final_matrix(
        lambda cols: [c for c in cols if str(c).endswith(' Intensity') and not is_legacy_lfq_intensity_col(c)],
        matrix_label='raw intensities',
    )
    raw_final.to_csv(output_raw, sep='\t')
    logging.info(f"Saved raw intensity matrix to: {output_raw}")

    if ibaq_dir and output_ibaq:
        def _ibaq_species_from_filename(name):
            return re.sub(r'_mcIBAQ\.intensities\.tsv$', '', name)

        def _ibaq_file_filter(name):
            return name.endswith('_mcIBAQ.intensities.tsv')

        def _ibaq_intensity_columns(cols):
            out = []
            for c in cols:
                c_str = str(c).strip()
                if c_str in {'Protein', 'Protein ID', 'Indistinguishable Proteins'}:
                    continue
                if normalize_pnumber(c_str) in p_to_label:
                    out.append(c)
            return out

        ibaq_final = build_final_matrix(
            _ibaq_intensity_columns,
            matrix_label='mcIBAQ intensities',
            source_dir=ibaq_dir,
            file_filter=_ibaq_file_filter,
            species_from_filename=_ibaq_species_from_filename,
            p_column_to_pnumber=lambda x: str(x).strip(),
        )
        ibaq_final.to_csv(output_ibaq, sep='\t')
        logging.info(f"Saved mcIBAQ intensity matrix to: {output_ibaq}")

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
        species = col.split('_PO:', 1)[0] if '_PO:' in col else col
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
        species = col.split("_PO:", 1)[0] if "_PO:" in col else col
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
    
    requested_output = Path(args.output)
    output_dir = requested_output.parent
    raw_output = output_dir / "protein_orthogroup_po_matrix_raw_intensities.csv"
    og_desc_path = output_dir / "OG_Desc.tsv"
    expanded_output = output_dir / f"search_protein_{requested_output.name}"

    def _has_mcibaq_files(directory):
        try:
            return any(name.endswith('_mcIBAQ.intensities.tsv') for name in os.listdir(directory))
        except OSError:
            return False

    mcibaq_mode = (
        requested_output.name.endswith('mcIBAQ_intensities.csv')
        or _has_mcibaq_files(args.intensities)
    )

    matrix_for_stats = str(requested_output)

    if mcibaq_mode:
        build_matrix_fast(
            args.orthogroups,
            args.intensities,
            args.tissue_ontology,
            args.metadata,
            str(raw_output),
            ibaq_dir=args.intensities,
            output_ibaq=str(requested_output),
            protein_descriptions=protein_descriptions,
            og_desc_path=str(og_desc_path),
            og_annotation_table=args.og_annotation_table,
        )
    else:
        build_matrix_fast(
            args.orthogroups,
            args.intensities,
            args.tissue_ontology,
            args.metadata,
            str(raw_output),
            ibaq_dir=args.intensities,
            output_ibaq=str(requested_output),
            protein_descriptions=protein_descriptions,
            og_desc_path=str(og_desc_path),
            og_annotation_table=args.og_annotation_table,
        )

    expanded_row_count = write_search_protein_expanded_matrix(
        matrix_path=matrix_for_stats,
        orthogroups_path=args.orthogroups,
        output_path=expanded_output,
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
    if not Path(matrix_for_stats).exists():
        logging.warning(f"Expected matrix not found for stats: {matrix_for_stats}. Falling back to {raw_output}")
        matrix_for_stats = str(raw_output)

    try:
        matrix_df = pd.read_csv(matrix_for_stats, sep='\t', index_col=0)
    except pd.errors.EmptyDataError:
        logging.warning(f"Matrix file is empty for stats: {matrix_for_stats}")
        matrix_df = pd.DataFrame()
    
    tissue_cols = list(matrix_df.columns[1:]) if not matrix_df.empty else []
    nonzero_cols = []
    if tissue_cols:
        numeric_matrix = matrix_df[tissue_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        nonzero_mask = (numeric_matrix > 0).any(axis=0)
        nonzero_cols = [col for col, keep in nonzero_mask.items() if keep]

    detected_species = sorted({col.split('_PO:', 1)[0] for col in nonzero_cols if '_PO:' in col})
    ogs_with_values = int(((matrix_df[tissue_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).sum(axis=1) > 0).sum())) if tissue_cols else 0
    
    # Write statistics to file
    stats_file = output_dir / "stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"=== Statistics ===\n\n")
        f.write(f"Number of proteins in input FASTA: {num_proteins}\n")
        f.write(f"Number of orthogroups containing input proteins: {len(ogs_with_input_proteins)}\n")
        f.write(f"Number of orthogroup rows in matrix: {len(matrix_df)}\n")
        f.write(f"Number of search-protein rows in expanded matrix: {expanded_row_count}\n")
        f.write(f"Number of orthogroup rows with any values: {ogs_with_values}\n")
        f.write(f"Number of detected tissue columns: {len(nonzero_cols)}\n")
        f.write(f"Number of detected species: {len(detected_species)}\n")
    
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
