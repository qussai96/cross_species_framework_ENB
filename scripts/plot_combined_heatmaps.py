"""
Cross-species heatmaps: Hormones + Cell Surface Receptors
Tissues: Carpel | Leaves | Fruit Flesh
Crops: Apple, Pear, Quince, Sweet Cherry, Sour Cherry, Apricot, Peach, Plum
Normalizations: log10(x+1) | row z-score | min-max (excl. zeros)
Y-axis: search_protein_id from annotation TSVs, sorted alphabetically
X-axis: crops hierarchically clustered per tissue
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path("/home/students/q.abbas/+proj-q.abbas/Workflow_orthogroups/"
            "cross_species_framework_ENB/Hormones_Julien_list")
CSR_CSV   = BASE / "Cell_surface_receptors/protein_orthogroup_po_matrix_mcIBAQ_intensities.csv"
HORM_CSV  = BASE / "Hormones_list/protein_orthogroup_po_matrix_mcIBAQ_intensities.csv"
CSR_ANN   = BASE / "Cell_surface_receptors/Orthogroups_matched_to_Cell_surface_receptors.tsv"
HORM_ANN  = BASE / "Hormones_list/Orthogroups_matched_to_Hormones_list.tsv"
OUTDIR    = BASE / "heatmaps_combined"
OUTDIR.mkdir(exist_ok=True)

# ── Crop → column prefix mapping ──────────────────────────────────────────────
CROP_PREFIXES = {
    "Apple":        "Mdomesticus_Gala_haploid_v2.pep.fa",
    "Pear":         "PyrusCommunis_BartlettDHv2.0.pep",
    "Quince":       "GCA_015708375.1_Cydonia_oblonga",
    "SweetCherry":  "GCF_002207925.1_Prunus_avium",
    "SourCherry":   "Pcer_mont_QTL_update2_proteins",
    "Apricot":      "GCA_903112645.1_Prunus_armeniaca",
    "Peach":        "Ppersica_Lovell_2D_v3.0.proteins.fa",
    "Plum":         "P.domestica_Genome-Dardick_v1.0.proteins",
}

CROP_DISPLAY = {
    "Apple":       "Apple",
    "Pear":        "Pear",
    "Quince":      "Quince",
    "SweetCherry": "Sweet Cherry",
    "SourCherry":  "Sour Cherry",
    "Apricot":     "Apricot",
    "Peach":       "Peach",
    "Plum":        "Plum",
}

# ── Tissue keyword matching (applied to the label AFTER the PO:XXXXXXX_ part) ─
TISSUE_KEYWORDS = {
    "Carpel": [
        r"carpel",
        r"pistil",
    ],
    "Leaves": [
        r"(?<![_\w])leaf(?![_\w])",   # leaf not surrounded by word/underscore chars
        r"leaves",
    ],
    "FruitFlesh": [
        r"flesh",
        r"mesocarp",
    ],
}

TISSUE_DISPLAY = {
    "Carpel":     "Carpel (PO:0009046)",
    "Leaves":     "Leaves (PO:0006340)",
    "FruitFlesh": "Fruit Flesh (PO:0030110)",
}


def tissue_label(col: str):
    """Return the text portion after 'PO:XXXXXXX_' (or after the last '_PO:0000000_')."""
    m = re.search(r"_PO:\d+_(.+)$", col)
    if m:
        return m.group(1)
    # Fallback for P.domestica columns that use PO:0000000 or no PO tag
    m2 = re.search(r"_PO:0000000_(.+)$", col)
    if m2:
        return m2.group(1)
    return None


def assign_tissue(col: str):
    """Return the tissue key that matches, or None."""
    label = tissue_label(col)
    if label is None:
        return None
    label_low = label.lower()
    for tissue, patterns in TISSUE_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, label_low):
                return tissue
    return None


def load_and_filter(csv_path: Path) -> pd.DataFrame:
    """Load CSV, keep only rows with at least one non-zero value, return full df."""
    print(f"  Loading {csv_path.name} …")
    df = pd.read_csv(csv_path, sep="\t", index_col=0)
    df.fillna(0, inplace=True)
    return df


def build_tissue_matrix(df: pd.DataFrame, tissue: str) -> pd.DataFrame:
    """
    For a given tissue, find all matching columns per crop, average them,
    return a DataFrame: rows=proteins, cols=crop names (display).
    """
    cols = list(df.columns)
    out = {}
    for crop, prefix in CROP_PREFIXES.items():
        crop_cols = [c for c in cols if c.startswith(prefix)]
        tissue_cols = [c for c in crop_cols if assign_tissue(c) == tissue]
        if tissue_cols:
            out[CROP_DISPLAY[crop]] = df[tissue_cols].mean(axis=1)
        else:
            # Fill with zeros (crop has no data for this tissue)
            out[CROP_DISPLAY[crop]] = pd.Series(0.0, index=df.index)
    return pd.DataFrame(out)


def load_annotation(tsv_path: Path) -> dict:
    """
    Load OG → search_protein_id mapping.
    Returns dict: {OG_ID: first_search_protein_id_label}
    """
    ann = pd.read_csv(tsv_path, sep="\t", usecols=["Orthogroup", "search_protein_id"])
    result = {}
    for _, row in ann.iterrows():
        og  = str(row["Orthogroup"]).strip()
        ids = str(row["search_protein_id"]).strip()
        # Take first entry before comma; strip whitespace
        first = ids.split(",")[0].strip()
        result[og] = first
    return result


def make_row_label(index_str: str, ann_map: dict) -> str:
    """
    Given an index string like '[HORM] OG0000268|...' or '[CSR] OG0000179|...',
    extract OG ID and look up search_protein_id.
    Returns: 'search_protein_id [OGxxxxxxx]'
    """
    # Strip source tag
    clean = re.sub(r"^\[(HORM|CSR)\]\s*", "", index_str)
    # Extract OG ID (everything before |)
    og_id = clean.split("|")[0].strip()
    label = ann_map.get(og_id, og_id)
    return f"{label} [{og_id}]"


def minmax_excl_zeros(row: pd.Series) -> pd.Series:
    """Min-max normalize a row excluding zero values."""
    vals = row.copy().astype(float)
    non_zero = vals[vals > 0]
    if non_zero.empty or non_zero.max() == non_zero.min():
        return pd.Series(np.nan, index=row.index)
    mn, mx = non_zero.min(), non_zero.max()
    result = (vals - mn) / (mx - mn)
    result[vals == 0] = np.nan          # mask zeros as NaN
    return result


def cluster_columns(df: pd.DataFrame) -> list:
    """
    Return column order after hierarchical clustering using average linkage
    on the column-wise correlation distance. Falls back to original order
    if fewer than 2 valid columns.
    """
    data = df.values.astype(float)
    # Replace NaN with 0 for distance computation
    data_filled = np.nan_to_num(data, nan=0.0)
    if data_filled.shape[1] < 2:
        return list(df.columns)
    try:
        col_dist = pdist(data_filled.T, metric="correlation")
        # Replace NaN distances (all-zero columns) with max distance
        col_dist = np.nan_to_num(col_dist, nan=col_dist[~np.isnan(col_dist)].max()
                                  if np.any(~np.isnan(col_dist)) else 1.0)
        Z = linkage(col_dist, method="average")
        order = leaves_list(Z)
        return [df.columns[i] for i in order]
    except Exception:
        return list(df.columns)
    matrix: pd.DataFrame,
    title: str,
    outpath: Path,
    cmap: str = "viridis",
    vmin=None, vmax=None,
    center=None,
    label: str = "",
):
    """Draw and save a heatmap. Rows = proteins, cols = crops."""
    if matrix.empty or matrix.shape[0] == 0:
        print(f"    [SKIP] empty matrix for {title}")
        return

    n_rows, n_cols = matrix.shape
    fig_h = max(6, 0.35 * n_rows + 2)
    fig_w = max(5, 0.9 * n_cols + 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    data = matrix.values.astype(float)

    if center is not None:
        # diverging
        if vmin is None:
            vmax_abs = np.nanmax(np.abs(data[~np.isnan(data)])) if np.any(~np.isnan(data)) else 1
            vmin, vmax = -vmax_abs, vmax_abs
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    else:
        im = ax.imshow(data, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=label)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(matrix.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_title(title, fontsize=11, pad=10)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {outpath.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data …")
    df_csr  = load_and_filter(CSR_CSV)
    df_horm = load_and_filter(HORM_CSV)

    # Add source tag to index so rows stay distinct when concatenating
    df_csr.index  = [f"[CSR] {i}"  for i in df_csr.index]
    df_horm.index = [f"[HORM] {i}" for i in df_horm.index]

    for tissue in ["Carpel", "Leaves", "FruitFlesh"]:
        print(f"\n── Tissue: {TISSUE_DISPLAY[tissue]} ──")

        mat_csr  = build_tissue_matrix(df_csr,  tissue)
        mat_horm = build_tissue_matrix(df_horm, tissue)

        # Stack hormones on top, then receptors
        combined = pd.concat([mat_horm, mat_csr], axis=0)

        # Drop rows that are all zero (no detections in any crop)
        row_sums = combined.sum(axis=1)
        combined = combined[row_sums > 0]

        if combined.empty:
            print(f"  [SKIP] no valid data for {tissue}")
            continue

        print(f"  Combined matrix: {combined.shape[0]} proteins × {combined.shape[1]} crops")

        # Shorten row labels
        combined.index = [protein_short_name(i) for i in combined.index]

        tissue_dir = OUTDIR / tissue
        tissue_dir.mkdir(exist_ok=True)

        # ── Heatmap 1: log10(x + 1) ───────────────────────────────────────────
        log_mat = np.log10(combined.values.astype(float) + 1)
        log_df  = pd.DataFrame(log_mat, index=combined.index, columns=combined.columns)
        make_heatmap(
            log_df,
            title=f"{TISSUE_DISPLAY[tissue]}\nlog₁₀(mcIBAQ + 1)",
            outpath=tissue_dir / f"{tissue}_log10.png",
            cmap="YlOrRd",
            label="log₁₀(mcIBAQ + 1)",
        )

        # ── Heatmap 2: row-wise z-score ───────────────────────────────────────
        def safe_zscore(row):
            std = row.std()
            if std == 0 or np.isnan(std):
                return pd.Series(np.nan, index=row.index)
            return (row - row.mean()) / std

        zscore_df = combined.apply(safe_zscore, axis=1)
        make_heatmap(
            zscore_df,
            title=f"{TISSUE_DISPLAY[tissue]}\nRow z-score per protein",
            outpath=tissue_dir / f"{tissue}_zscore.png",
            cmap="RdBu_r",
            center=0,
            label="z-score",
        )

        # ── Heatmap 3: min-max per protein (excluding zeros) ──────────────────
        mm_df = combined.apply(minmax_excl_zeros, axis=1)
        make_heatmap(
            mm_df,
            title=f"{TISSUE_DISPLAY[tissue]}\nMin-max per protein (zeros masked)",
            outpath=tissue_dir / f"{tissue}_minmax.png",
            cmap="Blues",
            vmin=0, vmax=1,
            label="normalized intensity",
        )

    print("\nDone. All heatmaps saved to:", OUTDIR)


if __name__ == "__main__":
    main()
