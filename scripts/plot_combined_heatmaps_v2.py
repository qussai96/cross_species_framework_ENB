"""
Cross-species heatmaps: Hormones + Cell Surface Receptors
Tissues: Carpel | Leaves | Fruit Flesh
Crops: Apple, Pear, Quince, Sweet Cherry, Sour Cherry, Apricot, Peach, Plum
Normalizations: log10(x+1) | row z-score | min-max (excl. zeros)

Y-axis : search_protein_id from annotation TSVs, sorted alphabetically
X-axis : crops hierarchically clustered by expression similarity (per tissue)
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path("/home/students/q.abbas/+proj-q.abbas/Workflow_orthogroups/"
                 "cross_species_framework_ENB/Hormones_Julien_list")
CSR_CSV   = BASE / "Cell_surface_receptors/protein_orthogroup_po_matrix_mcIBAQ_intensities.csv"
HORM_CSV  = BASE / "Hormones_list/protein_orthogroup_po_matrix_mcIBAQ_intensities.csv"
CSR_ANN   = BASE / "Cell_surface_receptors/Orthogroups_matched_to_Cell_surface_receptors.tsv"
HORM_ANN  = BASE / "Hormones_list/Orthogroups_matched_to_Hormones_list.tsv"
OUTDIR    = BASE / "heatmaps_combined"
OUTDIR.mkdir(exist_ok=True)

# ── Crop → column prefix mapping ───────────────────────────────────────────────
CROP_PREFIXES = {
    "Apple":       "Mdomesticus_Gala_haploid_v2.pep.fa",
    "Pear":        "PyrusCommunis_BartlettDHv2.0.pep",
    "Quince":      "GCA_015708375.1_Cydonia_oblonga",
    "SweetCherry": "GCF_002207925.1_Prunus_avium",
    "SourCherry":  "Pcer_mont_QTL_update2_proteins",
    "Apricot":     "GCA_903112645.1_Prunus_armeniaca",
    "Peach":       "Ppersica_Lovell_2D_v3.0.proteins.fa",
    "Plum":        "P.domestica_Genome-Dardick_v1.0.proteins",
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

# ── Tissue keyword matching (on the text AFTER PO:XXXXXXX_) ───────────────────
# Uses token-based matching (split on underscores/colons) to avoid false
# negatives from underscore-separated naming conventions.
TISSUE_TOKENS = {
    "Carpel":     {"carpel", "pistil"},
    "Leaves":     {"leaf", "leaves", "leave"},   # "leave" covers truncated Quince label
    "FruitFlesh": {"flesh", "mesocarp"},
}

TISSUE_DISPLAY = {
    "Carpel":     "Carpel (PO:0009046)",
    "Leaves":     "Leaves (PO:0006340)",
    "FruitFlesh": "Fruit Flesh (PO:0030110)",
}


# ── Helper functions ───────────────────────────────────────────────────────────

def tissue_label(col: str):
    """Extract the tissue text portion after PO:XXXXXXX_."""
    m = re.search(r"_PO:\d+_(.+)$", col)
    return m.group(1) if m else None


def assign_tissue(col: str):
    """Return the tissue key matching the column, or None.
    Splits the tissue label on underscores/colons to avoid false negatives
    from names like 'fully_opened_leaf' or 'adult_plant_fully_expanded_leave'.
    """
    lbl = tissue_label(col)
    if lbl is None:
        return None
    # Tokenise: lower-case, split on non-alpha chars
    tokens = set(re.split(r"[_:\s]+", lbl.lower()))
    for tissue, kw_set in TISSUE_TOKENS.items():
        if tokens & kw_set:   # any overlap
            return tissue
    return None


def load_csv(path: Path) -> pd.DataFrame:
    print(f"  Loading {path.name} …")
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.fillna(0, inplace=True)
    return df


def load_annotation(tsv_path: Path) -> dict:
    """
    Return {OG_ID: first_search_protein_id} from annotation TSV.
    Multiple comma-separated IDs → first entry only.
    """
    ann = pd.read_csv(tsv_path, sep="\t", usecols=["Orthogroup", "search_protein_id"])
    result = {}
    for _, row in ann.iterrows():
        og    = str(row["Orthogroup"]).strip()
        ids   = str(row["search_protein_id"]).strip()
        first = ids.split(",")[0].strip()
        result[og] = first
    return result


def make_row_label(index_str: str, ann_horm: dict, ann_csr: dict) -> str:
    """
    Convert '[CSR] OG0000179|description' → 'AtRLP11_AT1G71390 [OG0000179]'.
    Uses the source-specific annotation map (HORM or CSR) to avoid label
    cross-contamination for the 7 OG IDs shared between both annotation files.
    """
    m     = re.match(r"^\[(HORM|CSR)\]\s*", index_str)
    src   = m.group(1) if m else "HORM"
    clean = re.sub(r"^\[(HORM|CSR)\]\s*", "", index_str)
    og_id = clean.split("|")[0].strip()
    ann_map = ann_csr if src == "CSR" else ann_horm
    label   = ann_map.get(og_id, og_id)
    return f"{label} [{og_id}] ({src})"


def build_tissue_matrix(df: pd.DataFrame, tissue: str) -> pd.DataFrame:
    """
    For each crop, include every individual column matching the tissue keyword
    as a separate column labelled 'CropName[tissue_label]'.
    Crops with no matching samples are excluded entirely (no zero-fill).
    Duplicate tissue labels (same name from different PO numbers) are
    disambiguated with a _2, _3 … suffix so no data is silently dropped.
    Returns protein × sample DataFrame.
    """
    cols = list(df.columns)
    out  = {}
    for crop, prefix in CROP_PREFIXES.items():
        crop_cols   = [c for c in cols if c.startswith(prefix)]
        tissue_cols = [c for c in crop_cols if assign_tissue(c) == tissue]
        name_count: dict = {}
        for col in tissue_cols:
            lbl      = tissue_label(col) or col
            base     = f"{CROP_DISPLAY[crop]}[{lbl}]"
            # Disambiguate duplicate names (same label from different PO numbers)
            count = name_count.get(base, 0) + 1
            name_count[base] = count
            col_name = base if count == 1 else f"{base}_{count}"
            out[col_name] = df[col]
    return pd.DataFrame(out)


def cluster_column_order(df: pd.DataFrame) -> list:
    """
    Hierarchical clustering of columns using average linkage + correlation distance.
    NaN values replaced with 0 for distance computation.
    """
    data = np.nan_to_num(df.values.astype(float), nan=0.0)
    if data.shape[1] < 2:
        return list(df.columns)
    try:
        dist = pdist(data.T, metric="correlation")
        dist = np.nan_to_num(dist, nan=float(np.nanmax(dist)) if np.any(~np.isnan(dist)) else 1.0)
        Z    = linkage(dist, method="average")
        order = leaves_list(Z)
        return [df.columns[i] for i in order]
    except Exception as e:
        print(f"    [WARN] clustering failed ({e}), using original order")
        return list(df.columns)


def minmax_excl_zeros(row: pd.Series) -> pd.Series:
    """Row-wise min-max normalization; zeros are masked as NaN."""
    vals     = row.copy().astype(float)
    non_zero = vals[vals > 0]
    if non_zero.empty or non_zero.max() == non_zero.min():
        return pd.Series(np.nan, index=row.index)
    mn, mx   = non_zero.min(), non_zero.max()
    result   = (vals - mn) / (mx - mn)
    result[vals == 0] = np.nan
    return result


def safe_zscore(row: pd.Series) -> pd.Series:
    """Row-wise z-score; all-constant rows → NaN."""
    std = row.std()
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=row.index)
    return (row - row.mean()) / std


def make_heatmap(
    matrix: pd.DataFrame,
    title: str,
    outpath: Path,
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
    center=None,
    cbar_label: str = "",
):
    """Save a heatmap PNG. Rows = proteins (y-axis), Cols = crops (x-axis)."""
    if matrix.empty:
        print(f"    [SKIP] empty matrix for '{title}'")
        return

    n_rows, n_cols = matrix.shape
    fig_h = max(6, 0.30 * n_rows + 2.5)
    fig_w = max(5, 0.55 * n_cols + 3.5)   # narrower per-column width fits more samples

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    data = matrix.values.astype(float)

    if center is not None:
        valid = data[~np.isnan(data)]
        if valid.size == 0:
            vmax_abs = 1.0
        else:
            vmax_abs = max(abs(float(np.nanmin(valid))), abs(float(np.nanmax(valid))))
        if vmin is None:
            vmin, vmax = -vmax_abs, vmax_abs
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    else:
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=cbar_label)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(matrix.columns, rotation=60, ha="right", fontsize=7, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_title(title, fontsize=11, pad=10)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {outpath.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading CSVs …")
    df_csr  = load_csv(CSR_CSV)
    df_horm = load_csv(HORM_CSV)

    # Tag index rows with source
    df_csr.index  = [f"[CSR] {i}"  for i in df_csr.index]
    df_horm.index = [f"[HORM] {i}" for i in df_horm.index]

    print("Loading annotation TSVs …")
    ann_csr  = load_annotation(CSR_ANN)
    ann_horm = load_annotation(HORM_ANN)
    # Keep separate: 7 OG IDs are shared between both files with different labels.
    # Source-specific lookup is applied per row in make_row_label().

    for tissue in ["Carpel", "Leaves", "FruitFlesh"]:
        print(f"\n── Tissue: {TISSUE_DISPLAY[tissue]} ──")

        mat_csr  = build_tissue_matrix(df_csr,  tissue)
        mat_horm = build_tissue_matrix(df_horm, tissue)

        combined = pd.concat([mat_horm, mat_csr], axis=0)

        # Drop all-zero rows
        combined = combined[combined.sum(axis=1) > 0]
        if combined.empty:
            print(f"  [SKIP] no valid data")
            continue

        # ── Y-axis: replace index with source-specific search_protein_id labels ──
        combined.index = [make_row_label(i, ann_horm, ann_csr) for i in combined.index]

        # Remove true duplicates: same OG appearing in both HORM and CSR matrices.
        # These share an identical label (incl. source tag) only when OG IDs overlap;
        # keep the row with the higher total expression.
        combined["_sum"] = combined.sum(axis=1, numeric_only=True)
        combined = combined.sort_values("_sum", ascending=False)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.drop(columns="_sum")

        # ── Sort rows alphabetically by label ──
        combined = combined.sort_index()

        print(f"  Matrix: {combined.shape[0]} proteins × {combined.shape[1]} crops")

        # ── X-axis: cluster crops by log10 expression similarity ──
        log_vals = np.log10(combined.values.astype(float) + 1)
        log_df   = pd.DataFrame(log_vals, index=combined.index, columns=combined.columns)
        col_order = cluster_column_order(log_df)

        tissue_dir = OUTDIR / tissue
        tissue_dir.mkdir(exist_ok=True)

        # ── Heatmap 1: log10(mcIBAQ + 1) ──
        log_ordered = log_df[col_order]
        make_heatmap(
            log_ordered,
            title=f"{TISSUE_DISPLAY[tissue]}\nlog\u2081\u2080(mcIBAQ + 1)",
            outpath=tissue_dir / f"{tissue}_log10.png",
            cmap="YlOrRd",
            cbar_label="log\u2081\u2080(mcIBAQ + 1)",
        )

        # ── Heatmap 2: row z-score ──
        zscore_df = combined[col_order].apply(safe_zscore, axis=1)
        make_heatmap(
            zscore_df,
            title=f"{TISSUE_DISPLAY[tissue]}\nRow z-score per protein",
            outpath=tissue_dir / f"{tissue}_zscore.png",
            cmap="RdBu_r",
            center=0,
            cbar_label="z-score",
        )

        # ── Heatmap 3: min-max per protein (zeros masked) ──
        mm_df = combined[col_order].apply(minmax_excl_zeros, axis=1)
        make_heatmap(
            mm_df,
            title=f"{TISSUE_DISPLAY[tissue]}\nMin-max per protein (zeros masked)",
            outpath=tissue_dir / f"{tissue}_minmax.png",
            cmap="Blues",
            vmin=0, vmax=1,
            cbar_label="normalized intensity",
        )

    print(f"\nDone. All heatmaps saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
