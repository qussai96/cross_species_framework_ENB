#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def minmax_per_row(matrix: np.ndarray) -> np.ndarray:
    row_min = matrix.min(axis=1, keepdims=True)
    row_max = matrix.max(axis=1, keepdims=True)
    denom = row_max - row_min
    return np.divide(matrix - row_min, denom, out=np.zeros_like(matrix), where=denom != 0)


def zscore_per_row(matrix: np.ndarray) -> np.ndarray:
    row_mean = matrix.mean(axis=1, keepdims=True)
    row_std = matrix.std(axis=1, keepdims=True)
    return np.divide(matrix - row_mean, row_std, out=np.zeros_like(matrix), where=row_std != 0)


def extract_orthogroup_id(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    return text.split("|", 1)[0].strip()


def compact_label(text: str, max_len: int = 110) -> str:
    s = str(text)
    if len(s) <= max_len:
        return s
    keep = max_len - 3
    left = keep // 2
    right = keep - left
    return s[:left] + "..." + s[-right:]


def split_matrix_column(col: str) -> Tuple[str, Optional[str], Optional[str]]:
    if "_PO:" not in col:
        return col, None, None
    plant, suffix = col.split("_PO:", 1)
    if "_" not in suffix:
        return plant, "PO:" + suffix, None
    po, tissue = suffix.split("_", 1)
    return plant, "PO:" + po, tissue


def normalize_species_key(name: str) -> str:
    s = str(name).strip()
    for suffix in [".helixer.faa", ".helixer.fa", ".helixer", ".pep.all.faa", ".pep.all.fa", ".faa", ".fa", ".fasta"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def read_plant_name_map(metadata_path: Optional[Union[str, Path]]) -> Dict[str, str]:
    if not metadata_path:
        return {}

    path = Path(metadata_path)
    if not path.exists():
        return {}

    meta = pd.read_csv(path, sep="\t", dtype=str)
    if not {"FASTA", "Name"}.issubset(meta.columns):
        return {}

    out: Dict[str, str] = {}
    for _, row in meta[["FASTA", "Name"]].dropna().iterrows():
        fasta = normalize_species_key(str(row["FASTA"]))
        name = str(row["Name"]).strip()
        if fasta and name:
            out[fasta] = name
    return out


def pretty_sample_label(col: str, plant_name_map: Optional[Dict[str, str]] = None) -> str:
    plant, _, tissue = split_matrix_column(col)
    display_name = plant
    if plant_name_map:
        normalized_plant = normalize_species_key(plant)
        display_name = plant_name_map.get(normalized_plant, plant)
    if tissue:
        return f"{display_name} + {tissue.replace('_', ' ')}"
    return display_name.replace("_", " ")


def build_sample_labels(sample_columns: List[str], metadata_path: Optional[Union[str, Path]] = None) -> List[str]:
    plant_name_map = read_plant_name_map(metadata_path)
    return [pretty_sample_label(col, plant_name_map) for col in sample_columns]


def build_label_map(matched_df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
    if "Orthogroup" not in matched_df.columns:
        raise ValueError("Matched TSV must contain column: Orthogroup")

    label_col = None
    for candidate in ["search_protein_id", "functional_description", "notes"]:
        if candidate in matched_df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError(
            "Matched TSV must contain one of: search_protein_id, functional_description, notes"
        )

    tmp = matched_df[["Orthogroup", label_col]].copy()
    tmp["Orthogroup"] = tmp["Orthogroup"].astype(str).str.strip()
    tmp[label_col] = tmp[label_col].fillna("").astype(str).str.strip()
    tmp = tmp[tmp["Orthogroup"] != ""]

    ordered_orthogroups = tmp["Orthogroup"].drop_duplicates().tolist()

    mapping: Dict[str, str] = {}
    for orthogroup, group in tmp.groupby("Orthogroup", sort=False):
        seen = set()
        labels = []
        for value in group[label_col].tolist():
            if value and value not in seen:
                seen.add(value)
                labels.append(value)
        mapping[orthogroup] = ", ".join(labels) if labels else "NA"

    return mapping, ordered_orthogroups


def choose_figure_size(n_rows: int, n_cols: int) -> Tuple[float, float]:
    width = max(16.0, n_cols * 0.065 + 6.0)
    height = max(10.0, n_rows * 0.16 + 4.0)
    return width, height


def draw_heatmap(
    matrix: np.ndarray,
    title: str,
    out_path: Path,
    x_labels: List[str],
    y_labels: List[str],
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    n_rows, n_cols = matrix.shape
    fig_w, fig_h = choose_figure_size(n_rows, n_cols)

    if len(x_labels) != n_cols:
        if len(x_labels) > n_cols:
            x_labels = x_labels[:n_cols]
        else:
            x_labels = x_labels + [""] * (n_cols - len(x_labels))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel("Plant + Tissue")
    ax.set_ylabel("Matched label / orthogroup")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=4)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.tick_params(axis="x", pad=1)
    ax.tick_params(axis="y", pad=1)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=270, labelpad=16)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate four pathway heatmaps from orthogroup matrix and matched orthogroup table."
    )
    parser.add_argument("--input", required=True, help="Input orthogroup PO matrix TSV/CSV path")
    parser.add_argument("--matched-tsv", required=True, help="Matched orthogroups TSV path")
    parser.add_argument("--output-dir", required=True, help="Directory for heatmap PNG outputs")
    parser.add_argument(
        "--all-tissues-template",
        default=None,
        help=(
            "Optional matrix file used as a tissue-column template. "
            "When set, heatmaps include all template tissue columns in template order; "
            "missing columns in --input are added as zeros."
        ),
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata TSV with FASTA and Name columns used to resolve plant display names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    matched_path = Path(args.matched_tsv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, sep="\t")
    if df.shape[1] < 2:
        raise ValueError("Input must have one ID column and at least one numeric sample column.")

    matched_df = pd.read_csv(matched_path, sep="\t", dtype=str)
    label_map, ordered_orthogroups = build_label_map(matched_df)

    id_col = df.columns[0]
    work = df.copy()
    work["_orthogroup"] = work[id_col].map(extract_orthogroup_id)
    work = work[work["_orthogroup"].isin(set(ordered_orthogroups))].copy()
    if work.empty:
        raise ValueError("No overlapping orthogroups between matrix and matched TSV.")

    order_rank = {orthogroup: i for i, orthogroup in enumerate(ordered_orthogroups)}
    work["_order_rank"] = work["_orthogroup"].map(order_rank).fillna(len(order_rank)).astype(int)
    work = work.sort_values("_order_rank", kind="stable")

    helper_cols = {"_orthogroup", "_order_rank"}
    input_sample_columns = [c for c in df.columns if c != id_col]
    sample_columns = [c for c in input_sample_columns if c not in helper_cols]

    if args.all_tissues_template:
        template_path = Path(args.all_tissues_template)
        template_df = pd.read_csv(template_path, sep="\t", nrows=0)
        template_id_col = template_df.columns[0]
        template_sample_cols = [
            c for c in template_df.columns if c != template_id_col and c not in helper_cols
        ]

        # Keep template order, then append any input columns absent from template.
        sample_columns = template_sample_cols + [
            c for c in sample_columns if c not in set(template_sample_cols)
        ]

    sample_columns = [
        col for _, col in sorted(
            zip(build_sample_labels(sample_columns, args.metadata), sample_columns),
            key=lambda item: item[0].casefold(),
        )
    ]
    sample_labels = build_sample_labels(sample_columns, args.metadata)

    # Reindex to guarantee all requested tissues are present (missing -> zero).
    work = work.reindex(columns=[id_col, "_orthogroup", "_order_rank"] + sample_columns, fill_value=0.0)
    row_labels = [
        compact_label(f"{label_map.get(orthogroup, 'NA')} [{orthogroup}]")
        for orthogroup in work["_orthogroup"].tolist()
    ]

    numeric_df = work[sample_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    matrix = numeric_df.to_numpy(dtype=float)

    log10_matrix = np.log10(matrix + 1.0)
    minmax_matrix = minmax_per_row(matrix)
    log10_minmax_matrix = minmax_per_row(log10_matrix)
    zscore_matrix = zscore_per_row(matrix)

    transformations = [
        (log10_matrix, "log10(intensity + 1)", output_dir / "heatmap_log10_intensity_all_tissues.jpg", "OrRd", None, None),
        (minmax_matrix, "Min-Max per row (raw intensity)", output_dir / "heatmap_minmax_per_row_all_tissues.jpg", "OrRd", 0.0, 1.0),
        (log10_minmax_matrix, "log10(intensity + 1) + Min-Max per row", output_dir / "heatmap_log10_minmax_per_row_all_tissues.jpg", "OrRd", 0.0, 1.0),
        (zscore_matrix, "Z-score per row (raw intensity)", output_dir / "heatmap_zscore_per_row_all_tissues.jpg", "OrRd", -3.0, 3.0),
    ]

    for mat, title, out_file, cmap, vmin, vmax in transformations:
        draw_heatmap(mat, title, out_file, sample_labels, row_labels, cmap, vmin=vmin, vmax=vmax)
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()