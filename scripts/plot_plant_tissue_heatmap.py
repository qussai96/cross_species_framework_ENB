#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def read_table_auto(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def normalize_species_col(name: str) -> str:
    s = str(name).strip()
    for suffix in [".helixer.faa", ".helixer", ".faa", ".fasta"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def split_multi_ids(cell: object) -> List[str]:
    if pd.isna(cell):
        return []
    return [x.strip() for x in str(cell).split(",") if x.strip()]


def compact_label(text: str, max_len: int = 95) -> str:
    s = str(text)
    if len(s) <= max_len:
        return s
    keep = max_len - 3
    left = keep // 2
    right = keep - left
    return s[:left] + "..." + s[-right:]


def get_orthogroup_col(df: pd.DataFrame) -> str:
    if "Orthogroup" in df.columns:
        return "Orthogroup"
    return df.columns[0]


def get_matrix_orthogroup_col(df: pd.DataFrame) -> str:
    if "orthogroup_with_desc" in df.columns:
        return "orthogroup_with_desc"
    return df.columns[0]


def matrix_plants(matrix_df: pd.DataFrame, og_col: str) -> Dict[str, List[str]]:
    plants: Dict[str, List[str]] = {}
    for col in matrix_df.columns:
        if col == og_col:
            continue
        if "_PO:" not in col:
            continue
        plant_key = col.split("_PO:", 1)[0]
        plants.setdefault(plant_key, []).append(col)
    return plants


def resolve_plant_key(user_plant: str, plant_keys: List[str]) -> str:
    exact = [p for p in plant_keys if p == user_plant]
    if exact:
        return exact[0]

    lower_exact = [p for p in plant_keys if p.lower() == user_plant.lower()]
    if lower_exact:
        return lower_exact[0]

    contains = [p for p in plant_keys if user_plant.lower() in p.lower()]
    if len(contains) == 1:
        return contains[0]

    if not contains:
        raise ValueError(
            f"Plant '{user_plant}' not found in matrix columns. "
            f"Available plants: {', '.join(plant_keys)}"
        )

    raise ValueError(
        f"Plant '{user_plant}' matched multiple plants: {', '.join(contains)}. "
        "Please provide a more specific name."
    )


def choose_plant_interactive(plant_keys: List[str]) -> str:
    print("Available plants:\n")
    for i, p in enumerate(plant_keys, start=1):
        print(f"{i}. {p}")
    print()

    raw = input("Select plant by number or name: ").strip()
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(plant_keys):
            return plant_keys[idx - 1]
        raise ValueError(f"Invalid plant index: {idx}")

    return resolve_plant_key(raw, plant_keys)


def find_plant_protein_column(orth_df: pd.DataFrame, plant_key: str) -> str:
    normalized = {c: normalize_species_col(c) for c in orth_df.columns}

    exact = [c for c, n in normalized.items() if n == plant_key]
    if exact:
        return exact[0]

    contains = [c for c, n in normalized.items() if plant_key in n or n in plant_key]
    if len(contains) == 1:
        return contains[0]

    if not contains:
        raise ValueError(
            f"Could not map plant '{plant_key}' to any orthogroup species column."
        )

    raise ValueError(
        f"Plant '{plant_key}' matched multiple orthogroup columns: {', '.join(contains)}"
    )


def tissue_name_map(tissue_ontology: str | None) -> Dict[str, str]:
    if not tissue_ontology:
        return {}

    df = pd.read_csv(tissue_ontology, sep="\t", dtype=str)
    po_cols = [c for c in df.columns if c.lower() in {"po_1", "po", "po_id"}]
    if not po_cols:
        return {}
    po_col = po_cols[0]

    preferred_name_cols = ["PO_term_1", "TissueName"]
    candidate_name_cols = [c for c in preferred_name_cols if c in df.columns]
    if not candidate_name_cols:
        candidate_name_cols = [
            c
            for c in df.columns
            if c != po_col and ("name" in c.lower() or "tissue" in c.lower() or "label" in c.lower())
        ]
    if not candidate_name_cols:
        return {}
    name_col = candidate_name_cols[0]

    out = {}
    for _, row in df[[po_col, name_col]].dropna().iterrows():
        po = str(row[po_col]).strip()
        nm = str(row[name_col]).strip()
        if po and nm:
            out[po] = nm
    return out


def build_heatmap_table(
    matrix_df: pd.DataFrame,
    matrix_og_col: str,
    orth_df: pd.DataFrame,
    orth_og_col: str,
    plant_key: str,
    plant_cols: List[str],
    plant_protein_col: str,
) -> pd.DataFrame:
    m = matrix_df.copy()
    m["Orthogroup"] = m[matrix_og_col].astype(str).str.split("|", n=1).str[0]

    search_col = "search_protein_id" if "search_protein_id" in orth_df.columns else None
    if search_col is None:
        raise ValueError("Orthogroups table is missing 'search_protein_id' column.")

    keep_cols = [orth_og_col, search_col, plant_protein_col]
    og_map = orth_df[keep_cols].copy()
    og_map = og_map.rename(columns={orth_og_col: "Orthogroup"})

    rows = []
    for _, row in og_map.iterrows():
        proteins = split_multi_ids(row[plant_protein_col])
        if not proteins:
            continue
        search_ids = ",".join(split_multi_ids(row[search_col]))
        orthogroup = str(row["Orthogroup"])
        for plant_protein in proteins:
            rows.append(
                {
                    "Orthogroup": orthogroup,
                    "search_protein_id": search_ids,
                    "plant_protein_id": plant_protein,
                }
            )

    if not rows:
        raise ValueError(
            f"No plant proteins found for plant '{plant_key}' in orthogroups file column '{plant_protein_col}'."
        )

    expanded = pd.DataFrame(rows)
    merged = expanded.merge(m[["Orthogroup"] + plant_cols], on="Orthogroup", how="inner")

    for col in plant_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    detected = merged[plant_cols].sum(axis=1) > 0
    merged = merged.loc[detected].copy()

    if merged.empty:
        raise ValueError(
            f"No detected proteins for plant '{plant_key}' (all selected intensities are 0)."
        )

    merged["y_label"] = (
        merged["search_protein_id"].astype(str)
        + " / "
        + merged["plant_protein_id"].astype(str)
        + " / "
        + merged["Orthogroup"].astype(str)
    )

    out = merged[["y_label"] + plant_cols].drop_duplicates(subset=["y_label"])
    out = out.set_index("y_label")
    out["_total_intensity"] = out[plant_cols].sum(axis=1)
    out = out.sort_values(by="_total_intensity", ascending=False)
    out = out.drop(columns=["_total_intensity"])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot tissue heatmap for proteins detected in one selected plant."
    )
    parser.add_argument("--matrix", required=True, help="protein_orthogroup_po_matrix.csv path")
    parser.add_argument("--orthogroups", required=True, help="Assigned_Orthogroups.tsv path")
    parser.add_argument("--plant", default=None, help="Plant key/name to select (optional interactive selection)")
    parser.add_argument(
        "--all-plants",
        action="store_true",
        help="Generate outputs for every plant found in the matrix",
    )
    parser.add_argument(
        "--tissue-ontology",
        default=None,
        help="Optional tissue ontology TSV for mapping PO IDs to readable tissue names",
    )
    parser.add_argument(
        "--output-png",
        default=None,
        help="Output PNG heatmap path (default: <matrix_dir>/<plant>_plant_tissues_heatmap.png)",
    )
    parser.add_argument(
        "--output-tsv",
        default=None,
        help="Output TSV table path (default: <matrix_dir>/<plant>_plant_tissues_intensities.tsv)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for batch mode or as the default base directory for single-plant outputs",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200,
        help="Max proteins (rows) to plot, ranked by total intensity",
    )
    parser.add_argument(
        "--figscale",
        type=float,
        default=0.35,
        help="Figure scaling factor for dynamic width/height",
    )
    parser.add_argument(
        "--max-label-len",
        type=int,
        default=95,
        help="Maximum displayed y-label length in heatmap (full values remain in TSV)",
    )
    return parser.parse_args()


def render_plant(
    *,
    matrix_df: pd.DataFrame,
    matrix_og_col: str,
    orth_df: pd.DataFrame,
    orth_og_col: str,
    plant_key: str,
    plant_cols: List[str],
    plant_protein_col: str,
    tissue_ontology_path: str | None,
    matrix_dir: Path,
    output_dir: Path | None,
    output_png: str | None,
    output_tsv: str | None,
    max_rows: int,
    figscale: float,
    max_label_len: int,
) -> None:
    table = build_heatmap_table(
        matrix_df=matrix_df,
        matrix_og_col=matrix_og_col,
        orth_df=orth_df,
        orth_og_col=orth_og_col,
        plant_key=plant_key,
        plant_cols=plant_cols,
        plant_protein_col=plant_protein_col,
    )

    if max_rows > 0 and len(table) > max_rows:
        table = table.iloc[: max_rows].copy()

    po_to_name = tissue_name_map(tissue_ontology_path)
    pretty_cols = []
    for col in plant_cols:
        po_raw = col.split("_PO:", 1)[1] if "_PO:" in col else col
        po_key = po_raw if str(po_raw).startswith("PO:") else f"PO:{po_raw}"
        tissue = po_to_name.get(po_key, po_to_name.get(po_raw, po_raw))
        pretty_cols.append(tissue)

    plot_table = table.copy()
    plot_table.columns = pretty_cols

    safe_plant = plant_key.replace("/", "_")
    base_dir = output_dir if output_dir else matrix_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    out_png = Path(output_png) if output_png else base_dir / f"{safe_plant}_plant_tissues_heatmap.png"
    out_tsv = Path(output_tsv) if output_tsv else base_dir / f"{safe_plant}_plant_tissues_intensities.tsv"
    raw_out_png = base_dir / f"{safe_plant}_plant_tissues_raw_heatmap.png"

    table.to_csv(out_tsv, sep="\t")

    display_vals = np.log10(plot_table + 1.0)
    row_min = display_vals.min(axis=1)
    row_max = display_vals.max(axis=1)
    row_range = (row_max - row_min).replace(0, np.nan)
    display_vals = display_vals.sub(row_min, axis=0).div(row_range, axis=0)
    display_vals = display_vals.fillna(0.0)
    display_vals.index = [compact_label(v, max_len=max_label_len) for v in display_vals.index]

    width = max(14, len(plot_table.columns) * 0.7 + 6)
    height = max(8, len(plot_table.index) * max(figscale, 0.22) + 2)

    plt.figure(figsize=(width, height))

    ax = sns.heatmap(
        display_vals,
        cmap='OrRd',
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Normalized intensity (0-1)"},
    )
    plt.title(f"Plant Tissue Heatmap: {plant_key}\n(y = search_protein_id / plant_protein_id / orthogroup)")
    plt.xlabel("Tissues")
    plt.ylabel("Proteins")
    plt.xticks(rotation=90, ha="center", fontsize=9)
    plt.yticks(fontsize=8)
    ax.tick_params(axis="y", pad=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    raw_width = max(14, len(table.columns) * 0.7 + 6)
    raw_height = max(8, len(table.index) * max(figscale, 0.22) + 2)

    plt.figure(figsize=(raw_width, raw_height))
    raw_ax = sns.heatmap(
        table,
        cmap='OrRd',
        cbar_kws={"label": "Intensity"},
    )
    plt.title(f"Plant Tissue Raw Intensities: {plant_key}\n(y = search_protein_id / plant_protein_id / orthogroup)")
    plt.xlabel("Tissues")
    plt.ylabel("Proteins")
    plt.xticks(rotation=90, ha="center", fontsize=9)
    plt.yticks(fontsize=8)
    raw_ax.tick_params(axis="y", pad=2)
    plt.tight_layout()
    plt.savefig(raw_out_png, dpi=200)
    plt.close()

    print(f"Plant selected: {plant_key}")
    print(f"Orthogroup plant column: {plant_protein_col}")
    print(f"Rows plotted: {len(plot_table)}")
    print(f"Tissues: {len(plot_table.columns)}")
    print(f"Saved table: {out_tsv}")
    print(f"Saved heatmap: {out_png}")
    print(f"Saved raw heatmap: {raw_out_png}")


def main() -> None:
    args = parse_args()

    tissue_ontology_path = args.tissue_ontology
    if not tissue_ontology_path:
        default_tissue_ontology = Path(__file__).resolve().parents[2] / "ENB_TissueOntology_long.tsv"
        if default_tissue_ontology.exists():
            tissue_ontology_path = str(default_tissue_ontology)

    matrix_df = read_table_auto(args.matrix)
    matrix_og_col = get_matrix_orthogroup_col(matrix_df)

    plants = matrix_plants(matrix_df, matrix_og_col)
    if not plants:
        raise ValueError("No plant tissue columns found in matrix (expected '<plant>_PO:<id>' columns).")

    plant_keys = sorted(plants.keys())

    orth_df = pd.read_csv(args.orthogroups, sep="\t", dtype=str)
    orth_og_col = get_orthogroup_col(orth_df)
    matrix_dir = Path(args.matrix).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.all_plants:
        failures = []
        for plant_key in plant_keys:
            plant_cols = plants[plant_key]
            try:
                plant_protein_col = find_plant_protein_column(orth_df, plant_key)
                render_plant(
                    matrix_df=matrix_df,
                    matrix_og_col=matrix_og_col,
                    orth_df=orth_df,
                    orth_og_col=orth_og_col,
                    plant_key=plant_key,
                    plant_cols=plant_cols,
                    plant_protein_col=plant_protein_col,
                    tissue_ontology_path=tissue_ontology_path,
                    matrix_dir=matrix_dir,
                    output_dir=output_dir,
                    output_png=None,
                    output_tsv=None,
                    max_rows=args.max_rows,
                    figscale=args.figscale,
                    max_label_len=args.max_label_len,
                )
            except Exception as exc:
                failures.append((plant_key, str(exc)))
                print(f"Skipping plant '{plant_key}': {exc}")

        print(f"Processed {len(plant_keys) - len(failures)} plants; skipped {len(failures)} plants.")
        return

    plant_key = resolve_plant_key(args.plant, plant_keys) if args.plant else choose_plant_interactive(plant_keys)
    plant_cols = plants[plant_key]
    plant_protein_col = find_plant_protein_column(orth_df, plant_key)
    render_plant(
        matrix_df=matrix_df,
        matrix_og_col=matrix_og_col,
        orth_df=orth_df,
        orth_og_col=orth_og_col,
        plant_key=plant_key,
        plant_cols=plant_cols,
        plant_protein_col=plant_protein_col,
        tissue_ontology_path=tissue_ontology_path,
        matrix_dir=matrix_dir,
        output_dir=output_dir,
        output_png=args.output_png,
        output_tsv=args.output_tsv,
        max_rows=args.max_rows,
        figscale=args.figscale,
        max_label_len=args.max_label_len,
    )


if __name__ == "__main__":
    main()
