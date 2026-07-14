#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def split_matrix_column(col: str) -> tuple[str, str | None, str | None]:
    if "_PO:" not in col:
        return col, None, None
    plant, suffix = col.split("_PO:", 1)
    if "_" not in suffix:
        return plant, f"PO:{suffix}", None
    po, tissue = suffix.split("_", 1)
    return plant, f"PO:{po}", tissue


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
        plant_key, _, _ = split_matrix_column(col)
        if plant_key == col:
            continue
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


def read_plant_name_map(metadata_path: str | None) -> Dict[str, str]:
    if not metadata_path:
        return {}

    meta = pd.read_csv(metadata_path, sep="\t", dtype=str)
    required_cols = {"FASTA", "Name"}
    if not required_cols.issubset(meta.columns):
        raise ValueError(f"Metadata TSV must contain columns {sorted(required_cols)}")

    out: Dict[str, str] = {}
    for _, row in meta[["FASTA", "Name"]].dropna().iterrows():
        fasta = normalize_species_col(str(row["FASTA"]))
        name = str(row["Name"]).strip()
        if fasta and name:
            out[fasta] = name
    return out


def safe_filename(text: str) -> str:
    value = str(text).strip()
    value = re.sub(r"[\s/\\]+", "_", value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value or "output"


def _norm_text(s: str) -> str:
    return " ".join(str(s).replace("_", " ").strip().lower().split())


def build_outlier_tissue_keys(
    problem_report_path: str | None,
    tissue_ontology_path: str | None,
) -> Set[Tuple[str, str, str]]:
    if not problem_report_path or not tissue_ontology_path:
        return set()

    pr = pd.read_csv(problem_report_path, sep="\t", dtype=str)
    if pr.empty:
        return set()

    outlier_col_candidates = [c for c in pr.columns if c.strip().lower() in {"outlier", "oulier"}]
    plant_col_candidates = [c for c in pr.columns if c.strip().lower() == "plant"]
    tissue_col_candidates = [c for c in pr.columns if c.strip().lower() == "tissue"]
    if not outlier_col_candidates or not tissue_col_candidates:
        return set()

    outlier_col = outlier_col_candidates[0]
    plant_col = plant_col_candidates[0] if plant_col_candidates else None
    tissue_col = tissue_col_candidates[0]

    outlier_rows = pr[pr[outlier_col].astype(str).str.strip() == "1"]
    if outlier_rows.empty:
        return set()

    outlier_pnumbers = {
        str(v).strip()
        for v in outlier_rows[tissue_col].dropna().tolist()
        if str(v).strip()
    }
    if not outlier_pnumbers:
        return set()

    allowed_plants_by_pnumber: Dict[str, Set[str]] = {}
    if plant_col:
        for _, row in outlier_rows[[plant_col, tissue_col]].dropna(subset=[tissue_col]).iterrows():
            pn = str(row[tissue_col]).strip()
            if not pn:
                continue
            plant_raw = row.get(plant_col)
            if pd.isna(plant_raw):
                continue
            plant_norm = _norm_text(str(plant_raw))
            if not plant_norm:
                continue
            allowed_plants_by_pnumber.setdefault(pn, set()).add(plant_norm)

    ont = pd.read_csv(tissue_ontology_path, sep="\t", dtype=str)
    if ont.empty or "pNumber" not in ont.columns:
        return set()

    outlier_keys: Set[Tuple[str, str, str]] = set()
    ont_hits = ont[ont["pNumber"].astype(str).str.strip().isin(outlier_pnumbers)]

    for _, row in ont_hits.iterrows():
        pnum = str(row.get("pNumber", "")).strip()
        ont_plant_norm = _norm_text(str(row.get("PlantName", "")))
        allowed_plants = allowed_plants_by_pnumber.get(pnum)
        if allowed_plants and ont_plant_norm and ont_plant_norm not in allowed_plants:
            continue

        po_raw = row.get("PO_1")
        tissue_raw = row.get("TissueName")
        if pd.notna(po_raw) and pd.notna(tissue_raw):
            po = str(po_raw).strip()
            tissue_name = str(tissue_raw).strip()
            if po and tissue_name and not po.lower().startswith("no appropriate"):
                outlier_keys.add((ont_plant_norm, po, _norm_text(tissue_name)))

        # Fallback for datasets where column labels are PO-term based.
        term_raw = row.get("PO_term_1")
        if pd.notna(po_raw) and pd.notna(term_raw):
            po = str(po_raw).strip()
            term = str(term_raw).strip()
            if po and term and not po.lower().startswith("no appropriate"):
                outlier_keys.add((ont_plant_norm, po, _norm_text(term)))

    return outlier_keys


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
    parser.add_argument("--matrix", required=True, help="orthogroup matrix TSV path")
    parser.add_argument("--orthogroups", required=True, help="Assigned_Orthogroups.tsv path")
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata TSV with FASTA and Name columns for output filename labels",
    )
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
        "--problem-report",
        default=None,
        help="Optional ProblemReport TSV (outlier tissues with Outlier/Oulier and Tissue columns)",
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
    outlier_tissue_keys: Set[Tuple[str, str, str]],
    matrix_dir: Path,
    output_dir: Path | None,
    output_png: str | None,
    output_tsv: str | None,
    plant_display_name: str | None,
    max_rows: int,
    figscale: float,
    max_label_len: int,
) -> None:
    plant_norm = _norm_text(plant_display_name or plant_key)

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

    pretty_cols = []
    outlier_col_indices = set()
    for col in plant_cols:
        col_idx = len(pretty_cols)
        _, po_id, tissue_name = split_matrix_column(col)
        if po_id and tissue_name:
            pretty_cols.append(f"{po_id} {tissue_name.replace('_', ' ')}")
            if (plant_norm, po_id, _norm_text(tissue_name)) in outlier_tissue_keys:
                outlier_col_indices.add(col_idx)
        elif po_id:
            pretty_cols.append(po_id)
        else:
            pretty_cols.append(col.replace("_", " "))

    plot_table = table.copy()
    plot_table.columns = pretty_cols

    display_name = plant_display_name or plant_key
    safe_plant = safe_filename(display_name)
    base_dir = output_dir if output_dir else matrix_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    out_png = Path(output_png) if output_png else base_dir / f"{safe_plant}.png"
    out_tsv = Path(output_tsv) if output_tsv else base_dir / f"{safe_plant}_plant_tissues_maxlfq_no_edits.tsv"

    out_png_q1_q3 = out_png.with_name(f"{safe_plant}_q1_q3_capped{out_png.suffix}")
    out_png_row_minmax = out_png.with_name(
        f"{safe_plant}_min_max{out_png.suffix}"
    )
    out_png_orthogroup_collapsed = out_png.with_name(
        f"{safe_plant}_collapsed{out_png.suffix}"
    )

    table.to_csv(out_tsv, sep="\t")

    def save_heatmap(
        data: pd.DataFrame,
        output_path: Path,
        title: str,
        cbar_label: str,
        *,
        cmap: str = "OrRd",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        plot_data = data.copy()
        plot_data.index = [compact_label(v, max_len=max_label_len) for v in plot_data.index]

        width = max(14, len(plot_data.columns) * 0.7 + 6)
        height = max(8, len(plot_data.index) * max(figscale, 0.22) + 2)

        plt.figure(figsize=(width, height))
        ax = sns.heatmap(
            plot_data,
            cmap=cmap,
            cbar_kws={"label": cbar_label},
            vmin=vmin,
            vmax=vmax,
        )
        plt.title(title)
        plt.xlabel("Tissues")
        plt.ylabel("Proteins")
        plt.xticks(rotation=90, ha="center", fontsize=9)
        plt.yticks(fontsize=8)
        for idx, tick in enumerate(ax.get_xticklabels()):
            if idx in outlier_col_indices:
                tick.set_color("blue")
                tick.set_fontweight("bold")
        ax.tick_params(axis="y", pad=2)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    # 1) Same as current behavior.
    display_current = np.log10(plot_table.mask(plot_table <= 0))
    save_heatmap(
        data=display_current,
        output_path=out_png,
        title=f"Plant Tissue log10 Intensities: {plant_key}\n(y = search_protein_id / plant_protein_id / orthogroup)",
        cbar_label="log10(intensity)",
        cmap="OrRd",
    )

    # 2) log10(maxLFQ+1) with color scale capped at Q1..Q3.
    display_log_plus1 = np.log10(plot_table + 1.0)
    finite_vals = display_log_plus1.to_numpy().ravel()
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    q1 = float(np.quantile(finite_vals, 0.25))
    q3 = float(np.quantile(finite_vals, 0.75))
    if q1 == q3:
        q3 = q1 + 1e-9
    display_log_plus1_capped = display_log_plus1.clip(lower=q1, upper=q3)
    # save_heatmap(
    #     data=display_log_plus1_capped,
    #     output_path=out_png_q1_q3,
    #     title=(
    #         f"Plant Tissue log10(MaxLFQ+1), capped to Q1-Q3: {plant_key}\\n"
    #         f"(Q1={q1:.3f}, Q3={q3:.3f}; y = search_protein_id / plant_protein_id / orthogroup)"
    #     ),
    #     cbar_label="log10(MaxLFQ+1)",
    #     cmap="OrRd",
    #     vmin=q1,
    #     vmax=q3,
    # )

    # 3) Row-wise min-max scaling (per protein/orthogroup label) on log10(maxLFQ+1).
    row_mins = display_log_plus1.min(axis=1)
    row_maxs = display_log_plus1.max(axis=1)
    row_ranges = (row_maxs - row_mins).replace(0, np.nan)
    display_row_minmax = display_log_plus1.sub(row_mins, axis=0).div(row_ranges, axis=0).fillna(0.0)
    save_heatmap(
        data=display_row_minmax,
        output_path=out_png_row_minmax,
        title=f"Plant Tissue Row Min-Max on log10(MaxLFQ+1): {plant_key}\\n(y = search_protein_id / plant_protein_id / orthogroup)",
        cbar_label="row min-max (0-1)",
        cmap="OrRd",
        vmin=0.0,
        vmax=1.0,
    )

    # 4) Collapse by removing identical rows within each orthogroup (no max aggregation).
    orthogroup_labels = table.index.to_series().astype(str).str.rsplit(" / ", n=1).str[-1]
    collapsed = plot_table.copy()
    collapsed["_orthogroup"] = orthogroup_labels.values
    collapsed = collapsed.drop_duplicates(subset=["_orthogroup"] + list(plot_table.columns))
    collapsed = collapsed.set_index("_orthogroup")
    collapsed["_total_intensity"] = collapsed.sum(axis=1)
    collapsed = collapsed.sort_values(by="_total_intensity", ascending=False).drop(columns=["_total_intensity"])
    display_collapsed = np.log10(collapsed.mask(collapsed <= 0))
    save_heatmap(
        data=display_collapsed,
        output_path=out_png_orthogroup_collapsed,
        title=f"Plant Tissue log10 Intensities, orthogroup deduplicated: {plant_key}\n(y = orthogroup)",
        cbar_label="log10(intensity)",
        cmap="OrRd",
    )

    if display_name != plant_key:
        print(f"Plant selected: {display_name} ({plant_key})")
    else:
        print(f"Plant selected: {plant_key}")
    print(f"Orthogroup plant column: {plant_protein_col}")
    print(f"Rows plotted: {len(plot_table)}")
    print(f"Tissues: {len(plot_table.columns)}")
    print(f"Saved raw maxLFQ table (no edits): {out_tsv}")
    print(f"Saved heatmap (current): {out_png}")
    print(f"Saved heatmap (Q1-Q3 capped): {out_png_q1_q3}")
    print(f"Saved heatmap (row min-max): {out_png_row_minmax}")
    print(f"Saved heatmap (orthogroup-collapsed): {out_png_orthogroup_collapsed}")


def main() -> None:
    args = parse_args()

    tissue_ontology_path = args.tissue_ontology
    if not tissue_ontology_path:
        default_tissue_ontology = Path(__file__).resolve().parents[2] / "ENB_TissueOntology_long.tsv"
        if default_tissue_ontology.exists():
            tissue_ontology_path = str(default_tissue_ontology)

    problem_report_path = args.problem_report
    if not problem_report_path:
        default_problem_report = Path(__file__).resolve().parents[1] / "ProblemReport.tsv"
        if default_problem_report.exists():
            problem_report_path = str(default_problem_report)

    outlier_tissue_keys = build_outlier_tissue_keys(problem_report_path, tissue_ontology_path)
    plant_name_map = read_plant_name_map(args.metadata) if args.metadata else {}

    matrix_df = read_table_auto(args.matrix)
    matrix_og_col = get_matrix_orthogroup_col(matrix_df)

    plants = matrix_plants(matrix_df, matrix_og_col)
    if not plants:
        raise ValueError("No plant tissue columns found in matrix (expected '<plant>_PO:<id>_<tissue>' columns).")

    plant_keys = sorted(plants.keys())

    orth_df = pd.read_csv(args.orthogroups, sep="\t", dtype=str)
    orth_og_col = get_orthogroup_col(orth_df)
    matrix_dir = Path(args.matrix).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.all_plants:
        failures = []
        for plant_key in plant_keys:
            plant_cols = plants[plant_key]
            plant_display_name = plant_name_map.get(normalize_species_col(plant_key), plant_key)
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
                    outlier_tissue_keys=outlier_tissue_keys,
                    matrix_dir=matrix_dir,
                    output_dir=output_dir,
                    output_png=None,
                    output_tsv=None,
                    plant_display_name=plant_display_name,
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
    plant_display_name = plant_name_map.get(normalize_species_col(plant_key), plant_key)
    render_plant(
        matrix_df=matrix_df,
        matrix_og_col=matrix_og_col,
        orth_df=orth_df,
        orth_og_col=orth_og_col,
        plant_key=plant_key,
        plant_cols=plant_cols,
        plant_protein_col=plant_protein_col,
        tissue_ontology_path=tissue_ontology_path,
        outlier_tissue_keys=outlier_tissue_keys,
        matrix_dir=matrix_dir,
        output_dir=output_dir,
        output_png=args.output_png,
        output_tsv=args.output_tsv,
        plant_display_name=plant_display_name,
        max_rows=args.max_rows,
        figscale=args.figscale,
        max_label_len=args.max_label_len,
    )


if __name__ == "__main__":
    main()
