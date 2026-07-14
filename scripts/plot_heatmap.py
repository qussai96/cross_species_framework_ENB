#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from plotly import graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage


def split_matrix_column(col: str) -> tuple[str, str | None]:
    if "_PO:" not in col:
        return col, None
    plant, suffix = col.split("_PO:", 1)
    return plant, "PO:" + suffix


def normalize_species_key(name: str) -> str:
    s = str(name).strip()
    for suffix in [".helixer.faa", ".helixer", ".faa", ".fasta"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def read_table_auto(path: str | Path, index_col=None) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", index_col=index_col, dtype=str if index_col is None else None)


def parse_fasta_headers(input_fasta: str | Path) -> set[str]:
    proteins = set()
    with open(input_fasta, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                proteins.add(line[1:].strip().split()[0])
    return proteins


def build_species_intensity(
    matrix_file: str | Path,
    orthogroups_file: str | Path,
    input_fasta: str | Path,
    metadata_file: str | Path,
) -> pd.DataFrame:
    meta = read_table_auto(metadata_file)
    if "FASTA" not in meta.columns or "Name" not in meta.columns:
        raise ValueError("metadata must contain columns: FASTA, Name")

    meta["ortho_col"] = meta["FASTA"].map(normalize_species_key)
    ortho_to_name = dict(zip(meta["ortho_col"], meta["Name"]))

    df = pd.read_csv(matrix_file, sep=None, engine="python", index_col=0)
    og = read_table_auto(orthogroups_file)
    if "Orthogroup" not in og.columns:
        raise ValueError("orthogroups table must contain Orthogroup column")

    protein_to_og: dict[str, str] = {}
    for _, row in og.iterrows():
        og_id = row["Orthogroup"]
        for col in og.columns[1:]:
            cell = row[col]
            if pd.notna(cell):
                for protein in [p.strip() for p in str(cell).split(",") if p.strip()]:
                    protein_to_og[protein] = og_id

    proteins_in_fasta = parse_fasta_headers(input_fasta)
    ogs_to_keep = {protein_to_og[p] for p in proteins_in_fasta if p in protein_to_og}

    keep_idx = []
    for idx in df.index:
        base_og = idx.split("|")[0] if "|" in idx else idx
        if base_og in ogs_to_keep:
            keep_idx.append(idx)

    df_filtered = df.loc[keep_idx]
    if df_filtered.empty:
        raise ValueError("No orthogroups found for proteins in input FASTA")

    species_cols: dict[str, list[str]] = {}
    for col in df_filtered.columns:
        species, _ = split_matrix_column(col)
        species_cols.setdefault(species, []).append(col)

    species_intensity = pd.DataFrame(index=df_filtered.index, columns=species_cols.keys(), dtype=float)
    for species, cols in species_cols.items():
        species_data = df_filtered[cols]
        species_median = species_data.replace(0, np.nan).median(axis=1)
        has_zeros = (species_data == 0).any(axis=1)
        species_median = species_median.where(~(species_median.isna() & has_zeros), 0)
        species_intensity[species] = species_median.values

    species_intensity.columns = species_intensity.columns.map(
        lambda x: ortho_to_name.get(normalize_species_key(x), x)
    )
    species_intensity = species_intensity.dropna(axis=1, how="all")
    if species_intensity.empty:
        raise ValueError("No data left after collapsing to per-species medians")

    return species_intensity


def compute_rowwise_z(species_intensity: pd.DataFrame, treat_zero_as_missing: bool = True) -> pd.DataFrame:
    z_input = species_intensity.copy()
    if treat_zero_as_missing:
        z_input = z_input.replace(0, np.nan)

    log_df = z_input.copy()
    mask = log_df.notna()
    log_df[mask] = np.log2(log_df[mask] + 1)

    row_means = log_df.mean(axis=1)
    row_stds = log_df.std(axis=1, ddof=0)
    z_df = log_df.sub(row_means, axis=0)
    z_df = z_df.div(row_stds.replace(0, np.nan), axis=0)
    z_df = z_df.where(log_df.notna())

    constant_rows = row_stds.eq(0)
    if constant_rows.any():
        z_df.loc[constant_rows] = z_df.loc[constant_rows].where(log_df.loc[constant_rows].isna(), 0)

    return z_df.replace([np.inf, -np.inf], np.nan)


def save_static_zscore_heatmap(
    species_intensity: pd.DataFrame,
    output_file: str | Path,
    title: str,
    min_species_presence: int = 2,
) -> None:
    present_count = (species_intensity > 0).sum(axis=1)
    filtered = species_intensity.loc[present_count >= min_species_presence]
    if filtered.empty:
        raise ValueError(f"No orthogroups present in >={min_species_presence} species")

    z_df = compute_rowwise_z(filtered, treat_zero_as_missing=True)
    plot_data = z_df.T
    mask = plot_data.isna()
    cluster_data = plot_data.fillna(0)

    cmap = LinearSegmentedColormap.from_list("blue_white_red", ["#0000FF", "#FFFFFF", "#FF0000"], N=100)
    cmap.set_bad(color="white")

    if plot_data.shape[0] < 2:
        plt.figure(figsize=(max(4, 0.3 * plot_data.shape[1]), max(4, 0.5 * plot_data.shape[0])))
        ax = sns.heatmap(
            cluster_data,
            mask=mask,
            cmap=cmap,
            center=0,
            xticklabels=False,
            yticklabels=True,
            cbar_kws={"label": "Z-score (log2 intensity), white=missing/undetected"},
        )
        x_labels = [c.split("|")[0] if "|" in c else c for c in plot_data.columns]
        ax.set_xticks(np.arange(len(x_labels)) + 0.5)
        ax.set_xticklabels(x_labels, rotation=90, ha="center", fontsize=6)
        ax.set_xlabel("Orthogroups")
        plt.subplots_adjust(bottom=0.35)
        plt.title(f"{title}\n({plot_data.shape[0]} Species, {plot_data.shape[1]} Orthogroups)")
        plt.savefig(output_file, dpi=150)
        plt.close()
        return

    width = min(20, 0.5 * plot_data.shape[1])
    height = max(6, 0.35 * plot_data.shape[0])

    g = sns.clustermap(
        cluster_data,
        mask=mask,
        method="ward",
        metric="euclidean",
        cmap=cmap,
        center=0,
        figsize=(width, height),
        row_cluster=True,
        col_cluster=True,
        xticklabels=False,
        yticklabels=True,
        cbar_kws={"label": "Z-score per orthogroup (log2 intensity)"},
        dendrogram_ratio=(0.15, 0.15),
    )

    ordered_labels = [c.split("|")[0] if "|" in c else c for c in g.data2d.columns]
    g.ax_heatmap.set_xticks(np.arange(len(ordered_labels)) + 0.5)
    g.ax_heatmap.set_xticklabels(ordered_labels, rotation=90, ha="center", fontsize=6)
    g.ax_heatmap.set_xlabel("Orthogroups")
    g.fig.subplots_adjust(bottom=0.30)
    g.fig.suptitle(
        f"{title}\n({plot_data.shape[1]} Orthogroups, {plot_data.shape[0]} Species)",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    g.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


def save_interactive_zscore_heatmap(species_intensity: pd.DataFrame, output_file: str | Path, title: str) -> None:
    z_df = compute_rowwise_z(species_intensity, treat_zero_as_missing=True)
    plot_data = z_df.T
    cluster_data = plot_data.fillna(0)

    row_linkage = linkage(cluster_data, method="ward", metric="euclidean")
    row_order = dendrogram(row_linkage, no_plot=True)["leaves"]

    col_linkage = linkage(cluster_data.T, method="ward", metric="euclidean")
    col_order = dendrogram(col_linkage, no_plot=True)["leaves"]

    clustered = plot_data.iloc[row_order, col_order]
    og_labels = [c.split("|")[0] if "|" in c else c for c in clustered.columns]

    hover_text = []
    for species in clustered.index:
        row_hover = []
        for og in clustered.columns:
            z_val = clustered.loc[species, og]
            base_og = og.split("|")[0] if "|" in og else og
            if pd.isna(z_val):
                row_hover.append(f"Species: {species}<br>Orthogroup: {base_og}<br>Z-score: Missing or undetected")
            else:
                row_hover.append(f"Species: {species}<br>Orthogroup: {base_og}<br>Z-score: {z_val:.2f}")
        hover_text.append(row_hover)

    fig = go.Figure(
        data=go.Heatmap(
            z=clustered.values,
            x=og_labels,
            y=clustered.index,
            hovertext=hover_text,
            hoverinfo="text",
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="Z-score<br>(log2 intensity)"),
            connectgaps=False,
        )
    )

    fig.update_layout(
        title=f"{title}<br>({clustered.shape[1]} Orthogroups, {clustered.shape[0]} Species)",
        xaxis_title="Orthogroups",
        yaxis_title="Species (clustered)",
        height=max(800, 25 * clustered.shape[0]),
        width=max(1200, 18 * clustered.shape[1]),
        margin=dict(l=80, r=40, t=90, b=260),
        xaxis=dict(
            type="category",
            tickmode="array",
            tickvals=og_labels,
            ticktext=og_labels,
            tickangle=90,
            tickfont=dict(size=8),
            side="bottom",
            automargin=True,
        ),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )

    fig.write_html(output_file)


def orthogroup_parts(label: str) -> tuple[str, str]:
    if "|" in label:
        og, desc = label.split("|", 1)
        return og, desc
    return label, ""


def load_search_protein_map(matched_file: str | Path) -> pd.DataFrame:
    df = read_table_auto(matched_file)
    if "Orthogroup" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Orthogroup"})
    if "search_protein_id" not in df.columns:
        if len(df.columns) < 2:
            raise ValueError("Matched orthogroup file must include search_protein_id column or at least two columns")
        second = [c for c in df.columns if c != "Orthogroup"][0]
        df = df.rename(columns={second: "search_protein_id"})
    return df[["Orthogroup", "search_protein_id"]].drop_duplicates()


def save_broadly_shared_tables(
    species_intensity: pd.DataFrame,
    search_protein_df: pd.DataFrame,
    out_dir: Path,
    min_broad_species: int = 10,
) -> dict[str, pd.DataFrame]:
    n_species = species_intensity.shape[1]
    presence_counts = (species_intensity > 0).sum(axis=1)

    configs = [
        ("all_species", 1.0),
        ("90pct_species", 0.90),
        ("75pct_species", 0.75),
    ]

    results: dict[str, pd.DataFrame] = {}
    for tag, fraction in configs:
        threshold = max(min_broad_species, int(np.ceil(fraction * n_species)))
        df = (
            species_intensity.loc[presence_counts >= threshold]
            .assign(
                Orthogroup=lambda x: x.index.map(lambda idx: orthogroup_parts(idx)[0]),
                Description=lambda x: x.index.map(lambda idx: orthogroup_parts(idx)[1]),
                Detected_species=lambda x: presence_counts.loc[x.index].astype(int),
                Species_fraction=lambda x: presence_counts.loc[x.index] / n_species,
            )
            [["Orthogroup", "Description", "Detected_species", "Species_fraction"]]
            .sort_values(["Detected_species", "Orthogroup"], ascending=[False, True])
        )
        df = df.merge(search_protein_df, on="Orthogroup", how="left")
        df.to_csv(out_dir / f"orthogroups_broadly_shared_{tag}.tsv", sep="\t", index=False)
        results[tag] = df

    return results


def save_enriched_table(species_intensity: pd.DataFrame, search_protein_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    presence_counts = (species_intensity > 0).sum(axis=1)
    rows = []
    for label in species_intensity.index:
        row = species_intensity.loc[label].replace(0, np.nan).dropna()
        if len(row) < 2:
            continue

        log_row = np.log2(row + 1)
        std = log_row.std(ddof=0)
        if std == 0 or np.isnan(std):
            continue

        z_row = (log_row - log_row.mean()) / std
        best_species = z_row.idxmax()
        top_z = float(z_row.max())
        second_z = float(z_row.nlargest(2).iloc[1]) if len(z_row) > 1 else np.nan
        orthogroup, description = orthogroup_parts(label)

        rows.append(
            {
                "Orthogroup": orthogroup,
                "Description": description,
                "Best_species": best_species,
                "Max_zscore": top_z,
                "Second_best_zscore": second_z,
                "Species_detected": int(presence_counts.loc[label]),
                "gt_2": top_z > 2.0,
                "gt_2_5": top_z > 2.5,
                "gt_3": top_z > 3.0,
            }
        )

    enriched_df = pd.DataFrame(rows)
    if enriched_df.empty:
        enriched_df = pd.DataFrame(
            columns=[
                "Orthogroup",
                "Description",
                "Best_species",
                "Max_zscore",
                "Second_best_zscore",
                "Species_detected",
                "gt_2",
                "gt_2_5",
                "gt_3",
                "search_protein_id",
            ]
        )
    else:
        enriched_df = enriched_df.sort_values(["Max_zscore", "Species_detected"], ascending=[False, False])
        enriched_df = enriched_df.merge(search_protein_df, on="Orthogroup", how="left")

    enriched_df.to_csv(out_dir / "orthogroups_enriched_thresholds.tsv", sep="\t", index=False)
    return enriched_df


def save_summary_barplot(species_intensity: pd.DataFrame, name: str, out_dir: Path) -> pd.DataFrame:
    presence_counts = (species_intensity > 0).sum(axis=1)
    n_species = species_intensity.shape[1]

    def count_broadly_shared(fraction: float) -> int:
        threshold = int(np.ceil(fraction * n_species))
        return int((presence_counts >= threshold).sum())

    max_zscores = []
    for orthogroup in species_intensity.index:
        row = species_intensity.loc[orthogroup].replace(0, np.nan).dropna()
        if len(row) < 2:
            continue
        log_row = np.log2(row + 1)
        std = log_row.std(ddof=0)
        if std == 0 or np.isnan(std):
            continue
        z_row = (log_row - log_row.mean()) / std
        max_zscores.append(float(z_row.max()))

    max_zscores = pd.Series(max_zscores, dtype=float)

    summary_counts = pd.DataFrame(
        {
            "Panel": [
                "Broadly shared OGs",
                "Broadly shared OGs",
                "Broadly shared OGs",
                "Enriched candidates OGs",
                "Enriched candidates OGs",
                "Enriched candidates OGs",
            ],
            "Category": [
                "all species",
                ">=90% species",
                ">=75% species",
                "> 2 z-score",
                "> 2.5 z-score",
                "> 3 z-score",
            ],
            "Count": [
                count_broadly_shared(1.0),
                count_broadly_shared(0.90),
                count_broadly_shared(0.75),
                int((max_zscores > 2.0).sum()),
                int((max_zscores > 2.5).sum()),
                int((max_zscores > 3.0).sum()),
            ],
        }
    )

    summary_counts.to_csv(out_dir / "orthogroup_summary_barplot_counts.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    panel_specs = [
        ("Broadly shared OGs", "#2c7fb8", ["all species", ">=90% species", ">=75% species"]),
        ("Enriched candidates OGs", "#d95f02", ["> 2 z-score", "> 2.5 z-score", "> 3 z-score"]),
    ]

    for ax, (panel_name, color, category_order) in zip(axes, panel_specs):
        panel_df = summary_counts.loc[summary_counts["Panel"] == panel_name].copy()
        panel_df["Category"] = pd.Categorical(panel_df["Category"], categories=category_order, ordered=True)
        panel_df = panel_df.sort_values("Category")
        sns.barplot(data=panel_df, x="Category", y="Count", color=color, ax=ax)
        ax.set_title(panel_name)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", padding=3, fontsize=9)

    axes[0].set_ylabel("Number of orthogroups")
    axes[1].set_ylabel("")
    fig.suptitle(f"Orthogroup summary counts {name}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "orthogroup_summary_barplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return summary_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert plot_heatmap notebook workflow to script outputs")
    parser.add_argument("--name", required=True, help="Run name label used in plot titles")
    parser.add_argument("--protein_orthogroup_po_matrix", required=True, help="Orthogroup x PO intensity matrix")
    parser.add_argument("--orthogroups", required=True, help="Assigned orthogroups table")
    parser.add_argument("--input_fasta", required=True, help="Input FASTA used to filter orthogroups")
    parser.add_argument("--metadata", required=True, help="Metadata TSV (must have FASTA and Name columns)")
    parser.add_argument(
        "--matched_orthogroups",
        required=True,
        help="Orthogroups matched table (e.g., Orthogroups_matched_to_Hormones_full_list.tsv)",
    )
    parser.add_argument("--output_dir", default=None, help="Output directory; defaults to matrix parent folder")
    parser.add_argument("--min_species_presence", type=int, default=2, help="Minimum species presence for z-score heatmap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    matrix_file = Path(args.protein_orthogroup_po_matrix)
    out_dir = Path(args.output_dir) if args.output_dir else matrix_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    species_intensity = build_species_intensity(
        matrix_file=args.protein_orthogroup_po_matrix,
        orthogroups_file=args.orthogroups,
        input_fasta=args.input_fasta,
        metadata_file=args.metadata,
    )

    species_intensity.to_csv(out_dir / "heatmap_original_values.tsv", sep="\t")

    title = f"Protein Orthogroups Expression Across Plants - {args.name}"
    save_static_zscore_heatmap(
        species_intensity=species_intensity,
        output_file=out_dir / "heatmap_z_score.png",
        title=title,
        min_species_presence=args.min_species_presence,
    )
    save_interactive_zscore_heatmap(
        species_intensity=species_intensity,
        output_file=out_dir / "heatmap_z_score_interactive.html",
        title=title,
    )

    search_protein_df = load_search_protein_map(args.matched_orthogroups)
    broad_tables = save_broadly_shared_tables(
        species_intensity=species_intensity,
        search_protein_df=search_protein_df,
        out_dir=out_dir,
    )
    enriched_df = save_enriched_table(
        species_intensity=species_intensity,
        search_protein_df=search_protein_df,
        out_dir=out_dir,
    )

    summary_counts = save_summary_barplot(species_intensity=species_intensity, name=args.name, out_dir=out_dir)

    print(f"Saved heatmaps to: {out_dir}")
    print(f"Saved barplot to: {out_dir / 'orthogroup_summary_barplot.png'}")
    print("Saved 6 TSV files:")
    print(f" - {out_dir / 'heatmap_original_values.tsv'}")
    print(f" - {out_dir / 'orthogroup_summary_barplot_counts.tsv'}")
    print(f" - {out_dir / 'orthogroups_broadly_shared_all_species.tsv'}")
    print(f" - {out_dir / 'orthogroups_broadly_shared_90pct_species.tsv'}")
    print(f" - {out_dir / 'orthogroups_broadly_shared_75pct_species.tsv'}")
    print(f" - {out_dir / 'orthogroups_enriched_thresholds.tsv'}")
    print(
        "Counts: "
        f"all={len(broad_tables['all_species'])}, "
        f"90%={len(broad_tables['90pct_species'])}, "
        f"75%={len(broad_tables['75pct_species'])}, "
        f"enriched_gt2={int((enriched_df['gt_2'] == True).sum()) if not enriched_df.empty else 0}, "
        f"enriched_gt2.5={int((enriched_df['gt_2_5'] == True).sum()) if not enriched_df.empty else 0}, "
        f"enriched_gt3={int((enriched_df['gt_3'] == True).sum()) if not enriched_df.empty else 0}, "
        f"summary_rows={len(summary_counts)}"
    )


if __name__ == "__main__":
    main()
