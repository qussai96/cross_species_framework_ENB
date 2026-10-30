#!/usr/bin/env python3
"""Generate tree-ordered species and species-tissue intensity heatmaps."""

from __future__ import annotations

import argparse
import csv
import gc
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm

from heatmap_outliers import outlier_column_indices

DEFAULT_TREE = Path("SpeciesTree_rooted.txt")
# Semantic heatmap palette: light gray is structural background/no orthogroup,
# darker gray is an orthogroup that is present but has missing/zero mcIBAQ, and
# red encodes positive abundance.
ABSENT_COLOR = "#E3E6E8"
UNDETECTED_COLOR = "#AEB4BA"
PLANT_SEPARATOR_COLOR = "#B7BCC1"


def normalize_key(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"(?:\.faa|\.fasta|\.fa)$", "", text)
    return re.sub(r"\.helixer$", "", text)


def split_matrix_column(column: str) -> tuple[str, str]:
    pnumber_match = re.fullmatch(r"(.+)_(P\d+)", str(column).strip(), re.IGNORECASE)
    if pnumber_match:
        return normalize_key(pnumber_match.group(1)), pnumber_match.group(2).upper()
    if "_PO:" not in column:
        return normalize_key(column), column
    species, suffix = column.split("_PO:", 1)
    _po_number, separator, tissue = suffix.partition("_")
    # PO identifiers are useful in the matrix but make the already dense tissue
    # axis unnecessarily long.  Display only the readable tissue name.
    label = tissue.replace("_", " ") if separator else "unknown tissue"
    return normalize_key(species), label


def pnumber_display_labels(
    ontology_path: Path | None, fallback_path: Path | None = None,
) -> dict[str, str]:
    """Return readable labels while keeping the unique p-number visible."""
    labels: dict[str, str] = {}
    if ontology_path and ontology_path.exists():
        table = pd.read_csv(ontology_path, sep="\t", dtype=str).fillna("")
        if {"pNumber", "TissueName"}.issubset(table.columns):
            for pnumber, tissue in table[["pNumber", "TissueName"]].itertuples(index=False):
                pnumber = str(pnumber).strip().upper()
                tissue = " ".join(str(tissue).replace("_", " ").split())
                if re.fullmatch(r"P\d+", pnumber) and tissue:
                    labels[pnumber] = f"{tissue} [{pnumber}]"
    if fallback_path and fallback_path.exists():
        table = pd.read_csv(fallback_path, sep="\t", dtype=str).fillna("")
        if {"p-number", "Tissue"}.issubset(table.columns):
            for pnumber, tissue in table[["p-number", "Tissue"]].itertuples(index=False):
                match = re.match(r"^(P\d+)", str(pnumber).strip(), re.IGNORECASE)
                tissue = " ".join(str(tissue).replace("_", " ").split())
                if match and tissue:
                    key = match.group(1).upper()
                    labels.setdefault(key, f"{tissue} [{key}]")
    return labels


def tree_order(tree_path: Path) -> list[str]:
    tree = Phylo.read(tree_path, "newick")
    return [normalize_key(clade.name) for clade in tree.get_terminals() if clade.name]


def class_order_rank(value: object) -> int:
    label = str(value).strip().casefold()
    if label in {"monocot", "monocots"}:
        return 0
    if label in {"eudicot", "eudicots", "dicot", "dicots"}:
        return 1
    if label in {"magnoliid", "magnoliids"}:
        return 2
    return 3


def metadata_species_maps(path: Path) -> tuple[dict[str, list[str]], dict[str, str], pd.DataFrame]:
    """Map a reference leaf to one or more independently sampled crops."""
    metadata = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    metadata["reference_key"] = metadata["FASTA"].map(normalize_key)
    shared = metadata["reference_key"].duplicated(keep=False)
    metadata["matrix_key"] = metadata["reference_key"]
    metadata.loc[shared, "matrix_key"] = metadata.loc[shared, "Name"].map(normalize_key)
    if metadata["matrix_key"].duplicated().any():
        rows = metadata.loc[metadata["matrix_key"].duplicated(keep=False),
                            ["Name", "FASTA", "matrix_key"]]
        raise ValueError("Metadata produces duplicate matrix keys:\n" + rows.to_string(index=False))
    displays_by_reference = (metadata.groupby("reference_key", sort=False)["matrix_key"]
                             .apply(list).to_dict())
    reference_by_display = dict(zip(metadata["matrix_key"], metadata["reference_key"]))
    return displays_by_reference, reference_by_display, metadata.set_index("matrix_key")


def read_membership(path: Path, orthogroups: list[str], species: list[str],
                    reference_by_display: dict[str, str]) -> np.ndarray:
    displays_by_reference: dict[str, list[str]] = {}
    for name in species:
        displays_by_reference.setdefault(reference_by_display.get(name, name), []).append(name)
    membership = {og: {name: False for name in species} for og in orthogroups}
    seen: set[str] = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            og = str(row.get("Orthogroup", "")).strip()
            if og not in membership:
                continue
            seen.add(og)
            for header, value in row.items():
                for name in displays_by_reference.get(normalize_key(header), []):
                    membership[og][name] = bool(str(value or "").strip())
    missing = [og for og in orthogroups if og not in seen]
    if missing:
        raise ValueError("Orthogroups missing from membership table: " + ", ".join(missing))
    return np.array([[membership[og][name] for name in species] for og in orthogroups])


def bold_query(value: str) -> str:
    """Render a query name in bold MathText."""
    escaped = value.replace("_", r"\_")
    return rf"$\mathbf{{{escaped}}}$"


def read_best_hits(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        first_line = handle.readline().rstrip("\n")
    fields = first_line.split("\t")
    if {"query", "target_protein", "bitscore"}.issubset(fields):
        table = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        if "status" in table.columns:
            table = table.loc[table["status"].eq("assigned")]
        table["_bitscore"] = pd.to_numeric(table["bitscore"], errors="coerce")
        table = table.sort_values(
            ["query", "_bitscore"], ascending=[True, False]
        ).drop_duplicates("query")
        return dict(zip(table["query"], table["target_protein"]))
    hits: dict[str, str] = {}
    with path.open(newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                hits.setdefault(row[0].strip(), row[1].strip())
    return hits


def read_hit_descriptions(path: Path | None, hit_ids: set[str]) -> dict[str, str]:
    if path is None or not hit_ids:
        return {}
    # Description-table IDs omit the species prefix found in DIAMOND targets.
    # Index every possible underscore-delimited suffix of the small hit set,
    # then stream the large table once.
    hits_by_suffix: dict[str, list[str]] = {}
    for hit_id in hit_ids:
        pieces = hit_id.split("_")
        for index in range(1, len(pieces)):
            hits_by_suffix.setdefault("_".join(pieces[index:]), []).append(hit_id)
    descriptions: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            fasta_id = str(row.get("fasta_id", "")).strip()
            for hit_id in hits_by_suffix.get(fasta_id, []):
                if hit_id.endswith("_" + fasta_id):
                    descriptions[hit_id] = str(row.get("description", "")).strip()
    return descriptions


def read_row_labels(path: Path, orthogroups: list[str], hits_path: Path | None,
                    descriptions_path: Path | None) -> tuple[list[str], list[int]]:
    table = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    query_by_og = table.drop_duplicates("Orthogroup").set_index("Orthogroup")[
        "search_protein_id"
    ].to_dict()
    missing = [og for og in orthogroups if not query_by_og.get(og, "").strip()]
    if missing:
        raise ValueError("Query-protein labels missing for: " + ", ".join(missing))
    labels: list[str] = []
    row_indices: list[int] = []
    for index, og in enumerate(orthogroups):
        queries = [value.strip() for value in query_by_og[og].split(",") if value.strip()]
        for query in queries:
            labels.append(query)
            row_indices.append(index)
    return labels, row_indices


def load_taxonomy(metadata: pd.DataFrame, species: list[str]) -> pd.DataFrame:
    rows = []
    for name in species:
        row = metadata.loc[normalize_key(name)] if normalize_key(name) in metadata.index else None
        class_name = row["Class"] if row is not None else "unknown"
        if class_name.casefold() == "monocots":
            class_name = "monocots"
        elif class_name.casefold() in {"eudicots", "dicots"}:
            class_name = "eudicots"
        elif class_name.casefold() in {"magnoliid", "magnoliids"}:
            class_name = "magnoliids"
        rows.append({
            "Name": row["Name"] if row is not None else name,
            "Class": class_name or "unknown",
            "Order": (row["Order"] if row is not None else "") or "unknown",
            "Family": (row["Family"] if row is not None else "") or "unknown",
        })
    return pd.DataFrame(rows)


def draw_annotation_bar(ax, values: list[str], label: str) -> None:
    """Draw a labelled categorical strip without a large external legend."""
    categories = [value for value in dict.fromkeys(values) if value]
    lookup = {value: index for index, value in enumerate(categories)}
    gap_code = len(categories)
    codes = np.array([[lookup.get(value, gap_code) for value in values]])
    palette = {"Species": "Pastel1", "Class": "Set2", "Order": "tab20",
               "Family": "gist_ncar"}[label]
    colors = plt.get_cmap(palette)(np.linspace(0, 1, max(1, len(categories))))
    colors = np.vstack([colors, np.array([1.0, 1.0, 1.0, 1.0])])
    ax.imshow(codes, aspect="auto", interpolation="nearest", cmap=ListedColormap(colors),
              vmin=-0.5, vmax=max(0.5, gap_code + 0.5))
    ax.set_yticks([0], [label], fontsize=9, fontweight="bold")
    ax.set_xticks([])
    ax.tick_params(length=0)
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            if start and values[start] and values[start - 1]:
                ax.axvline(start - 0.5, color="white",
                           linewidth=4.5 if label == "Class" else 1.4)
            span = index - start
            # Even narrow taxonomy boxes need an identifier.  Long names are
            # shortened below, so do not suppress small order/family groups.
            minimum_label_span = 3
            if values[start] and span >= minimum_label_span:
                available = max(4, int(span * 1.7))
                text = values[start]
                if len(text) > available:
                    text = text[:max(1, available - 3)] + "..."
                relative_width = span / len(values)
                if relative_width >= 0.10:
                    font_size = 9.0
                elif relative_width >= 0.04:
                    font_size = 8.0
                else:
                    font_size = 7.0
                red, green, blue = colors[lookup[values[start]], :3]
                # WCAG relative luminance gives annotation labels enough
                # contrast on both the pale and dark categorical colours.
                channels = np.array([red, green, blue])
                channels = np.where(channels <= 0.04045, channels / 12.92,
                                    ((channels + 0.055) / 1.055) ** 2.4)
                luminance = float(np.dot(channels, [0.2126, 0.7152, 0.0722]))
                text_color = "#FFFFFF" if luminance < 0.36 else "#111827"
                ax.text((start + index - 1) / 2, 0, text, ha="center", va="center",
                        fontsize=font_size, fontweight="bold", color=text_color,
                        clip_on=True)
            start = index
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_tree(ax, tree_path: Path, centers: dict[str, float], limits: tuple[float, float]) -> None:
    tree = Phylo.read(tree_path, "newick")
    depths = tree.depths()
    if max(depths.values()) == 0:
        depths = tree.depths(unit_branch_lengths=True)
    max_depth = max(depths.values()) or 1.0
    cache = {}
    segments = []

    def coordinates(clade):
        if clade in cache:
            return cache[clade]
        if clade.is_terminal():
            key = normalize_key(clade.name)
            # Align every terminal tip to the same baseline.  Unequal terminal
            # branch depths otherwise make short right-edge branches appear
            # truncated even though the leaf is present.
            cache[clade] = ((centers[key], max_depth) if key in centers else (None, None))
            return cache[clade]
        children = [coordinates(child) for child in clade.clades]
        children = [(x, y) for x, y in children if x is not None]
        if not children:
            cache[clade] = (None, None)
            return cache[clade]
        x = sum(position for position, _ in children) / len(children)
        y = depths[clade]
        for child_x, child_y in children:
            segments.append([(x, y), (child_x, y)])
            segments.append([(child_x, y), (child_x, child_y)])
        cache[clade] = (x, y)
        return cache[clade]

    coordinates(tree.root)
    # Draw all branches as one clipped collection.  Individual unclipped lines
    # can disappear near the right edge of very wide (30k+ pixel) raster images.
    ax.add_collection(LineCollection(segments, colors="#374151", linewidths=0.72,
                                     clip_on=True, zorder=2))
    # Mark terminal coordinates explicitly.  Without endpoint markers, the two
    # rightmost short terminal branches can look clipped at the plot boundary.
    for terminal in tree.get_terminals():
        x, y = coordinates(terminal)
        if x is not None:
            ax.plot(x, y, marker="o", markersize=1.8, color="#374151",
                    markeredgewidth=0, clip_on=True, zorder=3)
    ax.set_xlim(*limits)
    ax.set_ylim(max_depth * 1.02, -max_depth * 0.02)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_heatmap(*, values: np.ndarray, present: np.ndarray, row_labels: list[str],
                 x_labels: list[str], species: list[str], taxonomy: pd.DataFrame,
                 column_species: list[str], tree_path: Path, output: Path,
                 tissue_level: bool, vmax: float,
                 reference_by_display: dict[str, str],
                 outlier_indices: set[int] | None = None,
                 output_dpi: int = 160,
                 values_are_minmax: bool = False,
                 values_are_zscores: bool = False,
                 show_tree_taxonomy: bool = True,
                 undetected: np.ndarray | None = None,
                 absent_color: str = ABSENT_COLOR,
                 undetected_color: str = UNDETECTED_COLOR,
                 show_zscore_state_legend: bool = False,
                 show_raw_scale_legend: bool = False) -> None:
    by_species = taxonomy.copy()
    by_species.index = species

    # At tissue resolution, insert real spacer columns between plant
    # blocks.  A larger spacer marks class transitions.  This makes each tissue
    # label visually traceable to its own heatmap block instead of relying on a
    # line drawn over adjacent data cells.
    if tissue_level:
        expanded_values: list[np.ndarray] = []
        expanded_present: list[np.ndarray] = []
        expanded_labels: list[str] = []
        expanded_column_species: list[str] = []
        expanded_outliers: set[int] = set()
        expanded_undetected: list[np.ndarray] = []
        previous_species = None
        for current_species in species:
            positions = [i for i, value in enumerate(column_species)
                         if value == current_species]
            if not positions:
                continue
            if previous_species is not None:
                previous_class = by_species.loc[previous_species, "Class"]
                current_class = by_species.loc[current_species, "Class"]
                gap_width = 6 if current_class != previous_class else 2
                for _ in range(gap_width):
                    expanded_values.append(np.zeros(values.shape[0]))
                    expanded_present.append(np.zeros(values.shape[0], dtype=bool))
                    expanded_labels.append("")
                    expanded_column_species.append("")
                    if undetected is not None:
                        expanded_undetected.append(np.zeros(values.shape[0], dtype=bool))
            for position in positions:
                new_index = len(expanded_values)
                expanded_values.append(values[:, position])
                expanded_present.append(present[:, position])
                expanded_labels.append(x_labels[position])
                expanded_column_species.append(current_species)
                if undetected is not None:
                    expanded_undetected.append(undetected[:, position])
                if position in (outlier_indices or set()):
                    expanded_outliers.add(new_index)
            previous_species = current_species
        values = np.column_stack(expanded_values)
        present = np.column_stack(expanded_present)
        x_labels = expanded_labels
        column_species = expanded_column_species
        outlier_indices = expanded_outliers
        if undetected is not None:
            undetected = np.column_stack(expanded_undetected)

    transformed = values if (values_are_minmax or values_are_zscores) else np.log10(values + 1.0)
    masked = np.ma.masked_where(~present, transformed)
    if values_are_zscores:
        cmap = LinearSegmentedColormap.from_list(
            "mcibaq_zscore", ["#2166AC", "#DCEAF4", "#FFFFFF", "#F6B2AE", "#76000A"]
        )
        finite = transformed[present & np.isfinite(transformed)]
        zlimit = max(1.0, min(3.0, float(np.max(np.abs(finite))) if finite.size else 1.0))
        color_norm = TwoSlopeNorm(vmin=-zlimit, vcenter=0.0, vmax=zlimit)
    elif show_raw_scale_legend:
        cmap = LinearSegmentedColormap.from_list(
            "mcibaq_positive_red",
            ["#FCE8E6", "#F6B2AE", "#E6655F", "#BC292B", "#76000A"],
        )
        positive_logged = transformed[present & (values > 0)]
        raw_vmin = float(positive_logged.min()) if positive_logged.size else 0.0
        raw_vmax = float(positive_logged.max()) if positive_logged.size else max(1.0, vmax)
        if raw_vmax <= raw_vmin:
            raw_vmax = raw_vmin + 1.0
        color_norm = Normalize(vmin=raw_vmin, vmax=raw_vmax)
    else:
        cmap = LinearSegmentedColormap.from_list(
            "mcibaq_white_to_red",
            [
                (0.0, "#FFFFFF"),
                (1e-6, "#FCE8E6"),
                (0.30, "#F6B2AE"),
                (0.60, "#E6655F"),
                (0.82, "#BC292B"),
                (1.0, "#76000A"),
            ],
        )
        color_norm = Normalize(vmin=0, vmax=vmax)
    cmap.set_bad(absent_color)
    nrows, ncols = values.shape
    if tissue_level and not show_tree_taxonomy:
        # Per-plant panels have far fewer columns, so spend the available image
        # budget on row separation. At 180 DPI the 96-inch cap remains safely
        # below the 32,768-pixel renderer limit while keeping very large query
        # sets substantially more readable.
        height = max(20.0, min(96.0, nrows * 0.07 + 7.0))
    else:
        height = max(20.0, min(36.0, nrows * 0.12 + 7.0))
    if tissue_level:
        # Reserve additional horizontal room for every plant block.  This keeps
        # adjacent plants visually distinct even when the matrix has thousands
        # of tissue columns.
        # Keep the raster below the 32,768-pixel coordinate boundary used by
        # parts of the rendering stack.  Exceeding it caused the final tree
        # branches to vanish even though their tissue columns were present.
        # The real spacer columns below still preserve the wider plant gaps.
        width = max(24.0, min(128.0, ncols * 0.06 + len(species) * 0.14 + 10.0))
        annotation_names = ["Class", "Order", "Family"]
    else:
        width = max(18.0, min(30.0, ncols * 0.17 + 8.0))
        annotation_names = ["Class", "Order", "Family"]

    annotations = {
        "Species": [by_species.loc[name, "Name"] if name else "" for name in column_species],
        "Class": [by_species.loc[name, "Class"] if name else "" for name in column_species],
        "Order": [by_species.loc[name, "Order"] if name else "" for name in column_species],
        "Family": [by_species.loc[name, "Family"] if name else "" for name in column_species],
    }
    # Spacer columns should create distance between tissue blocks, not break a
    # taxonomy box shared by the plants on both sides.  Fill only gaps whose
    # left and right categories are identical; gaps at real category changes
    # remain white separators.
    for rank, values_for_rank in annotations.items():
        start = 0
        while start < len(values_for_rank):
            if values_for_rank[start]:
                start += 1
                continue
            end = start
            while end < len(values_for_rank) and not values_for_rank[end]:
                end += 1
            left = values_for_rank[start - 1] if start > 0 else ""
            right = values_for_rank[end] if end < len(values_for_rank) else ""
            if left and left == right:
                values_for_rank[start:end] = [left] * (end - start)
            start = end
    fig = plt.figure(figsize=(width, height))
    ratios = ([1.15] + [0.16] * len(annotation_names) + [5.2]
              if show_tree_taxonomy else [5.2])
    grid = fig.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0.045)
    centers = {}
    for name in species:
        positions = [i for i, value in enumerate(column_species) if value == name]
        centers[name] = (positions[0] + positions[-1]) / 2
    tree_centers: dict[str, list[float]] = {}
    for name, center in centers.items():
        tree_centers.setdefault(reference_by_display.get(name, name), []).append(center)
    if show_tree_taxonomy:
        draw_tree(fig.add_subplot(grid[0]), tree_path,
                  {name: sum(values) / len(values) for name, values in tree_centers.items()},
                  (-0.5, ncols - 0.5))
        for row, rank in enumerate(annotation_names, start=1):
            annotation_ax = fig.add_subplot(grid[row])
            draw_annotation_bar(annotation_ax, annotations[rank], rank)
    heat_ax = fig.add_subplot(grid[-1])
    if undetected is not None:
        # Draw structural states independently from the continuous z-score
        # layer so an undetected zero can never be confused with z=0.
        state = np.ma.masked_all(values.shape, dtype=float)
        state[~present] = 0.0
        state[undetected] = 1.0
        state_cmap = ListedColormap([absent_color, undetected_color])
        state_cmap.set_bad((1, 1, 1, 0))
        heat_ax.imshow(state, aspect="auto", interpolation="nearest",
                       cmap=state_cmap, vmin=0, vmax=1, rasterized=True)
        masked = np.ma.masked_where((~present) | undetected, transformed)
        cmap.set_bad((1, 1, 1, 0))
    heat_ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap,
                   norm=color_norm, rasterized=True)
    # Allocate label size from the vertical pitch of the *heatmap panel*, not
    # the full figure.  A fixed lower bound previously made labels overlap in
    # large categories once the 36-inch figure-height cap was reached.  Keep a
    # small renderer-safe floor, but never force normal labels to be larger
    # than roughly 78% of their row pitch.  Figure dimensions and panel ratios
    # remain unchanged.
    heat_panel_fraction = ratios[-1] / sum(ratios)
    row_pitch_points = height * 72.0 * heat_panel_fraction / max(1, nrows)
    row_font_size = max(
        0.8,
        min(9.0 if not show_tree_taxonomy else 8.0, row_pitch_points * 0.78),
    )
    heat_ax.set_yticks(np.arange(nrows), row_labels, fontsize=row_font_size)
    for tick in heat_ax.get_yticklabels():
        tick.set_fontweight("bold")
    heat_ax.tick_params(axis="y", length=0, pad=2)
    heat_ax.set_ylabel("")
    heat_ax.set_xticks(np.arange(ncols))
    if tissue_level and not show_tree_taxonomy:
        x_tick_font_size = 11.0
    else:
        x_tick_font_size = 4.2 if tissue_level else 6.5
    heat_ax.set_xticklabels(x_labels, rotation=90, fontsize=x_tick_font_size)
    if tissue_level and not show_tree_taxonomy:
        for tick in heat_ax.get_xticklabels():
            tick.set_fontweight("bold")
    for index, tick in enumerate(heat_ax.get_xticklabels()):
        if index in (outlier_indices or set()):
            tick.set_color("blue")
            tick.set_fontweight("bold")
    # At tissue resolution, reserve the first x-axis tier for the species name
    # and push the individual tissue labels into the tier below it.
    heat_ax.tick_params(
        axis="x", length=0,
        pad=(4 if tissue_level and not show_tree_taxonomy else
             30 if tissue_level else 1),
    )
    if tissue_level:
        gap_positions = [index for index, name in enumerate(column_species) if not name]
        for index in gap_positions:
            # A cool structural separator cannot be mistaken for either a
            # dark-gray undetected cell or a positive red abundance cell.
            heat_ax.axvspan(index - 0.5, index + 0.5,
                            color=PLANT_SEPARATOR_COLOR,
                            linewidth=0, zorder=10)
        # Put species names below their tissue groups, rather than alongside the
        # tree.  A short rule makes the extent of every species block explicit.
        for name in species:
            positions = [i for i, value in enumerate(column_species) if value == name]
            start, end = positions[0], positions[-1]
            center = (start + end) / 2
            display_name = by_species.loc[name, "Name"]
            if not show_tree_taxonomy:
                heat_ax.set_title(display_name, fontsize=26, fontweight="bold", pad=16)
                continue
            heat_ax.plot([start - 0.45, end + 0.45], [-0.006, -0.006],
                         transform=heat_ax.get_xaxis_transform(), color="#4B5563",
                         linewidth=0.45, clip_on=False)
            if display_name == "BlackPepper" and len(positions) <= 5:
                display_name = "B.Pepper"
            # Very narrow species blocks cannot accommodate a horizontal name.
            # Rotate only those labels instead of allowing adjacent names to merge.
            rotate_name = len(positions) < max(3, len(display_name) / 3.0)
            heat_ax.text(center, -0.012, display_name,
                         transform=heat_ax.get_xaxis_transform(), ha="center", va="top",
                         fontsize=9.0 if not show_tree_taxonomy else 7.2,
                         fontweight="bold", color="#111827",
                         rotation=90 if rotate_name else 0, clip_on=False)
    for spine in heat_ax.spines.values():
        spine.set_visible(False)

    # Reserve approximately 4.8 physical inches for long query labels and the
    # upper-left legend regardless of whether the canvas is the 30-inch species
    # view or the 128-inch tissue view. A fixed fractional minimum previously
    # made the tissue plot's white margin several times wider in real units.
    left = min(0.32, max(0.04, 4.8 / width))
    per_plant_panel = tissue_level and not show_tree_taxonomy
    fig.subplots_adjust(left=left, right=0.975,
                        bottom=0.20 if tissue_level else 0.15,
                        top=0.94 if per_plant_panel else 0.99)
    if show_zscore_state_legend or show_raw_scale_legend:
        # Overlay the legend in the otherwise unused upper-left space beside
        # the tree.  add_axes does not participate in layout, so no plot panel
        # is resized or shifted.
        legend_width = max(0.03, left - 0.018)
        legend_ax = fig.add_axes([0.008, 0.825, legend_width, 0.145], zorder=20)
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)
        legend_ax.axis("off")
        if show_raw_scale_legend:
            legend_title = "Shared scale\nlog10(1 + mcIBAQ)"
        else:
            legend_title = ("Positive mcIBAQ\nglobal min-max"
                            if values_are_minmax else "Positive mcIBAQ\nglobal z-score")
        legend_ax.text(0.0, 0.96, legend_title,
                       ha="left", va="top", fontsize=7.0, fontweight="bold",
                       linespacing=0.95)
        if show_raw_scale_legend:
            gradient = np.linspace(color_norm.vmin, color_norm.vmax, 256).reshape(-1, 1)
            legend_norm = color_norm
        else:
            gradient = np.linspace(-1, 1, 256).reshape(-1, 1)
            legend_norm = (Normalize(vmin=-1, vmax=1) if values_are_minmax
                           else TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
        legend_ax.imshow(gradient, extent=(0.0, 0.075, 0.27, 0.72),
                         aspect="auto", cmap=cmap,
                         norm=legend_norm,
                         interpolation="nearest", origin="lower")
        if show_raw_scale_legend:
            bottom_legend_text = f"{color_norm.vmin:.1f}"
            center_legend_text = f"{(color_norm.vmin + color_norm.vmax) / 2:.1f}"
            top_legend_text = f"{color_norm.vmax:.1f}"
        else:
            bottom_legend_text = "Lowest positive" if values_are_minmax else "Below mean"
            center_legend_text = "" if values_are_minmax else "Mean"
            top_legend_text = "Highest positive" if values_are_minmax else "Above mean"
        legend_ax.text(0.10, 0.27, bottom_legend_text, ha="left", va="center",
                       fontsize=6.5, fontweight="bold")
        legend_ax.text(0.10, 0.495, center_legend_text, ha="left", va="center",
                       fontsize=6.5, fontweight="bold")
        legend_ax.text(0.10, 0.72, top_legend_text, ha="left", va="center",
                       fontsize=6.5, fontweight="bold")
        legend_ax.add_patch(plt.Rectangle((0.42, 0.52), 0.075, 0.12,
                                          color=absent_color, ec="#737980", lw=0.4))
        legend_ax.text(0.52, 0.58, "Orthogroup absent", ha="left", va="center",
                       fontsize=6.5, fontweight="bold")
        legend_ax.add_patch(plt.Rectangle((0.42, 0.31), 0.075, 0.12,
                                          color=undetected_color, ec="#A7ADB3", lw=0.4))
        legend_ax.text(0.52, 0.37, "OG present, mcIBAQ = 0", ha="left", va="center",
                       fontsize=6.5, fontweight="bold")
    image_output = output if output.suffix.lower() in {".jpg", ".jpeg", ".png"} else output.with_suffix(".jpg")
    image_output.parent.mkdir(parents=True, exist_ok=True)
    save_options = {
        "dpi": output_dpi,
        "facecolor": "white",
    }
    if image_output.suffix.lower() in {".jpg", ".jpeg"}:
        save_options["pil_kwargs"] = {"quality": 95, "subsampling": 0}
    # Long query-protein labels can extend beyond the fixed species-panel
    # margin.  Expanding the saved bounding box preserves every label without
    # reducing the heatmap width.  Avoid this for the 32k-pixel tissue panel.
    if not tissue_level:
        save_options.update(bbox_inches="tight", pad_inches=0.12)
    fig.savefig(image_output, **save_options)
    # A workflow can render more than 100 large per-plant figures in one
    # process.  Explicitly release artists and collect cyclic references so
    # resident memory does not grow until Slurm/the kernel kills the job.
    fig.clear()
    plt.close(fig)
    gc.collect()
    print(f"Wrote {image_output}")


def row_minmax(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Scale each row to [0, 1] across proteome-present cells only."""
    normalized = np.zeros_like(values, dtype=float)
    for row_index in range(values.shape[0]):
        valid = present[row_index]
        if not np.any(valid):
            continue
        row_values = values[row_index, valid].astype(float)
        row_min = float(row_values.min())
        row_max = float(row_values.max())
        if row_max > row_min:
            normalized[row_index, valid] = (row_values - row_min) / (row_max - row_min)
    return normalized


def whole_positive_zscore(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Globally z-score only positive log10(mcIBAQ + 1) matrix cells."""
    normalized = np.zeros_like(values, dtype=float)
    logged = np.log10(values.astype(float) + 1.0)
    valid = present & (values > 0)
    valid_values = logged[valid]
    if valid_values.size:
        standard_deviation = float(valid_values.std())
        if standard_deviation > 0:
            normalized[valid] = (
                valid_values - float(valid_values.mean())
            ) / standard_deviation
    return normalized


def whole_positive_minmax(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Globally min-max scale positive log10(mcIBAQ + 1) matrix cells."""
    normalized = np.zeros_like(values, dtype=float)
    logged = np.log10(values.astype(float) + 1.0)
    valid = present & (values > 0)
    valid_values = logged[valid]
    if valid_values.size:
        minimum = float(valid_values.min())
        maximum = float(valid_values.max())
        if maximum > minimum:
            normalized[valid] = (valid_values - minimum) / (maximum - minimum)
    return normalized


def default_species_output(tissue_output: Path) -> Path:
    stem = tissue_output.stem
    stem = stem.replace("_all_tissues", "_species") if "_all_tissues" in stem else stem + "_species"
    return tissue_output.with_name(stem + tissue_output.suffix)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot query orthogroups at species and species-tissue resolution.")
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--tree", default=DEFAULT_TREE, type=Path)
    parser.add_argument("--orthogroups", required=True, type=Path)
    parser.add_argument("--hits", type=Path,
                        help="Headerless DIAMOND best-hit TSV used in row labels")
    parser.add_argument("--protein-descriptions", type=Path,
                        help="TSV with species, fasta_id, and description columns")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--outliers", type=Path,
                        help="QC TSV whose Outlier=1 tissues are labelled blue")
    parser.add_argument("--tissue-ontology", type=Path,
                        help="Tissue ontology TSV used to resolve outlier pNumbers")
    parser.add_argument("--pnumber-tissues", type=Path,
                        help="Fallback p-number-to-tissue display table")
    parser.add_argument("--output", required=True, type=Path,
                        help="Species-tissue image (species image is derived automatically)")
    parser.add_argument("--species-output", type=Path,
                        help="Optional explicit species-level image path")
    parser.add_argument("--tissue-dpi", type=int, default=250,
                        help="Resolution of the all-tissues image (default: 250 DPI)")
    parser.add_argument("--no-minmax", action="store_true",
                        help="Do not generate row-wise min-max companion heatmaps")
    parser.add_argument("--other-heatmaps-dir", type=Path,
                        help="Write all-tissue min-max and z-score companion heatmaps here")
    parser.add_argument("--per-plant-dir", type=Path,
                        help="Write one raw-intensity tissue heatmap per sampled plant here")
    parser.add_argument("--per-plant-dpi", type=int, default=180,
                        help="Resolution for individual plant heatmaps (default: 180 DPI)")
    parser.add_argument("--per-plant-only", action="store_true",
                        help="Generate only the per-plant heatmaps")
    parser.add_argument("--species-zscore-only", action="store_true",
                        help="Generate only the global positive-mcIBAQ species z-score heatmap")
    parser.add_argument("--tissue-minmax-only", action="store_true",
                        help="Generate only the global positive-mcIBAQ all-tissues min-max heatmap")
    parser.add_argument("--raw-tissue-only", action="store_true",
                        help="Generate only the raw all-tissues heatmap with its shared-scale legend")
    args = parser.parse_args()

    matrix = pd.read_csv(args.matrix, sep="\t", dtype=str)
    id_column = matrix.columns[0]
    matrix["_orthogroup"] = (matrix[id_column].fillna("").astype(str)
                              .str.split("|", n=1).str[0].str.strip())
    if matrix["_orthogroup"].duplicated().any():
        duplicates = matrix.loc[matrix["_orthogroup"].duplicated(), "_orthogroup"].tolist()
        raise ValueError("Duplicate orthogroups in matrix: " + ", ".join(duplicates))
    matrix = matrix.set_index("_orthogroup")
    orthogroups = matrix.index.tolist()
    row_labels, expanded_row_indices = read_row_labels(
        args.orthogroups, orthogroups, args.hits, args.protein_descriptions
    )

    matrix_columns = [column for column in matrix.columns if column != id_column]
    matrix_by_species: dict[str, list[str]] = {}
    display_labels = pnumber_display_labels(args.tissue_ontology, args.pnumber_tissues)
    tissue_label_by_column = {}
    for column in matrix_columns:
        name, tissue_label = split_matrix_column(column)
        matrix_by_species.setdefault(name, []).append(column)
        tissue_label_by_column[column] = display_labels.get(tissue_label, tissue_label)
    numeric = matrix[matrix_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    ordered_tree_species = tree_order(args.tree)
    if not ordered_tree_species:
        raise ValueError("Species tree has no named leaves")
    displays_by_reference, reference_by_display, metadata = metadata_species_maps(args.metadata)
    expanded_tree_species = [display for reference in ordered_tree_species
                             for display in displays_by_reference.get(reference, [reference])]
    tree_position = {name: index for index, name in enumerate(expanded_tree_species)}
    expanded_tree_species.sort(
        key=lambda name: (
            class_order_rank(metadata.loc[name, "Class"])
            if name in metadata.index else 3,
            tree_position[name],
        )
    )
    # Retain every sampled species represented by matrix columns.  A species
    # with no detected query intensity is a valid zero/undetected result and
    # must not disappear from a presence heatmap.
    ordered_species = [name for name in expanded_tree_species
                       if name in matrix_by_species]
    if not ordered_species:
        raise ValueError("No species with proteomics data remain after filtering")
    no_columns = [name for name in expanded_tree_species if name not in matrix_by_species]
    all_zero = [name for name in expanded_tree_species if name in matrix_by_species
                and not (numeric[matrix_by_species[name]].to_numpy() > 0).any()]
    print(f"Retained {len(ordered_species)} species with matrix columns; omitted "
          f"{len(no_columns)} without matrix columns. "
          f"Retained {len(all_zero)} all-zero/undetected species.")
    if all_zero:
        print("All-zero species retained: " + ", ".join(all_zero))

    taxonomy = load_taxonomy(metadata, ordered_species)
    membership = read_membership(
        args.orthogroups, orthogroups, ordered_species, reference_by_display
    )
    species_values = np.column_stack([
        numeric[matrix_by_species[name]].max(axis=1).to_numpy() for name in ordered_species
    ])
    if np.any((~membership) & (species_values > 0)):
        raise ValueError("Found positive intensity for an orthogroup absent from the proteome")

    tissue_columns = [column for name in ordered_species for column in matrix_by_species[name]]
    tissue_outlier_indices = outlier_column_indices(
        tissue_columns, args.outliers, args.tissue_ontology, args.metadata
    )
    print(f"Outlier tissue columns labelled blue: {len(tissue_outlier_indices)}")
    # Do not filter tissue columns by their individual values: once a species has
    # at least one positive tissue, its complete tissue block is displayed.
    tissue_species = [split_matrix_column(column)[0] for column in tissue_columns]
    tissue_values = numeric[tissue_columns].to_numpy()
    species_index = {name: index for index, name in enumerate(ordered_species)}
    tissue_present = np.column_stack([
        membership[:, species_index[name]] for name in tissue_species
    ])
    if np.any((~tissue_present) & (tissue_values > 0)):
        raise ValueError("Found positive tissue intensity for an orthogroup absent from the proteome")
    species_values = species_values[expanded_row_indices, :]
    membership = membership[expanded_row_indices, :]
    tissue_values = tissue_values[expanded_row_indices, :]
    tissue_present = tissue_present[expanded_row_indices, :]
    print(f"Expanded {len(orthogroups)} orthogroups to {len(row_labels)} query-protein rows.")
    print(f"Displaying all {len(tissue_columns)} tissue columns for the retained species.")

    vmax = max(1.0, float(np.log10(max(float(tissue_values.max()),
                                      float(species_values.max())) + 1.0)))
    species_output = args.species_output or default_species_output(args.output)
    category = args.output.name.split("_heatmap_", 1)[0]
    species_zscore_output = (
        args.other_heatmaps_dir / f"{category}_all_plants_tissues_z_score.png"
        if args.other_heatmaps_dir else None
    )
    tissue_minmax_output = (
        args.other_heatmaps_dir / f"{category}_all_plants_tissues_min_max.jpg"
        if args.other_heatmaps_dir else None
    )
    if args.tissue_minmax_only or args.species_zscore_only or args.raw_tissue_only:
        if args.other_heatmaps_dir is None:
            raise ValueError("normalized-only modes require --other-heatmaps-dir")
        args.other_heatmaps_dir.mkdir(parents=True, exist_ok=True)
        if args.tissue_minmax_only:
            tissue_minmax = whole_positive_minmax(tissue_values, tissue_present)
            draw_heatmap(
                values=tissue_minmax, present=tissue_present, row_labels=row_labels,
                x_labels=[tissue_label_by_column[column] for column in tissue_columns],
                species=ordered_species, taxonomy=taxonomy, column_species=tissue_species,
                tree_path=args.tree, output=tissue_minmax_output,
                tissue_level=True, vmax=1.0,
                reference_by_display=reference_by_display,
                outlier_indices=tissue_outlier_indices, output_dpi=args.tissue_dpi,
                values_are_minmax=True,
                undetected=tissue_present & (tissue_values <= 0),
                absent_color=ABSENT_COLOR, show_zscore_state_legend=True)
        if args.species_zscore_only:
            species_zscores = whole_positive_zscore(species_values, membership)
            draw_heatmap(
                values=species_zscores, present=membership, row_labels=row_labels,
                x_labels=taxonomy["Name"].tolist(), species=ordered_species,
                taxonomy=taxonomy, column_species=ordered_species,
                tree_path=args.tree, output=species_zscore_output,
                tissue_level=False, vmax=1.0,
                reference_by_display=reference_by_display, outlier_indices=set(),
                output_dpi=160, values_are_zscores=True,
                undetected=membership & (species_values <= 0),
                absent_color=ABSENT_COLOR, show_zscore_state_legend=True)
        if args.raw_tissue_only:
            draw_heatmap(
                values=tissue_values, present=tissue_present, row_labels=row_labels,
                x_labels=[tissue_label_by_column[column] for column in tissue_columns],
                species=ordered_species, taxonomy=taxonomy, column_species=tissue_species,
                tree_path=args.tree, output=args.output,
                tissue_level=True, vmax=vmax,
                reference_by_display=reference_by_display,
                outlier_indices=tissue_outlier_indices, output_dpi=args.tissue_dpi,
                undetected=tissue_present & (tissue_values <= 0),
                absent_color=ABSENT_COLOR, show_raw_scale_legend=True)
        return
    if not args.per_plant_only:
        draw_heatmap(
            values=species_values, present=membership, row_labels=row_labels,
            x_labels=taxonomy["Name"].tolist(), species=ordered_species, taxonomy=taxonomy,
            column_species=ordered_species, tree_path=args.tree, output=species_output,
            tissue_level=False, vmax=vmax,
            reference_by_display=reference_by_display, outlier_indices=set(), output_dpi=160,
            undetected=membership & (species_values <= 0))
        draw_heatmap(
            values=tissue_values, present=tissue_present, row_labels=row_labels,
            x_labels=[tissue_label_by_column[column] for column in tissue_columns],
            species=ordered_species, taxonomy=taxonomy, column_species=tissue_species,
            tree_path=args.tree, output=args.output,
            tissue_level=True, vmax=vmax,
            reference_by_display=reference_by_display,
            outlier_indices=tissue_outlier_indices, output_dpi=args.tissue_dpi,
            undetected=tissue_present & (tissue_values <= 0),
            absent_color=ABSENT_COLOR, show_raw_scale_legend=True)

    if args.per_plant_dir:
        args.per_plant_dir.mkdir(parents=True, exist_ok=True)
        for plant in ordered_species:
            positions = [index for index, name in enumerate(tissue_species) if name == plant]
            if not positions:
                continue
            plant_values = tissue_values[:, positions]
            plant_present = tissue_present[:, positions]
            plant_outliers = {
                local_index for local_index, source_index in enumerate(positions)
                if source_index in tissue_outlier_indices
            }
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(metadata.loc[plant, "Name"]))
            draw_heatmap(
                values=plant_values, present=plant_present, row_labels=row_labels,
                x_labels=[tissue_label_by_column[tissue_columns[index]] for index in positions],
                species=[plant], taxonomy=taxonomy.iloc[[species_index[plant]]],
                column_species=[plant] * len(positions), tree_path=args.tree,
                output=args.per_plant_dir / f"{safe_name}.jpg",
                tissue_level=True, vmax=vmax,
                reference_by_display=reference_by_display,
                outlier_indices=plant_outliers, output_dpi=args.per_plant_dpi,
                show_tree_taxonomy=False,
                undetected=plant_present & (plant_values <= 0))

        if args.per_plant_only:
            return

    if args.no_minmax and not args.other_heatmaps_dir:
        return

    species_minmax = row_minmax(species_values, membership)
    tissue_minmax = whole_positive_minmax(tissue_values, tissue_present)
    if args.other_heatmaps_dir:
        args.other_heatmaps_dir.mkdir(parents=True, exist_ok=True)
        draw_heatmap(
            values=tissue_minmax, present=tissue_present, row_labels=row_labels,
            x_labels=[tissue_label_by_column[column] for column in tissue_columns],
            species=ordered_species, taxonomy=taxonomy, column_species=tissue_species,
            tree_path=args.tree, output=tissue_minmax_output,
            tissue_level=True, vmax=1.0,
            reference_by_display=reference_by_display,
            outlier_indices=tissue_outlier_indices, output_dpi=args.tissue_dpi,
            values_are_minmax=True,
            undetected=tissue_present & (tissue_values <= 0),
            absent_color=ABSENT_COLOR, show_zscore_state_legend=True)
        species_zscores = whole_positive_zscore(species_values, membership)
        draw_heatmap(
            values=species_zscores, present=membership, row_labels=row_labels,
            x_labels=taxonomy["Name"].tolist(), species=ordered_species,
            taxonomy=taxonomy, column_species=ordered_species,
            tree_path=args.tree, output=species_zscore_output,
            tissue_level=False, vmax=1.0,
            reference_by_display=reference_by_display, outlier_indices=set(),
            output_dpi=160, values_are_zscores=True,
            undetected=membership & (species_values <= 0),
            absent_color=ABSENT_COLOR, show_zscore_state_legend=True)
        if args.no_minmax:
            return
    draw_heatmap(
        values=species_minmax, present=membership, row_labels=row_labels,
        x_labels=taxonomy["Name"].tolist(), species=ordered_species, taxonomy=taxonomy,
        column_species=ordered_species, tree_path=args.tree,
        output=species_output.with_name(species_output.stem + "_minmax.jpg"),
        tissue_level=False, vmax=1.0,
        reference_by_display=reference_by_display, outlier_indices=set(), output_dpi=160,
        values_are_minmax=True,
        undetected=membership & (species_values <= 0))
    draw_heatmap(
        values=tissue_minmax, present=tissue_present, row_labels=row_labels,
        x_labels=[tissue_label_by_column[column] for column in tissue_columns],
        species=ordered_species, taxonomy=taxonomy, column_species=tissue_species,
        tree_path=args.tree,
        output=args.output.with_name(args.output.stem + "_minmax.jpg"),
        tissue_level=True, vmax=1.0,
        reference_by_display=reference_by_display,
        outlier_indices=tissue_outlier_indices, output_dpi=args.tissue_dpi,
        values_are_minmax=True,
        undetected=tissue_present & (tissue_values <= 0))


if __name__ == "__main__":
    main()
