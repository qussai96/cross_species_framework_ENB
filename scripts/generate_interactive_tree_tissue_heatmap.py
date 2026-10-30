#!/usr/bin/env python3
"""Generate an offline interactive tree-ordered all-tissues mcIBAQ heatmap."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from Bio import Phylo

from generate_tree_ordered_presence_heatmap import (
    ABSENT_COLOR,
    UNDETECTED_COLOR,
    class_order_rank,
    load_taxonomy,
    metadata_species_maps,
    normalize_key,
    outlier_column_indices,
    pnumber_display_labels,
    read_best_hits,
    read_hit_descriptions,
    read_membership,
    split_matrix_column,
    tree_order,
)
from protein_orthogroup_pnumber_matrix import (
    normalize_pnumber,
)


def fasta_lengths(path: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    identifier: str | None = None
    length = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    lengths[identifier] = length
                fields = line[1:].split()
                if not fields:
                    raise ValueError(f"Empty FASTA header at line {line_number}: {path}")
                identifier = fields[0]
                if identifier in lengths:
                    raise ValueError(f"Duplicate FASTA identifier: {identifier}")
                length = 0
            else:
                if identifier is None:
                    raise ValueError(f"Sequence before first FASTA header: {path}")
                length += len("".join(line.split()))
    if identifier is not None:
        lengths[identifier] = length
    return lengths


def row_metadata(orthogroup_path: Path, orthogroups: list[str], hits_path: Path,
                 descriptions_path: Path, query_fasta: Path,
                 merge_audit_path: Path | None = None,
                 ) -> tuple[list[dict[str, object]], list[int]]:
    table = pd.read_csv(
        orthogroup_path, sep="\t", dtype=str,
        usecols=["Orthogroup", "search_protein_id"],
    ).fillna("")
    query_by_og = table.drop_duplicates("Orthogroup").set_index("Orthogroup")[
        "search_protein_id"
    ].to_dict()
    missing = [og for og in orthogroups if not query_by_og.get(og, "").strip()]
    if missing:
        raise ValueError("Query-protein metadata missing for: " + ", ".join(missing))

    best_hits = read_best_hits(hits_path)
    descriptions = read_hit_descriptions(descriptions_path, set(best_hits.values()))
    lengths = fasta_lengths(query_fasta)
    displayed_orthogroups: dict[str, str] = {}
    if merge_audit_path is not None:
        audit = pd.read_csv(merge_audit_path, sep="\t", dtype=str).fillna("")
        required = {"synthetic_orthogroup", "family_mapped_orthogroups"}
        if not required.issubset(audit.columns):
            raise ValueError(f"Merge audit must contain {sorted(required)}")
        displayed_orthogroups = dict(zip(
            audit["synthetic_orthogroup"], audit["family_mapped_orthogroups"]
        ))
    records: list[dict[str, object]] = []
    row_indices: list[int] = []
    for matrix_index, orthogroup in enumerate(orthogroups):
        queries = [item.strip() for item in query_by_og[orthogroup].split(",")
                   if item.strip()]
        for query in queries:
            hit = best_hits.get(query, "no_hit")
            records.append({
                "query": query,
                "hit": hit,
                "description": descriptions.get(hit, "description unavailable"),
                "orthogroup": displayed_orthogroups.get(orthogroup, orthogroup),
                "query_length": lengths.get(query),
            })
            row_indices.append(matrix_index)
    return records, row_indices


def expand_tissue_blocks(values: np.ndarray, present: np.ndarray,
                         columns: list[str], column_species: list[str],
                         tissue_labels: list[str], species: list[str],
                         taxonomy: pd.DataFrame, outliers: set[int]
                         ) -> tuple[np.ndarray, np.ndarray, list[str], list[str],
                                    list[str], set[int]]:
    value_columns: list[np.ndarray] = []
    present_columns: list[np.ndarray] = []
    expanded_columns: list[str] = []
    expanded_species: list[str] = []
    expanded_labels: list[str] = []
    expanded_outliers: set[int] = set()
    previous_species: str | None = None
    gap_number = 0

    for current_species in species:
        positions = [index for index, name in enumerate(column_species)
                     if name == current_species]
        if not positions:
            continue
        if previous_species is not None:
            previous_class = taxonomy.loc[previous_species, "Class"]
            current_class = taxonomy.loc[current_species, "Class"]
            gap_width = 6 if previous_class != current_class else 2
            for _ in range(gap_width):
                value_columns.append(np.full(values.shape[0], np.nan))
                present_columns.append(np.zeros(values.shape[0], dtype=bool))
                expanded_columns.append(f"__species_gap_{gap_number}")
                expanded_species.append("")
                expanded_labels.append("")
                gap_number += 1
        for position in positions:
            new_index = len(value_columns)
            value_columns.append(values[:, position])
            present_columns.append(present[:, position])
            expanded_columns.append(columns[position])
            expanded_species.append(current_species)
            expanded_labels.append(tissue_labels[position])
            if position in outliers:
                expanded_outliers.add(new_index)
        previous_species = current_species

    return (
        np.column_stack(value_columns),
        np.column_stack(present_columns),
        expanded_columns,
        expanded_species,
        expanded_labels,
        expanded_outliers,
    )


def fill_internal_taxonomy_gaps(values: list[str]) -> list[str]:
    result = values.copy()
    start = 0
    while start < len(result):
        if result[start]:
            start += 1
            continue
        end = start
        while end < len(result) and not result[end]:
            end += 1
        left = result[start - 1] if start > 0 else ""
        right = result[end] if end < len(result) else ""
        if left and left == right:
            result[start:end] = [left] * (end - start)
        start = end
    return result


def discrete_colorscale(colors: list[str]) -> list[list[object]]:
    count = len(colors)
    scale: list[list[object]] = []
    for index, color in enumerate(colors):
        scale.append([index / count, color])
        scale.append([(index + 1) / count, color])
    return scale


def taxonomy_trace(values: list[str], x_values: list[str], rank: str) -> go.Heatmap:
    categories = [value for value in dict.fromkeys(values) if value]
    lookup = {value: index for index, value in enumerate(categories)}
    gap_code = len(categories)
    codes = [[lookup.get(value, gap_code) for value in values]]
    cmap_name = {"Class": "Set2", "Order": "tab20", "Family": "gist_ncar"}[rank]
    cmap = colormaps[cmap_name]
    colors = [to_hex(cmap(position)) for position in
              np.linspace(0, 1, max(1, len(categories)), endpoint=True)]
    colors.append("#FFFFFF")
    return go.Heatmap(
        z=codes,
        x=x_values,
        y=[rank],
        text=[values],
        colorscale=discrete_colorscale(colors),
        zmin=-0.5,
        zmax=len(colors) - 0.5,
        showscale=False,
        hovertemplate=f"<b>{rank}:</b> %{{text}}<extra></extra>",
    )


def taxonomy_text_trace(values: list[str], x_values: list[str]) -> go.Scatter:
    centers: list[str] = []
    labels: list[str] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            span = index - start
            if values[start] and span >= 3:
                maximum = max(4, int(span * 1.7))
                label = values[start]
                if len(label) > maximum:
                    label = label[:max(1, maximum - 3)] + "..."
                centers.append(x_values[(start + index - 1) // 2])
                labels.append(label)
            start = index
    return go.Scatter(
        x=centers, y=[0] * len(centers), text=labels, mode="text",
        textfont=dict(size=9, color="#111827"), hoverinfo="skip",
        showlegend=False,
    )


def tree_trace(tree_path: Path, centers: dict[str, float]) -> go.Scatter:
    tree = Phylo.read(tree_path, "newick")
    depths = tree.depths()
    if max(depths.values()) == 0:
        depths = tree.depths(unit_branch_lengths=True)
    maximum_depth = max(depths.values()) or 1.0
    cache: dict[object, tuple[float | None, float | None]] = {}
    x_values: list[float | None] = []
    y_values: list[float | None] = []

    def coordinates(clade) -> tuple[float | None, float | None]:
        if clade in cache:
            return cache[clade]
        if clade.is_terminal():
            key = normalize_key(clade.name)
            cache[clade] = ((centers[key], maximum_depth)
                            if key in centers else (None, None))
            return cache[clade]
        children = [coordinates(child) for child in clade.clades]
        children = [(x, y) for x, y in children if x is not None and y is not None]
        if not children:
            cache[clade] = (None, None)
            return cache[clade]
        x_position = sum(x for x, _ in children) / len(children)
        y_position = depths[clade]
        for child_x, child_y in children:
            x_values.extend([x_position, child_x, None, child_x, child_x, None])
            y_values.extend([y_position, y_position, None, y_position, child_y, None])
        cache[clade] = (x_position, y_position)
        return cache[clade]

    coordinates(tree.root)
    return go.Scatter(
        x=x_values, y=y_values, mode="lines",
        line=dict(color="#374151", width=1), hoverinfo="skip", showlegend=False,
    )


def heatmap_colorscale(vmax: float, absent_value: float) -> tuple[list[list[object]], float]:
    zero_position = (0.0 - absent_value) / (vmax - absent_value)
    scale: list[list[object]] = [
        [0.0, ABSENT_COLOR],
        [max(0.0, zero_position - 1e-7), ABSENT_COLOR],
        [zero_position, UNDETECTED_COLOR],
        [min(1.0, zero_position + 1e-7), "#FCE8E6"],
    ]
    for fraction, color in (
        (0.30, "#F6B2AE"),
        (0.60, "#E6655F"),
        (0.82, "#BC292B"),
        (1.00, "#76000A"),
    ):
        position = zero_position + fraction * (1.0 - zero_position)
        scale.append([float(position), color])
    return scale, zero_position


def sample_name(plant: object, tissue: object) -> str:
    """Return a compact Plant: Tissue label without reference-accession text."""
    plant_text = re.sub(r"\s+", " ", str(plant)).strip()
    tissue_text = re.sub(r"\s+", " ", str(tissue)).strip()
    return f"{plant_text}: {tissue_text}"


def unique_invisible_label(label: str, index: int) -> str:
    """Make a categorical Plotly value unique without changing visible text."""
    binary = format(index, "012b")
    suffix = "".join("\u200b" if digit == "0" else "\u200c" for digit in binary)
    return label + "\u2063" + suffix


def split_protein_ids(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).replace(";", ",").split(",")
            if item.strip() and item.strip().lower() not in {"nan", "na", "none"}]


def truncate_hover_text(value: object, limit: int = 70) -> str:
    """Keep verbose UniProt fields readable in a compact hover panel."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > limit:
        text = text[:limit - 3].rstrip() + "..."
    return html.escape(text) if text else "Not available"


def uniprot_by_refseq(path: Path) -> dict[str, tuple[str, str, str]]:
    """Index UniProt metadata by RefSeq, using the first-column Entry ID."""
    table = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    required = {"Entry", "Protein names", "Gene Names", "RefSeq"}
    if not required.issubset(table.columns):
        raise ValueError(
            f"UniProt mapping table must contain {sorted(required)}; got {table.columns.tolist()}"
        )
    mapping: dict[str, list[tuple[str, str, str]]] = {}
    for row in table[["Entry", "Protein names", "Gene Names", "RefSeq"]].itertuples(index=False):
        # Entry is the first column of the UniProt mapping file and is the
        # accession shown in the hover panel (not "Entry Name").
        entry = str(row[0]).strip()
        if not entry:
            continue
        protein_name = truncate_hover_text(row[1])
        gene_name = truncate_hover_text(row[2])
        for refseq in str(row[3]).split(";"):
            refseq = refseq.strip()
            if refseq:
                mapping.setdefault(refseq, []).append((entry, protein_name, gene_name))
    # A RefSeq accession can occasionally have redundant mappings.  Select a
    # reproducible first Entry accession rather than mixing metadata records.
    return {refseq: sorted(entries, key=lambda item: item[0])[0]
            for refseq, entries in mapping.items()}


def mapped_uniprot_fields(source: str, mapping: dict[str, tuple[str, str, str]]
                          ) -> tuple[str, str, str]:
    """Resolve an embedded RefSeq ID to UniProt Entry, protein, and gene."""
    if source in {"No ortholog was found for this protein in this species",
                  "Not applicable (orthogroup absent from reference proteome)"}:
        return ("Not applicable", "Not applicable", "Not applicable")
    if source in {"No detected mcIBAQ (0)", "source protein unavailable", ""}:
        return ("Not applicable", "Not applicable", "Not applicable")
    refseq_ids = re.findall(r"(?:XP|NP|YP|WP)_\d+(?:\.\d+)?", source)
    for refseq in refseq_ids:
        if refseq in mapping:
            return mapping[refseq]
    return ("No mapped UniProt Entry", "Not applicable", "Not applicable")


def protein_sources_by_cell(
    assigned_path: Path,
    intensities_dir: Path,
    tissue_ontology: Path,
    pnumber_tissues: Path,
    ordered_species: list[str],
    reference_by_display: dict[str, str],
    metadata: pd.DataFrame,
    matrix_columns_by_species: dict[str, list[str]],
    orthogroups: list[str],
    aggregation: str,
) -> dict[tuple[str, str], str]:
    """Return the plant protein accession(s) responsible for each matrix cell."""
    if aggregation not in {"max", "sum", "median"}:
        raise ValueError("aggregation must be max, sum, or median")

    assigned = pd.read_csv(assigned_path, sep="\t", dtype=str).fillna("")
    assigned = assigned.drop_duplicates("Orthogroup").set_index("Orthogroup")
    reference_headers = {
        normalize_key(column): column for column in assigned.columns
        if column not in {"search_protein_id"}
    }
    targets: dict[str, dict[str, list[str]]] = {}
    for species in ordered_species:
        reference = reference_by_display.get(species, species)
        header = reference_headers.get(reference)
        if header is None:
            continue
        protein_to_ogs: dict[str, list[str]] = {}
        for og in orthogroups:
            if og not in assigned.index:
                continue
            for protein in split_protein_ids(assigned.at[og, header]):
                protein_to_ogs.setdefault(protein, []).append(og)
        targets[species] = protein_to_ogs
    display_by_input_crop = {
        normalize_key(row["Name"]): matrix_key
        for matrix_key, row in metadata.iterrows()
    }

    observations: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for path in sorted(intensities_dir.glob("*_mcIBAQ.intensities.tsv")):
        input_crop = normalize_key(path.name.replace("_mcIBAQ.intensities.tsv", ""))
        species = display_by_input_crop.get(input_crop, input_crop)
        protein_to_ogs = targets.get(species, {})
        if not protein_to_ogs:
            continue
        expected_columns = set(matrix_columns_by_species[species])
        for chunk in pd.read_csv(path, sep="\t", chunksize=200_000, low_memory=False):
            protein_column = "Protein ID" if "Protein ID" in chunk.columns else "Protein"
            if protein_column not in chunk.columns:
                continue
            # Primary protein IDs are what the matrix builder uses whenever
            # they include a known orthogroup protein.
            selected = chunk[chunk[protein_column].astype(str).isin(protein_to_ogs)]
            if selected.empty:
                continue
            intensity_columns = [
                column for column in chunk.columns
                if normalize_pnumber(column) and str(normalize_pnumber(column)).startswith("P")
            ]
            column_map: dict[str, str] = {}
            for column in intensity_columns:
                pnumber = normalize_pnumber(column)
                matrix_column = f"{species}_{pnumber}"
                if matrix_column in expected_columns:
                    column_map[column] = matrix_column
            if not column_map:
                continue
            for row in selected[[protein_column, *column_map]].itertuples(index=False, name=None):
                protein = str(row[0]).strip()
                orthogroups_for_protein = protein_to_ogs.get(protein, [])
                if not orthogroups_for_protein:
                    continue
                for offset, column in enumerate(column_map, start=1):
                    value = pd.to_numeric(row[offset], errors="coerce")
                    if pd.isna(value):
                        continue
                    for orthogroup in orthogroups_for_protein:
                        observations.setdefault((orthogroup, column_map[column]), []).append(
                            (protein, float(value))
                        )

    sources: dict[tuple[str, str], str] = {}
    for key, values in observations.items():
        positive = [(protein, value) for protein, value in values if value > 0]
        if not positive:
            continue
        if aggregation == "max":
            maximum = max(value for _, value in positive)
            # A heatmap cell displays one maximum value.  In the rare case of
            # an exact tie, use the alphabetically first protein consistently
            # rather than implying that all tied proteins were plotted.
            proteins = [min(protein for protein, value in positive
                            if np.isclose(value, maximum, rtol=1e-9, atol=1e-12))]
        else:
            proteins = sorted({protein for protein, _ in positive})
        sources[key] = ", ".join(proteins)
    return sources


def orthogroup_proteins_by_species(
    assigned_path: Path,
    ordered_species: list[str],
    reference_by_display: dict[str, str],
    orthogroups: list[str],
) -> dict[tuple[str, str], str]:
    """Return reference-proteome proteins assigned to every OG/species pair.

    This is deliberately independent of peptide detection: a pale-yellow
    (zero mcIBAQ) cell still has an orthologous protein and should show it.
    """
    assigned = pd.read_csv(assigned_path, sep="\t", dtype=str).fillna("")
    assigned = assigned.drop_duplicates("Orthogroup").set_index("Orthogroup")
    headers = {
        normalize_key(column): column for column in assigned.columns
        if column != "search_protein_id"
    }
    proteins: dict[tuple[str, str], str] = {}
    for species in ordered_species:
        header = headers.get(reference_by_display.get(species, species))
        if header is None:
            continue
        for orthogroup in orthogroups:
            if orthogroup not in assigned.index:
                continue
            members = split_protein_ids(assigned.at[orthogroup, header])
            if members:
                members = sorted(set(members))
                # With no proteomic detection there is no intensity-based
                # choice.  Show the first ortholog while signaling that the
                # orthogroup may contain additional protein IDs.
                suffix = ", etc." if len(members) > 1 else ""
                proteins[(orthogroup, species)] = members[0] + suffix
    return proteins


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an offline interactive version of the tree-ordered all-tissues heatmap."
    )
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--orthogroups", required=True, type=Path)
    parser.add_argument("--hits", required=True, type=Path)
    parser.add_argument("--query-fasta", required=True, type=Path)
    parser.add_argument("--protein-descriptions", required=True, type=Path)
    parser.add_argument("--intensities", required=True, type=Path,
                        help="Directory of per-plant *_mcIBAQ.intensities.tsv files")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--outliers", required=True, type=Path)
    parser.add_argument("--tissue-ontology", required=True, type=Path)
    parser.add_argument("--pnumber-tissues", required=True, type=Path)
    parser.add_argument("--uniprot-mapping", required=True, type=Path,
                        help="TSV with UniProt Entry and RefSeq columns")
    parser.add_argument("--aggregation", choices=["max", "sum", "median"], default="max")
    parser.add_argument("--merge-audit", type=Path,
                        help="Optional synthetic-row to merged-orthogroups audit TSV")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help=("Maximum query-protein rows per HTML file. Outputs above this "
              "limit are divided as evenly as possible among the minimum "
              "number of chunks and named "
              "interactive_heatmap_<category>_<start>-<end>.html; "
              "use 0 to disable chunking."),
    )
    args = parser.parse_args()
    if args.chunk_size < 0:
        parser.error("--chunk-size must be zero or a positive integer")

    matrix = pd.read_csv(args.matrix, sep="\t", dtype=str)
    id_column = matrix.columns[0]
    matrix["_orthogroup"] = (matrix[id_column].fillna("").astype(str)
                              .str.split("|", n=1).str[0].str.strip())
    if matrix["_orthogroup"].duplicated().any():
        raise ValueError("Matrix contains duplicate orthogroups")
    matrix = matrix.set_index("_orthogroup")
    orthogroups = matrix.index.tolist()
    records, expanded_row_indices = row_metadata(
        args.orthogroups, orthogroups, args.hits,
        args.protein_descriptions, args.query_fasta, args.merge_audit,
    )

    matrix_columns = [column for column in matrix.columns if column != id_column]
    display_labels = pnumber_display_labels(args.tissue_ontology, args.pnumber_tissues)
    matrix_by_species: dict[str, list[str]] = {}
    tissue_by_column: dict[str, str] = {}
    for column in matrix_columns:
        species_name, tissue = split_matrix_column(column)
        matrix_by_species.setdefault(species_name, []).append(column)
        tissue_by_column[column] = display_labels.get(tissue, tissue)
    numeric = matrix[matrix_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    displays_by_reference, reference_by_display, metadata = metadata_species_maps(args.metadata)
    ordered_tree_species = tree_order(args.tree)
    expanded_tree_species = [display for reference in ordered_tree_species
                             for display in displays_by_reference.get(reference, [reference])]
    tree_position = {name: index for index, name in enumerate(expanded_tree_species)}
    expanded_tree_species.sort(key=lambda name: (
        class_order_rank(metadata.loc[name, "Class"]) if name in metadata.index else 3,
        tree_position[name],
    ))
    ordered_species = [name for name in expanded_tree_species if name in matrix_by_species]
    if not ordered_species:
        raise ValueError("No species with matrix columns remain after ordering")

    taxonomy = load_taxonomy(metadata, ordered_species)
    taxonomy.index = ordered_species
    membership = read_membership(
        args.orthogroups, orthogroups, ordered_species, reference_by_display,
    )
    tissue_columns = [column for name in ordered_species for column in matrix_by_species[name]]
    tissue_species = [split_matrix_column(column)[0] for column in tissue_columns]
    tissue_labels = [tissue_by_column[column] for column in tissue_columns]
    tissue_values = numeric[tissue_columns].to_numpy()[expanded_row_indices, :]
    species_index = {name: index for index, name in enumerate(ordered_species)}
    tissue_present = np.column_stack([
        membership[:, species_index[name]] for name in tissue_species
    ])[expanded_row_indices, :]
    if np.any((~tissue_present) & (tissue_values > 0)):
        raise ValueError("Positive intensity found where the orthogroup is absent from the proteome")
    outlier_indices = outlier_column_indices(
        tissue_columns, args.outliers, args.tissue_ontology, args.metadata,
    )
    source_lookup = protein_sources_by_cell(
        args.orthogroups, args.intensities, args.tissue_ontology,
        args.pnumber_tissues, ordered_species, reference_by_display,
        metadata, matrix_by_species, orthogroups, args.aggregation,
    )
    ortholog_lookup = orthogroup_proteins_by_species(
        args.orthogroups, ordered_species, reference_by_display, orthogroups,
    )
    uniprot_mapping = uniprot_by_refseq(args.uniprot_mapping)
    source_before_expansion = np.full(
        tissue_values.shape, "No detected mcIBAQ (0)", dtype=object
    )
    for row_index, original_index in enumerate(expanded_row_indices):
        orthogroup = orthogroups[original_index]
        for column_index, column in enumerate(tissue_columns):
            if not tissue_present[row_index, column_index]:
                source_before_expansion[row_index, column_index] = (
                    "No ortholog was found for this protein in this species"
                )
            elif tissue_values[row_index, column_index] > 0:
                source_before_expansion[row_index, column_index] = source_lookup.get(
                    (orthogroup, column), "source protein unavailable"
                )
            else:
                # The orthogroup is present but was not quantified in this
                # tissue.  Show its actual reference protein ID(s), rather
                # than treating this as an absent ortholog.
                species_name = split_matrix_column(column)[0]
                source_before_expansion[row_index, column_index] = ortholog_lookup.get(
                    (orthogroup, species_name), "source protein unavailable"
                )
    (tissue_values, tissue_present, x_values, expanded_species,
     tissue_labels, outlier_indices) = expand_tissue_blocks(
        tissue_values, tissue_present, tissue_columns, tissue_species,
        tissue_labels, ordered_species, taxonomy, outlier_indices,
    )
    original_column_index = {column: index for index, column in enumerate(tissue_columns)}
    source_values = np.full(tissue_values.shape, "", dtype=object)
    uniprot_entry_values = np.full(tissue_values.shape, "", dtype=object)
    uniprot_protein_values = np.full(tissue_values.shape, "", dtype=object)
    uniprot_gene_values = np.full(tissue_values.shape, "", dtype=object)
    intensity_values = np.full(tissue_values.shape, "", dtype=object)
    for column_index, column in enumerate(x_values):
        if column in original_column_index:
            original_column = original_column_index[column]
            source_values[:, column_index] = source_before_expansion[:, original_column]
            uniprot_fields = [
                mapped_uniprot_fields(source, uniprot_mapping)
                for source in source_values[:, column_index]
            ]
            uniprot_entry_values[:, column_index] = [item[0] for item in uniprot_fields]
            uniprot_protein_values[:, column_index] = [item[1] for item in uniprot_fields]
            uniprot_gene_values[:, column_index] = [item[2] for item in uniprot_fields]
            for row_index in range(tissue_values.shape[0]):
                if not tissue_present[row_index, column_index]:
                    intensity_values[row_index, column_index] = (
                        "Not applicable"
                    )
                elif tissue_values[row_index, column_index] == 0:
                    intensity_values[row_index, column_index] = (
                        "Not identified with proteomics"
                    )
                else:
                    intensity_values[row_index, column_index] = (
                        f"{tissue_values[row_index, column_index]:.6g}"
                    )

    sample_values: list[str] = []
    family_values: list[str] = []
    order_values: list[str] = []
    for index, species_name in enumerate(expanded_species):
        if species_name:
            sample = sample_name(taxonomy.loc[species_name, "Name"], tissue_labels[index])
            if index in outlier_indices:
                sample = f"<span style='color:blue'><b>{html.escape(sample)}</b></span>"
            else:
                sample = html.escape(sample)
            label = f"<b>Sample:</b> {sample}"
            family = html.escape(str(taxonomy.loc[species_name, "Family"]))
            order = html.escape(str(taxonomy.loc[species_name, "Order"]))
        else:
            label = ""
            family = ""
            order = ""
        sample_values.append(label)
        family_values.append(family)
        order_values.append(order)
        x_values[index] = unique_invisible_label(label, index)

    log_values = np.log10(np.nan_to_num(tissue_values, nan=0.0) + 1.0)
    vmax = max(1.0, float(np.nanmax(log_values)))
    absent_value = -max(0.05, vmax * 0.015)
    display_values = log_values.copy()
    display_values[~tissue_present] = absent_value
    gap_mask = np.array([not name for name in expanded_species])
    display_values[:, gap_mask] = np.nan
    colorscale, _ = heatmap_colorscale(vmax, absent_value)

    hover_rows: list[str] = []
    query_tick_text: list[str] = []
    function_rows: list[str] = []
    orthogroup_rows: list[str] = []
    for record in records:
        query = str(record["query"])
        hover_rows.append(
            f"<b>Search ID:</b> {html.escape(query)}"
        )
        compact_query = query if len(query) <= 24 else query[:21] + "..."
        query_tick_text.append(f"<b>{html.escape(compact_query)}</b>")
        function_rows.append(html.escape(str(record["description"])))
        orthogroup_rows.append(html.escape(str(record["orthogroup"])))

    # Keep hover fields in the requested reading order.  Plotly customdata is
    # cell-shaped, so broadcast row and column metadata to every heatmap cell.
    function_values = np.broadcast_to(
        np.asarray(function_rows, dtype=object)[:, None], tissue_values.shape
    )
    orthogroup_values = np.broadcast_to(
        np.asarray(orthogroup_rows, dtype=object)[:, None], tissue_values.shape
    )
    family_matrix = np.broadcast_to(
        np.asarray(family_values, dtype=object)[None, :], tissue_values.shape
    )
    order_matrix = np.broadcast_to(
        np.asarray(order_values, dtype=object)[None, :], tissue_values.shape
    )

    plant_tick_values: list[str] = []
    plant_tick_text: list[str] = []
    for plant_index, species_name in enumerate(ordered_species):
        positions = [index for index, value in enumerate(expanded_species)
                     if value == species_name]
        center_index = positions[(len(positions) - 1) // 2]
        plant_tick_values.append(x_values[center_index])
        plant_name = html.escape(str(taxonomy.loc[species_name, "Name"]))
        # Two horizontal rows substantially reduce collisions between the 105
        # plant names while preserving their full labels.
        plant_tick_text.append(("<br>" if plant_index % 2 else "") + plant_name)
    def chunk_output_path(start: int, stop: int) -> Path:
        """Append one-based inclusive row bounds to the short output stem."""
        return args.output.with_name(
            f"{args.output.stem}_{start}-{stop}{args.output.suffix}"
        )

    def old_chunk_paths() -> list[Path]:
        """Find only chunk files belonging to this exact interactive output."""
        pattern = re.compile(
            rf"^{re.escape(args.output.stem)}_\d+-\d+{re.escape(args.output.suffix)}$"
        )
        return [path for path in args.output.parent.iterdir()
                if path.is_file() and pattern.fullmatch(path.name)]

    def write_chunk(row_start: int, row_stop: int, output: Path) -> None:
        """Write one zero-based half-open row slice as a self-contained HTML."""
        row_slice = slice(row_start, row_stop)
        chunk_hover_rows = hover_rows[row_slice]
        chunk_tick_text = query_tick_text[row_slice]
        figure = go.Figure(data=go.Heatmap(
            z=display_values[row_slice, :],
            x=x_values,
            y=chunk_hover_rows,
            customdata=np.stack([
                intensity_values[row_slice, :],
                uniprot_entry_values[row_slice, :],
                uniprot_protein_values[row_slice, :],
                uniprot_gene_values[row_slice, :],
                function_values[row_slice, :],
                orthogroup_values[row_slice, :],
                family_matrix[row_slice, :],
                order_matrix[row_slice, :],
            ], axis=-1),
            text=source_values[row_slice, :],
            colorscale=colorscale,
            zmin=absent_value,
            zmax=vmax,
            showscale=False,
            xgap=0,
            ygap=0,
            hovertemplate=(
                "%{x}"
                "<br>%{y}"
                "<br><b>Plant protein ID:</b> %{text}"
                "<br><b>NCBI Gene name:</b> %{customdata[4]}"
                "<br><b>Mapped UniProt ID:</b> %{customdata[1]}"
                "<br><b>UniProt protein name:</b> %{customdata[2]}"
                "<br><b>UniProt gene name:</b> %{customdata[3]}"
                "<br><b>Orthogroup:</b> %{customdata[5]}"
                "<br><b>Family:</b> %{customdata[6]}"
                "<br><b>Order:</b> %{customdata[7]}"
                "<br><b>Proteomics mcIBAQ:</b> %{customdata[0]}"
                "<extra></extra>"
            ),
            hoverongaps=False,
        ))
        figure.update_xaxes(
            categoryorder="array", categoryarray=x_values,
            tickmode="array", tickvals=plant_tick_values, ticktext=plant_tick_text,
            side="top", tickangle=0, tickfont=dict(size=9, color="#263238"),
            automargin=True, showline=True, linewidth=1, linecolor="#455A64",
            ticks="outside", ticklen=5, tickcolor="#455A64", showgrid=False,
            showspikes=True, spikecolor="rgba(38,50,56,0.45)",
            spikethickness=1, spikedash="dot", spikemode="across",
            rangeslider=dict(visible=False),
        )
        figure.update_yaxes(
            tickmode="array", tickvals=chunk_hover_rows, ticktext=chunk_tick_text,
            showticklabels=True, tickfont=dict(size=10, color="#263238"),
            ticks="outside", ticklen=4, tickcolor="#455A64",
            showline=True, linewidth=1, linecolor="#455A64",
            showgrid=False, autorange="reversed", automargin=True,
            showspikes=True, spikecolor="rgba(38,50,56,0.45)",
            spikethickness=1, spikedash="dot", spikemode="across",
            title_text=None,
        )
        figure.update_layout(
            template="plotly_white", autosize=False,
            width=max(1800, len(x_values) * 4 + 320),
            height=max(1000, (row_stop - row_start) * 18 + 220),
            margin=dict(l=240, r=45, t=150, b=55),
            paper_bgcolor="white", plot_bgcolor="#B7BCC1",
            showlegend=False, dragmode="zoom", hovermode="closest",
            hoverdistance=20,
            hoverlabel=dict(
                bgcolor="rgba(255,255,255,0.98)", bordercolor="#455A64",
                font_size=13, font_family="Arial", font_color="#17252A",
                align="left", namelength=-1,
            ),
            modebar=dict(bgcolor="rgba(255,255,255,0.85)", color="#455A64",
                         activecolor="#1565C0", orientation="h"),
        )
        figure.write_html(output, include_plotlyjs=True, full_html=True)
        print(f"Wrote {output} (rows {row_start + 1}-{row_stop})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for stale_path in old_chunk_paths():
        stale_path.unlink()
    if args.chunk_size and len(records) > args.chunk_size:
        if args.output.exists():
            args.output.unlink()
        # Balance the rows instead of leaving a tiny final file.  For example,
        # 520 rows become 260+260 and 1,200 rows become 400+400+400.  The chunk
        # count is the smallest one that keeps every file at or below the
        # requested limit.
        chunk_count = (len(records) + args.chunk_size - 1) // args.chunk_size
        base_size, extra_rows = divmod(len(records), chunk_count)
        row_start = 0
        for chunk_index in range(chunk_count):
            chunk_rows = base_size + (1 if chunk_index < extra_rows else 0)
            row_stop = row_start + chunk_rows
            write_chunk(
                row_start, row_stop,
                chunk_output_path(row_start + 1, row_stop),
            )
            row_start = row_stop
    else:
        write_chunk(0, len(records), args.output)
    print(f"Rows: {len(records)}; tissue columns: {len(tissue_columns)}; "
          f"display columns including spacers: {len(x_values)}")


if __name__ == "__main__":
    main()
