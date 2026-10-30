#!/usr/bin/env python3
"""Build category heatmaps from direct family-mapped orthogroup unions."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from generate_tree_ordered_presence_heatmap import normalize_key, split_matrix_column


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "Results/Hormones"
DEFAULT_MAPPINGS = Path("per_family_best_hits.tsv")
DEFAULT_MATRIX = Path("protein_orthogroup_pnumber_matrix_mcIBAQ_intensities.tsv")
DEFAULT_OGS = Path("Orthogroups.tsv")


def fasta_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return ids


def read_query_family_ogs(mapping_path: Path, query_order: list[str]):
    table = pd.read_csv(mapping_path, sep="\t", dtype=str).fillna("")
    table = table.loc[table["status"].eq("assigned") & table["orthogroup"].ne("")]
    duplicate = table.duplicated(["query", "family"], keep=False)
    if duplicate.any():
        raise ValueError(
            "More than one assigned OG for a query/family pair:\n"
            + table.loc[duplicate, ["query", "family", "orthogroup"]].to_string(index=False)
        )
    by_query: dict[str, OrderedDict[str, str]] = {}
    for row in table[["query", "family", "orthogroup"]].itertuples(index=False):
        by_query.setdefault(str(row.query), OrderedDict())[str(row.family)] = str(row.orthogroup)
    # Queries without any assigned family hit remain explicit all-zero/all-absent
    # rows; silently dropping them would change the Hormones query set.
    return {query: by_query.get(query, OrderedDict()) for query in query_order}


def family_maps(metadata_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    metadata = pd.read_csv(metadata_path, sep="\t", dtype=str).fillna("")
    metadata["reference_key"] = metadata["FASTA"].map(normalize_key)
    shared = metadata["reference_key"].duplicated(keep=False)
    metadata["matrix_key"] = metadata["reference_key"]
    metadata.loc[shared, "matrix_key"] = metadata.loc[shared, "Name"].map(normalize_key)
    matrix_family = dict(zip(metadata["matrix_key"], metadata["Family"]))
    reference_family: dict[str, str] = {}
    for reference, group in metadata.groupby("reference_key", sort=False):
        families = [value for value in group["Family"].unique() if value]
        if len(families) != 1:
            raise ValueError(f"Reference {reference} maps to families: {families}")
        reference_family[str(reference)] = families[0]
    return matrix_family, reference_family


def selected_matrix_rows(path: Path, wanted: set[str]):
    values: dict[str, np.ndarray] = {}
    descriptions: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row:
                continue
            identifier = row[0]
            og, _, description = identifier.partition("|")
            if og not in wanted:
                continue
            numeric = np.zeros(len(header) - 1, dtype=float)
            for index, value in enumerate(row[1:]):
                try:
                    number = float(value)
                    numeric[index] = number if np.isfinite(number) else 0.0
                except (TypeError, ValueError):
                    pass
            values[og] = numeric
            descriptions[og] = description
            if len(values) == len(wanted):
                break
    # An OG absent from the proteomics matrix is valid and contributes zeros.
    for og in wanted - values.keys():
        values[og] = np.zeros(len(header) - 1, dtype=float)
        descriptions[og] = ""
    return header, values, descriptions


def selected_memberships(path: Path, wanted: set[str]):
    memberships: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        species_columns = [name for name in (reader.fieldnames or []) if name != "Orthogroup"]
        for row in reader:
            og = str(row.get("Orthogroup", "")).strip()
            if og not in wanted:
                continue
            memberships[og] = {
                species: str(row.get(species, "")).strip()
                for species in species_columns
            }
            if len(memberships) == len(wanted):
                break
    for og in wanted - memberships.keys():
        memberships[og] = {species: "" for species in species_columns}
    return species_columns, memberships


def write_inputs(results: Path, query_family_ogs: dict[str, OrderedDict[str, str]],
                 matrix_path: Path, orthogroups_path: Path, metadata_path: Path):
    support = results / "other_files"
    support.mkdir(parents=True, exist_ok=True)
    work = results / ".family_merge_work"
    work.mkdir(parents=True, exist_ok=True)
    matrix_output = work / "family_OGs_merged_tissue_matrix.tsv"
    assigned_output = work / "family_OGs_merged_Assigned_Orthogroups.tsv"
    audit_output = support / "family_OGs_merged_query_to_orthogroups.tsv"
    public_matrix_output = results / "query_protein_tissue_mcIBAQ_matrix.tsv"

    wanted = {og for family_ogs in query_family_ogs.values()
              for og in family_ogs.values()}
    matrix_header, values_by_og, _descriptions = selected_matrix_rows(matrix_path, wanted)
    species_columns, membership_by_og = selected_memberships(orthogroups_path, wanted)
    matrix_family, reference_family = family_maps(metadata_path)
    column_families = []
    for column in matrix_header[1:]:
        matrix_species, _ = split_matrix_column(column)
        family = matrix_family.get(matrix_species, "")
        if not family:
            raise ValueError(f"No family metadata for matrix column: {column}")
        column_families.append(family)

    def family_specific_values(family_ogs: OrderedDict[str, str]) -> np.ndarray:
        merged = np.zeros(len(matrix_header) - 1, dtype=float)
        for column_index, family in enumerate(column_families):
            og = family_ogs.get(family)
            if og:
                merged[column_index] = values_by_og[og][column_index]
        return merged

    with matrix_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(matrix_header)
        for index, (query, family_ogs) in enumerate(query_family_ogs.items(), start=1):
            synthetic = f"FAMMERGE{index:04d}"
            unique_ogs = list(OrderedDict.fromkeys(family_ogs.values()))
            merged = family_specific_values(family_ogs)
            writer.writerow([
                f"{synthetic}|family-specific OG concatenation ({len(unique_ogs)} unique OGs)",
                *[f"{value:.10g}" for value in merged],
            ])

    with public_matrix_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query_protein_id", *matrix_header[1:]])
        for query, family_ogs in query_family_ogs.items():
            merged = family_specific_values(family_ogs)
            writer.writerow([query, *[f"{value:.10g}" for value in merged]])

    with assigned_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["Orthogroup", "search_protein_id",
                         "family_mapped_orthogroups", *species_columns])
        for index, (query, family_ogs) in enumerate(query_family_ogs.items(), start=1):
            synthetic = f"FAMMERGE{index:04d}"
            selected_members = []
            for species in species_columns:
                family = reference_family.get(normalize_key(species), "")
                og = family_ogs.get(family)
                selected_members.append(membership_by_og[og][species] if og else "")
            unique_ogs = list(OrderedDict.fromkeys(family_ogs.values()))
            writer.writerow([synthetic, query, ";".join(unique_ogs), *selected_members])

    with audit_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["synthetic_orthogroup", "query", "families_with_assigned_OG",
                         "number_of_unique_family_OGs", "family_to_orthogroup",
                         "family_mapped_orthogroups"])
        for index, (query, family_ogs) in enumerate(query_family_ogs.items(), start=1):
            unique_ogs = list(OrderedDict.fromkeys(family_ogs.values()))
            family_pairs = ";".join(
                f"{family}={og}" for family, og in family_ogs.items()
            )
            writer.writerow([
                f"FAMMERGE{index:04d}", query, len(family_ogs), len(unique_ogs),
                family_pairs, ";".join(unique_ogs),
            ])
    return matrix_output, assigned_output, audit_output, public_matrix_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--mappings", type=Path)
    parser.add_argument("--full-matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--orthogroups", type=Path, default=DEFAULT_OGS)
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--protein-descriptions", required=True, type=Path)
    parser.add_argument("--intensities", required=True, type=Path)
    parser.add_argument("--outliers", required=True, type=Path)
    parser.add_argument("--tissue-ontology", required=True, type=Path)
    parser.add_argument("--pnumber-tissues", required=True, type=Path)
    parser.add_argument("--uniprot-mapping", required=True, type=Path)
    parser.add_argument("--species-zscore-only", action="store_true",
                        help="Backfill only the global positive-mcIBAQ species z-score heatmap")
    parser.add_argument("--tissue-minmax-only", action="store_true",
                        help="Backfill only the global positive-mcIBAQ all-tissues min-max heatmap")
    parser.add_argument("--raw-tissue-only", action="store_true",
                        help="Backfill only the raw all-tissues heatmap and legend")
    parser.add_argument("--per-plant-only", action="store_true",
                        help="Backfill only the individual plant heatmaps")
    parser.add_argument("--interactive-only", action="store_true",
                        help="Backfill only the interactive heatmap HTML file(s)")
    args = parser.parse_args()

    category = args.results.name
    query_fasta = args.results / f"{category}_fasta.faa"
    if not query_fasta.exists():
        query_fasta = ROOT / "ENB_Fastas" / f"{category}.faa"
    mapping_path = args.mappings
    if mapping_path is None:
        mapping_path = (DEFAULT_MAPPINGS if category == "Hormones" else
                        args.results / "other_files/per_family_best_hits.tsv")
    queries = fasta_ids(query_fasta)
    query_family_ogs = read_query_family_ogs(mapping_path, queries)
    matrix, assigned, audit, public_matrix = write_inputs(
        args.results, query_family_ogs, args.full_matrix, args.orthogroups,
        args.metadata,
    )

    other = args.results / "other_files"
    other_heatmaps = args.results / "other_heatmaps"
    per_plant_heatmaps = other_heatmaps / "per_plant"
    hits_path = mapping_path
    tissue_output = args.results / f"{category}_heatmap_tree_all_tissues_family_OGs_merged.jpg"
    species_output = args.results / f"{category}_heatmap_tree_species.jpg"
    interactive_output = (
        args.results / f"interactive_heatmap_{category.lower()}.html"
    )
    command = [
        sys.executable, str(ROOT / "scripts/generate_tree_ordered_presence_heatmap.py"),
        "--matrix", str(matrix),
        "--tree", str(args.tree),
        "--orthogroups", str(assigned),
        "--hits", str(hits_path),
        "--protein-descriptions", str(args.protein_descriptions),
        "--metadata", str(args.metadata),
        "--outliers", str(args.outliers),
        "--tissue-ontology", str(args.tissue_ontology),
        "--pnumber-tissues", str(args.pnumber_tissues),
        "--tissue-dpi", "250",
        "--no-minmax",
        "--other-heatmaps-dir", str(other_heatmaps),
        "--per-plant-dir", str(per_plant_heatmaps),
        "--per-plant-dpi", "180",
        "--output", str(tissue_output),
        "--species-output", str(species_output),
    ]
    if args.species_zscore_only:
        command.append("--species-zscore-only")
    if args.tissue_minmax_only:
        command.append("--tissue-minmax-only")
    if args.raw_tissue_only:
        command.append("--raw-tissue-only")
    if args.per_plant_only:
        command.append("--per-plant-only")
    if not args.interactive_only:
        subprocess.run(command, check=True)
    if (args.species_zscore_only or args.tissue_minmax_only or
            args.raw_tissue_only or args.per_plant_only):
        shutil.rmtree(args.results / ".family_merge_work")
        if args.species_zscore_only:
            print("Wrote " + str(
                other_heatmaps / f"{category}_all_plants_tissues_z_score.png"
            ))
        if args.tissue_minmax_only:
            print("Wrote " + str(
                other_heatmaps / f"{category}_all_plants_tissues_min_max.jpg"
            ))
        if args.raw_tissue_only:
            print(f"Wrote {tissue_output}")
        if args.per_plant_only:
            print(f"Wrote {per_plant_heatmaps}")
        return
    interactive_command = [
        sys.executable, str(ROOT / "scripts/generate_interactive_tree_tissue_heatmap.py"),
        "--matrix", str(matrix),
        "--tree", str(args.tree),
        "--orthogroups", str(assigned),
        "--hits", str(hits_path),
        "--query-fasta", str(query_fasta),
        "--protein-descriptions", str(args.protein_descriptions),
        "--intensities", str(args.intensities),
        "--metadata", str(args.metadata),
        "--outliers", str(args.outliers),
        "--tissue-ontology", str(args.tissue_ontology),
        "--pnumber-tissues", str(args.pnumber_tissues),
        "--uniprot-mapping", str(args.uniprot_mapping),
        "--merge-audit", str(audit),
        "--aggregation", "max",
        "--chunk-size", "500",
        "--output", str(interactive_output),
    ]
    subprocess.run(interactive_command, check=True)
    shutil.rmtree(args.results / ".family_merge_work")
    if not args.interactive_only:
        print(f"Wrote {tissue_output}")
        print(f"Wrote {species_output}")
    interactive_pattern = f"interactive_heatmap_{category.lower()}*.html"
    for path in sorted(args.results.glob(interactive_pattern)):
        print(f"Wrote {path}")
    print(f"Wrote {public_matrix}")
    print(f"Audit: {audit}")


if __name__ == "__main__":
    main()
