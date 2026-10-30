#!/usr/bin/env python3
"""Resolve matrix tissue columns marked as outliers in the QC report."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _norm_text(value: object) -> str:
    return " ".join(str(value).replace("_", " ").strip().casefold().split())


def _species_key(value: object) -> str:
    text = str(value).strip()
    return re.sub(r"(?:\.helixer)?\.(?:faa|fasta|fa)$", "", text)


def _matrix_key_from_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9:]+", "_", str(value).strip())
    return re.sub(r"_+", "_", text).strip("_")


def matrix_plant_names(metadata_path: str | Path) -> dict[str, str]:
    metadata = pd.read_csv(metadata_path, sep="\t", dtype=str).fillna("")
    if not {"FASTA", "Name"}.issubset(metadata.columns):
        raise ValueError("Metadata must contain FASTA and Name columns")
    metadata["reference_key"] = metadata["FASTA"].map(_species_key)
    shared = metadata["reference_key"].duplicated(keep=False)
    metadata["matrix_key"] = metadata["reference_key"]
    metadata.loc[shared, "matrix_key"] = metadata.loc[shared, "Name"].map(
        _matrix_key_from_name
    )
    return {
        str(row.matrix_key): str(row.Name).strip()
        for row in metadata[["matrix_key", "Name"]].itertuples(index=False)
    }


def outlier_tissue_keys(
    outliers_path: str | Path,
    tissue_ontology_path: str | Path,
) -> set[str]:
    report = pd.read_csv(outliers_path, sep="\t", dtype=str).fillna("")
    columns = {column.strip().casefold(): column for column in report.columns}
    outlier_col = columns.get("outlier") or columns.get("oulier")
    tissue_col = columns.get("tissue")
    plant_col = columns.get("plant")
    if not outlier_col or not tissue_col:
        raise ValueError("Outlier table must contain Outlier and Tissue columns")
    report = report[report[outlier_col].str.strip().eq("1")]

    plants_by_pnumber: dict[str, set[str]] = {}
    for row in report.itertuples(index=False):
        values = row._asdict()
        pnumber = str(values[tissue_col]).strip()
        plant = _norm_text(values[plant_col]) if plant_col else ""
        if pnumber:
            plants_by_pnumber.setdefault(pnumber, set()).add(plant)

    # p-number is the sample identity.  Do not route outlier matching through
    # PO/tissue annotations: those can be missing, duplicated, or incorrect.
    keys: set[str] = set()
    for pnumber in plants_by_pnumber:
        match = re.match(r"^(P\d+)", pnumber, re.IGNORECASE)
        if not match:
            continue
        keys.add(match.group(1).upper())
    return keys


def outlier_column_indices(
    matrix_columns: list[str],
    outliers_path: str | Path | None,
    tissue_ontology_path: str | Path | None,
    metadata_path: str | Path | None,
) -> set[int]:
    if not outliers_path or not tissue_ontology_path or not metadata_path:
        return set()
    keys = outlier_tissue_keys(outliers_path, tissue_ontology_path)
    indices: set[int] = set()
    for index, column in enumerate(matrix_columns):
        match = re.fullmatch(r"(.+)_(P\d+)", str(column).strip(), re.IGNORECASE)
        if not match:
            continue
        plant_key, pnumber = match.group(1), match.group(2).upper()
        if pnumber in keys:
            indices.add(index)
    return indices
