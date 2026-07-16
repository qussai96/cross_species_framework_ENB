#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


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


def normalize_pnumber(value: object) -> str | None:
    if pd.isna(value) or value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"_[0-9]+$", "", text)
    text = re.sub(r"\.v\d+$", "", text)
    return text or None


def split_multi_ids(cell: object) -> List[str]:
    if pd.isna(cell):
        return []
    return [x.strip() for x in str(cell).split(",") if x.strip()]


def parse_fasta_ids_in_order(input_fasta: str | Path | None) -> List[str]:
    if not input_fasta:
        return []

    proteins: List[str] = []
    with open(input_fasta, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                proteins.append(line[1:].strip().split()[0])
    return proteins


def _normalize_id_token(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    return re.sub(r"\s+", "_", text)


def parse_search_ids_from_tsv_in_order(search_order_tsv: str | Path | None) -> List[str]:
    if not search_order_tsv:
        return []

    df = pd.read_csv(search_order_tsv, sep="\t", dtype=str)
    if df.empty:
        return []

    col_by_norm = {_norm_text(col): col for col in df.columns}
    required_norm_cols = ["hormone", "name", "locus", "function"]
    if any(col not in col_by_norm for col in required_norm_cols):
        raise ValueError(
            "TSV ordering file must contain columns: Hormone, Name, Locus, Function."
        )

    ordered_ids: List[str] = []
    seen: Set[str] = set()
    for _, row in df.iterrows():
        hormone = _normalize_id_token(row[col_by_norm["hormone"]])
        name = _normalize_id_token(row[col_by_norm["name"]])
        locus = _normalize_id_token(row[col_by_norm["locus"]])
        function = _normalize_id_token(row[col_by_norm["function"]])
        if not (hormone and name and locus and function):
            continue

        search_id = f"{hormone}_{name}_{locus}_{function}"
        if search_id not in seen:
            seen.add(search_id)
            ordered_ids.append(search_id)

    return ordered_ids


def compact_label(text: str, max_len: int = 95) -> str:
    s = str(text)
    if len(s) <= max_len:
        return s
    keep = max_len - 3
    left = keep // 2
    right = keep - left
    return s[:left] + "..." + s[-right:]


def split_y_label_parts(label: str) -> tuple[str, str, str]:
    search_id_text, plant_protein_id, orthogroup = str(label).rsplit(" / ", 2)
    return search_id_text, plant_protein_id, orthogroup


def build_fasta_order_rank(input_fasta: str | Path | None) -> Dict[str, int]:
    return {protein_id: idx for idx, protein_id in enumerate(parse_fasta_ids_in_order(input_fasta))}


def build_tsv_order_rank(search_order_tsv: str | Path | None) -> Dict[str, int]:
    return {
        protein_id: idx for idx, protein_id in enumerate(parse_search_ids_from_tsv_in_order(search_order_tsv))
    }


def order_table_by_fasta_hits(table: pd.DataFrame, fasta_order_rank: Dict[str, int]) -> pd.DataFrame:
    if table.empty or not fasta_order_rank:
        return table

    ordered = table.copy()
    ordered["_search_rank"] = [
        min((fasta_order_rank.get(search_id, len(fasta_order_rank)) for search_id in split_multi_ids(split_y_label_parts(label)[0])), default=len(fasta_order_rank))
        for label in ordered.index
    ]
    ordered["_original_order"] = np.arange(len(ordered))
    ordered = ordered.sort_values(["_search_rank", "_original_order"], kind="stable")
    return ordered.drop(columns=["_search_rank", "_original_order"])


def build_aligned_all_hits_table(
    full_original_table: pd.DataFrame,
    orth_df: pd.DataFrame,
    orth_og_col: str,
    plant_protein_col: str,
    input_fasta: str | Path | None,
) -> pd.DataFrame:
    if full_original_table.empty:
        return full_original_table

    search_col = "search_protein_id" if "search_protein_id" in orth_df.columns else None
    if search_col is None:
        raise ValueError("Orthogroups table is missing 'search_protein_id' column.")

    fasta_order_rank = build_fasta_order_rank(input_fasta)
    fallback_rank = len(fasta_order_rank)
    species_cols = [c for c in orth_df.columns if c not in {orth_og_col, search_col}]
    zero_row = pd.Series(0.0, index=full_original_table.columns, dtype=float)

    template_cols = [orth_og_col, search_col] + species_cols
    template_rows = orth_df[template_cols].copy()
    template_rows["_row_order"] = np.arange(len(template_rows))
    template_rows["_search_rank"] = [
        min((fasta_order_rank.get(search_id, fallback_rank) for search_id in split_multi_ids(cell)), default=fallback_rank)
        for cell in template_rows[search_col]
    ]
    template_rows = template_rows.sort_values(["_search_rank", "_row_order"], kind="stable")

    aligned_rows = []
    aligned_index = []
    plant_protein_ids = []

    for _, row in template_rows.iterrows():
        search_ids = split_multi_ids(row[search_col])
        if not search_ids:
            continue

        search_ids_text = ",".join(search_ids)
        orthogroup = str(row[orth_og_col])
        plant_proteins = split_multi_ids(row[plant_protein_col])
        max_hits = max((len(split_multi_ids(row[col])) for col in species_cols), default=0)
        max_hits = max(max_hits, 1)

        for slot_idx in range(max_hits):
            plant_protein_id = plant_proteins[slot_idx] if slot_idx < len(plant_proteins) else ""
            actual_label = None
            if plant_protein_id:
                actual_label = f"{search_ids_text} / {plant_protein_id} / {orthogroup}"

            if actual_label and actual_label in full_original_table.index:
                row_values = full_original_table.loc[actual_label]
            else:
                row_values = zero_row.copy()

            aligned_rows.append(row_values.to_numpy(copy=True))
            aligned_index.append(f"{search_ids_text} / hit_{slot_idx + 1} / {orthogroup}")
            plant_protein_ids.append(plant_protein_id)

    aligned = pd.DataFrame(aligned_rows, index=aligned_index, columns=full_original_table.columns)
    aligned.insert(0, "plant_protein_id", plant_protein_ids)
    aligned.index.name = "aligned_hit_id"
    return aligned


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


def load_plant_aliases(alias_file: str | Path | None = None) -> Dict[str, Set[str]]:
    if alias_file is None:
        alias_file = Path(__file__).resolve().parents[1] / "plant_name_aliases.tsv"

    path = Path(alias_file)
    if not path.exists():
        return {}

    aliases: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"name", "alias"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Alias file must contain columns {sorted(required)}: {path}")

        for row in reader:
            name_norm = _norm_text(row.get("name", ""))
            alias_norm = _norm_text(row.get("alias", ""))
            if not name_norm or not alias_norm:
                continue
            aliases.setdefault(name_norm, set()).add(alias_norm)
            aliases.setdefault(alias_norm, set()).add(name_norm)

    return aliases


def _plant_name_candidates(plant_display_name: str | None, plant_key: str) -> Set[str]:
    candidates: Set[str] = set()

    display_norm = _norm_text(plant_display_name or "")
    key_norm = _norm_text(plant_key)
    if display_norm:
        candidates.add(display_norm)
    if key_norm:
        candidates.add(key_norm)

    # Example: GCF_022201045.2_Citrus_sinensis -> citrus sinensis
    raw_key = str(plant_key)
    if "_" in raw_key:
        tail = raw_key.split("_", 2)[-1]
        tail_norm = _norm_text(tail)
        if tail_norm:
            candidates.add(tail_norm)

    # If the key starts with an accession, keep only genus/species tail as candidate.
    key_parts = key_norm.split()
    if len(key_parts) >= 3 and key_parts[0] in {"gca", "gcf"}:
        candidates.add(" ".join(key_parts[2:]))

    return {c for c in candidates if c}


def _filter_ontology_for_plant(
    ont: pd.DataFrame,
    plant_display_name: str | None,
    plant_key: str,
    plant_aliases: Dict[str, Set[str]] | None = None,
) -> pd.DataFrame:
    if "PlantName" not in ont.columns:
        return ont

    candidates = _plant_name_candidates(plant_display_name, plant_key)
    if plant_aliases:
        expanded = set(candidates)
        for c in list(candidates):
            expanded.update(plant_aliases.get(c, set()))
        candidates = expanded

    if not candidates:
        return ont

    exact_hits = ont[ont["_plant_norm"].isin(candidates)]
    if not exact_hits.empty:
        return exact_hits

    # Fuzzy fallback: handles alias/plural differences like
    # Orange <-> Sweet orange and Papayas <-> Papaya.
    fuzzy_hits = ont[
        ont["_plant_norm"].map(
            lambda pn: any(
                (len(c) >= 4) and (c in pn or pn in c)
                for c in candidates
            )
        )
    ]
    if not fuzzy_hits.empty:
        return fuzzy_hits

    return ont


FUNCTION_CATEGORY_ORDER: List[str] = [
    "Co-receptor",
    "Co-receptor/regulator",
    "Co-receptor/repressor",
    "Function",
    "Receptor",
    "Receptor regulator",
    "responsive gene",
    "Responsive gene",
    "Signaling",
    "Synthesis",
    "Transcription factor",
    "Transcription Factor",
    "Transcripton Factor",
]

FUNCTION_CATEGORY_RANK: Dict[str, int] = {
    category: idx for idx, category in enumerate(FUNCTION_CATEGORY_ORDER)
}

# Normalize common variants and typos to one sortable/display form.
FUNCTION_CATEGORY_CANONICAL: Dict[str, str] = {
    "co receptor": "Co-receptor",
    "co receptor regulator": "Co-receptor/regulator",
    "co receptor repressor": "Co-receptor/repressor",
    "function": "Function",
    "receptor": "Receptor",
    "receptor regulator": "Receptor regulator",
    "responsive gene": "Responsive gene",
    "signaling": "Signaling",
    "synthesis": "Synthesis",
    "transcription factor": "Transcription factor",
    "transcripton factor": "Transcription factor",
    "transcription facor": "Transcription factor",
}

FUNCTION_CATEGORY_COLORS: Dict[str, str] = {
    "Co-receptor": "#17becf",
    "Co-receptor/regulator": "#1f77b4",
    "Co-receptor/repressor": "#00a6a6",
    "Function": "#7f7f7f",
    "Receptor": "#d62728",
    "Receptor regulator": "#ff7f0e",
    "responsive gene": "#2ca02c",
    "Responsive gene": "#2ca02c",
    "Signaling": "#9467bd",
    "Synthesis": "#8c564b",
    "Transcription factor": "#e377c2",
    "Transcription Factor": "#e377c2",
    "Transcripton Factor": "#e377c2",
    "Unknown": "#4d4d4d",
}


def _normalize_function_phrase(text: str) -> str:
    normalized = _norm_text(text).replace("-", " ").replace("/", " ")
    return " ".join(normalized.split())


def canonical_function_label(text: str) -> str:
    key = _normalize_function_phrase(text)
    return FUNCTION_CATEGORY_CANONICAL.get(key, text)


def function_category_from_search_id(search_id: str) -> str:
    normalized_id = _normalize_function_phrase(search_id)

    # Match longest category suffix first (e.g. receptor regulator before receptor).
    candidate_suffixes = sorted(FUNCTION_CATEGORY_CANONICAL.keys(), key=len, reverse=True)
    for suffix in candidate_suffixes:
        if normalized_id.endswith(suffix):
            return FUNCTION_CATEGORY_CANONICAL[suffix]

    return "Unknown"


def function_category_from_row_label(label: str) -> str:
    text = str(label)
    search_id_text = text
    try:
        search_id_text, _, _ = split_y_label_parts(text)
    except ValueError:
        if " / " in text:
            search_id_text = text.split(" / ", 1)[0]
        elif "/" in text:
            search_id_text = text.split("/", 1)[0]

    categories = [
        function_category_from_search_id(search_id)
        for search_id in split_multi_ids(search_id_text)
    ]
    categories = [c for c in categories if c != "Unknown"]
    if not categories:
        return "Unknown"

    categories = [canonical_function_label(c) for c in categories]
    return min(categories, key=lambda c: FUNCTION_CATEGORY_RANK.get(c, len(FUNCTION_CATEGORY_ORDER)))


def sort_table_by_function_category(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table

    ordered = table.copy()
    ordered["_function_category"] = [
        function_category_from_row_label(label) for label in ordered.index
    ]
    ordered["_function_rank"] = [
        FUNCTION_CATEGORY_RANK.get(c, len(FUNCTION_CATEGORY_ORDER))
        for c in ordered["_function_category"]
    ]
    ordered["_original_order"] = np.arange(len(ordered))
    ordered = ordered.sort_values(["_function_rank", "_original_order"], kind="stable")
    return ordered.drop(columns=["_function_category", "_function_rank", "_original_order"])


def sort_table_by_search_order_rank(table: pd.DataFrame, search_order_rank: Dict[str, int]) -> pd.DataFrame:
    if table.empty or not search_order_rank:
        return table

    fallback_rank = len(search_order_rank)
    ordered = table.copy()
    ordered["_search_rank"] = [
        min(
            (search_order_rank.get(search_id, fallback_rank) for search_id in split_multi_ids(split_y_label_parts(label)[0])),
            default=fallback_rank,
        )
        for label in ordered.index
    ]
    ordered["_original_order"] = np.arange(len(ordered))
    ordered = ordered.sort_values(["_search_rank", "_original_order"], kind="stable")
    return ordered.drop(columns=["_search_rank", "_original_order"])


def is_transcription_factor_label(text: str) -> bool:
    t = _norm_text(text)
    keys = [
        "transcription factor",
        "transcription facor",
    ]
    return any(k in t for k in keys)


def is_receptor_label(text: str) -> bool:
    t = _norm_text(text)
    keys = [
        "receptor",
    ]
    return any(k in t for k in keys)


def normalize_tissue_label_for_matrix(value: object) -> str:
    if pd.isna(value) or value is None:
        return "unknown_tissue"
    text = str(value).strip()
    if not text:
        return "unknown_tissue"
    text = re.sub(r"[^A-Za-z0-9:]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown_tissue"


def _pnumber_numeric_key(pnum: str) -> int:
    m = re.search(r"(\d+)", str(pnum))
    if not m:
        return 10**18
    return int(m.group(1))


def sort_plant_cols_by_pnumber(
    plant_cols: List[str],
    tissue_ontology_path: str | None,
    plant_display_name: str | None,
    plant_key: str,
    plant_aliases: Dict[str, Set[str]] | None = None,
) -> List[str]:
    if not tissue_ontology_path:
        return plant_cols

    ont = pd.read_csv(tissue_ontology_path, sep="\t", dtype=str)
    required = {"pNumber", "PO_1", "TissueName"}
    if ont.empty or not required.issubset(ont.columns):
        return plant_cols

    ont = ont.copy()
    ont["_plant_norm"] = ont.get("PlantName", "").map(_norm_text)

    filtered = _filter_ontology_for_plant(ont, plant_display_name, plant_key, plant_aliases)

    key_to_pnum: Dict[Tuple[str, str], str] = {}
    for _, row in filtered.iterrows():
        po = str(row.get("PO_1", "")).strip()
        tissue_norm = normalize_tissue_label_for_matrix(row.get("TissueName"))
        pnum = str(row.get("pNumber", "")).strip()
        if not po or not tissue_norm or not pnum:
            continue
        key = (po, tissue_norm)
        existing = key_to_pnum.get(key)
        if existing is None or _pnumber_numeric_key(pnum) < _pnumber_numeric_key(existing):
            key_to_pnum[key] = pnum

    annotated = []
    for idx, col in enumerate(plant_cols):
        _, po_id, tissue_name = split_matrix_column(col)
        pnum = None
        if po_id and tissue_name:
            pnum = key_to_pnum.get((po_id, tissue_name))
        annotated.append((col, pnum, idx))

    annotated.sort(
        key=lambda x: (
            0 if x[1] else 1,
            _pnumber_numeric_key(x[1]) if x[1] else 10**18,
            str(x[1]) if x[1] else "",
            x[2],
        )
    )
    return [col for col, _, _ in annotated]


def build_po_tissue_to_pnumber_map(
    tissue_ontology_path: str | None,
    plant_display_name: str | None,
    plant_key: str,
    plant_aliases: Dict[str, Set[str]] | None = None,
) -> Dict[Tuple[str, str], str]:
    if not tissue_ontology_path:
        return {}

    ont = pd.read_csv(tissue_ontology_path, sep="\t", dtype=str)
    required = {"pNumber", "PO_1", "TissueName"}
    if ont.empty or not required.issubset(ont.columns):
        return {}

    ont = ont.copy()
    ont["_plant_norm"] = ont.get("PlantName", "").map(_norm_text)

    filtered = _filter_ontology_for_plant(ont, plant_display_name, plant_key, plant_aliases)

    key_to_pnum: Dict[Tuple[str, str], str] = {}
    for _, row in filtered.iterrows():
        po = str(row.get("PO_1", "")).strip()
        tissue_norm = normalize_tissue_label_for_matrix(row.get("TissueName"))
        pnum = str(row.get("pNumber", "")).strip()
        if not po or not tissue_norm or not pnum:
            continue
        key = (po, tissue_norm)
        existing = key_to_pnum.get(key)
        if existing is None or _pnumber_numeric_key(pnum) < _pnumber_numeric_key(existing):
            key_to_pnum[key] = pnum

    return key_to_pnum


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
    include_zero_rows: bool = False,
    sort_by_total_intensity: bool = True,
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

    if not include_zero_rows:
        detected = merged[plant_cols].sum(axis=1) > 0
        merged = merged.loc[detected].copy()

    if merged.empty:
        raise ValueError(
            f"No {'detected ' if not include_zero_rows else ''}proteins for plant '{plant_key}'."
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
    if sort_by_total_intensity:
        out["_total_intensity"] = out[plant_cols].sum(axis=1)
        out = out.sort_values(by="_total_intensity", ascending=False)
        out = out.drop(columns=["_total_intensity"])
    return out


def build_original_maxlfq_table(
    reference_table: pd.DataFrame,
    ordered_plant_cols: List[str],
    intensities_dir: str,
    tissue_ontology_path: str | None,
    plant_display_name: str | None,
    plant_key: str,
    plant_aliases: Dict[str, Set[str]] | None = None,
) -> pd.DataFrame:
    species_candidates = []
    for candidate in [plant_display_name, plant_key]:
        if candidate:
            species_candidates.append(str(candidate).strip())

    intensity_path = None
    for species_name in species_candidates:
        legacy_candidate = Path(intensities_dir) / f"{species_name}.proteins.tsv"
        mcibaq_candidate = Path(intensities_dir) / f"{species_name}_mcIBAQ.intensities.tsv"

        if mcibaq_candidate.exists():
            intensity_path = mcibaq_candidate
            break
        if legacy_candidate.exists():
            intensity_path = legacy_candidate
            break

    if intensity_path is None:
        raise FileNotFoundError(
            f"Could not find protein intensity TSV for plant '{plant_key}' in {intensities_dir}."
        )

    po_tissue_to_pnum = build_po_tissue_to_pnumber_map(
        tissue_ontology_path=tissue_ontology_path,
        plant_display_name=plant_display_name,
        plant_key=plant_key,
        plant_aliases=plant_aliases,
    )

    normalized_pnum_by_matrix_col: Dict[str, str] = {}
    for col in ordered_plant_cols:
        _, po_id, tissue_name = split_matrix_column(col)
        if not po_id or not tissue_name:
            continue
        pnum = po_tissue_to_pnum.get((po_id, tissue_name))
        if not pnum:
            continue
        normalized_pnum = normalize_pnumber(pnum)
        if normalized_pnum:
            normalized_pnum_by_matrix_col[col] = normalized_pnum

    intensity_df = pd.read_csv(intensity_path, sep="\t", low_memory=False)
    source_col_by_normalized_pnum: Dict[str, str] = {}

    maxlfq_cols = [c for c in intensity_df.columns if "MaxLFQ Intensity" in str(c)]
    if maxlfq_cols:
        for col in maxlfq_cols:
            raw_pnum = str(col).replace("MaxLFQ Intensity", "").strip()
            normalized_pnum = normalize_pnumber(raw_pnum)
            if normalized_pnum and normalized_pnum not in source_col_by_normalized_pnum:
                source_col_by_normalized_pnum[normalized_pnum] = col
    else:
        # mcIBAQ tables use p-number-style columns such as P099891_1, P099892_2, ...
        for col in intensity_df.columns:
            c = str(col).strip()
            if c in {"Protein", "Protein ID", "Indistinguishable Proteins"}:
                continue
            normalized_pnum = normalize_pnumber(c)
            if normalized_pnum and normalized_pnum not in source_col_by_normalized_pnum:
                source_col_by_normalized_pnum[normalized_pnum] = col

    resolved_source_cols_by_matrix_col = {
        matrix_col: source_col_by_normalized_pnum[normalized_pnum]
        for matrix_col, normalized_pnum in normalized_pnum_by_matrix_col.items()
        if normalized_pnum in source_col_by_normalized_pnum
    }
    resolved_matrix_cols = [col for col in ordered_plant_cols if col in resolved_source_cols_by_matrix_col]
    available_source_cols = list(dict.fromkeys(resolved_source_cols_by_matrix_col.values()))
    if not available_source_cols:
        raise ValueError(
            f"No intensity columns matched plant tissues for '{plant_key}' in {intensity_path.name}."
        )

    protein_ids = reference_table.index.to_series().map(lambda label: str(label).rsplit(" / ", 2)[1])
    target_proteins = set(protein_ids)

    protein_col = None
    if "Protein ID" in intensity_df.columns:
        protein_col = "Protein ID"
    elif "Protein" in intensity_df.columns:
        protein_col = "Protein"
    if protein_col is None:
        raise ValueError(
            f"Could not find a protein identifier column in {intensity_path.name} (expected 'Protein ID' or 'Protein')."
        )

    prot = intensity_df[[protein_col]].copy()
    prot = prot.rename(columns={protein_col: "Protein ID"})
    prot["row_id"] = prot.index
    prot["Protein ID"] = prot["Protein ID"].astype(str)

    if "Indistinguishable Proteins" in intensity_df.columns:
        extra = intensity_df[["Indistinguishable Proteins"]].dropna().copy()
        extra["row_id"] = extra.index
        extra = extra.rename(columns={"Indistinguishable Proteins": "Protein ID"})
        prot = pd.concat([prot, extra], ignore_index=True)

    prot["Protein ID"] = prot["Protein ID"].str.replace(";", ",", regex=False)
    prot["Protein ID"] = prot["Protein ID"].str.split(",")
    prot = prot.explode("Protein ID")
    prot["Protein ID"] = prot["Protein ID"].astype(str).str.strip()
    prot = prot[prot["Protein ID"].isin(target_proteins)].copy()

    if prot.empty:
        raise ValueError(
            f"None of the selected plant proteins were found in {intensity_path.name}."
        )

    values = intensity_df[available_source_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    values["row_id"] = intensity_df.index

    merged = prot.merge(values, on="row_id", how="left")
    per_protein = merged.groupby("Protein ID", as_index=True)[available_source_cols].max()

    # Keep only matrix tissue columns that are resolvable to source intensity columns;
    # unresolved columns would be all zeros and render as empty stripes in heatmaps.
    out = pd.DataFrame(0.0, index=reference_table.index, columns=resolved_matrix_cols)
    source_to_matrix_cols: Dict[str, List[str]] = {}
    for matrix_col, src_col in resolved_source_cols_by_matrix_col.items():
        source_to_matrix_cols.setdefault(src_col, []).append(matrix_col)

    for label in out.index:
        protein_id = str(label).rsplit(" / ", 2)[1]
        if protein_id not in per_protein.index:
            continue
        for src_col in available_source_cols:
            value = float(per_protein.at[protein_id, src_col])
            for matrix_col in source_to_matrix_cols.get(src_col, []):
                out.at[label, matrix_col] = value

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
    parser.add_argument(
        "--intensities",
        default=None,
        help="Optional directory containing per-plant *.proteins.tsv files used for original MaxLFQ outputs",
    )
    parser.add_argument(
        "--input-fasta",
        default=None,
        help="Optional input FASTA used to preserve original search hit order for exported TSVs",
    )
    parser.add_argument(
        "--search-order-tsv",
        default=None,
        help="Optional TSV with columns Hormone, Name, Locus, Function to order rows; unmatched proteins are placed at the end",
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
    intensities_dir: str | None,
    tissue_ontology_path: str | None,
    outlier_tissue_keys: Set[Tuple[str, str, str]],
    matrix_dir: Path,
    output_dir: Path | None,
    output_png: str | None,
    output_tsv: str | None,
    plant_display_name: str | None,
    input_fasta: str | None,
    search_order_rank: Dict[str, int] | None,
    plant_aliases: Dict[str, Set[str]] | None,
    max_rows: int,
    figscale: float,
    max_label_len: int,
) -> None:
    plant_norm = _norm_text(plant_display_name or plant_key)
    ordered_plant_cols = sort_plant_cols_by_pnumber(
        plant_cols=plant_cols,
        tissue_ontology_path=tissue_ontology_path,
        plant_display_name=plant_display_name,
        plant_key=plant_key,
        plant_aliases=plant_aliases,
    )

    full_table = build_heatmap_table(
        matrix_df=matrix_df,
        matrix_og_col=matrix_og_col,
        orth_df=orth_df,
        orth_og_col=orth_og_col,
        plant_key=plant_key,
        plant_cols=ordered_plant_cols,
        plant_protein_col=plant_protein_col,
        include_zero_rows=True,
        sort_by_total_intensity=False,
    )

    detected_mask = full_table[ordered_plant_cols].sum(axis=1) > 0
    table = full_table.loc[detected_mask].copy()
    if table.empty:
        raise ValueError(
            f"No detected proteins for plant '{plant_key}' (all selected intensities are 0)."
        )

    if search_order_rank:
        table = sort_table_by_search_order_rank(table, search_order_rank)
    else:
        table = sort_table_by_function_category(table)

    if max_rows > 0 and len(table) > max_rows:
        table = table.iloc[: max_rows].copy()

    full_original_table = full_table.copy()
    if intensities_dir:
        full_original_table = build_original_maxlfq_table(
            reference_table=full_table,
            ordered_plant_cols=ordered_plant_cols,
            intensities_dir=intensities_dir,
            tissue_ontology_path=tissue_ontology_path,
            plant_display_name=plant_display_name,
            plant_key=plant_key,
            plant_aliases=plant_aliases,
        )
    original_table = full_original_table.loc[table.index].copy()

    # Keep one consistent tissue-column set across all outputs.
    effective_plant_cols = list(original_table.columns) if intensities_dir else list(ordered_plant_cols)
    table = table[effective_plant_cols].copy()
    original_table = original_table[effective_plant_cols].copy()

    pretty_cols = []
    outlier_col_indices = set()
    for col in effective_plant_cols:
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

    original_plot_table = original_table.copy()
    original_plot_table.columns = pretty_cols

    all_hits_original_table = build_aligned_all_hits_table(
        full_original_table=full_original_table,
        orth_df=orth_df,
        orth_og_col=orth_og_col,
        plant_protein_col=plant_protein_col,
        input_fasta=input_fasta,
    )

    display_name = plant_display_name or plant_key
    safe_plant = safe_filename(display_name)
    base_dir = output_dir if output_dir else matrix_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    out_png = Path(output_png) if output_png else base_dir / f"{safe_plant}.png"
    out_tsv = Path(output_tsv) if output_tsv else base_dir / f"{safe_plant}_plant_tissues_maxlfq_no_edits.tsv"
    out_tsv_all_hits = base_dir / f"{safe_plant}_plant_tissues_all_hits_in_fasta_order.tsv"

    out_png_q1_q3 = out_png.with_name(f"{safe_plant}_q1_q3_capped{out_png.suffix}")
    out_png_row_minmax = out_png.with_name(
        f"{safe_plant}_min_max{out_png.suffix}"
    )
    out_png_orthogroup_collapsed = out_png.with_name(
        f"{safe_plant}_collapsed{out_png.suffix}"
    )

    original_table.to_csv(out_tsv, sep="\t")
    all_hits_original_table.to_csv(out_tsv_all_hits, sep="\t")

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
        raw_row_labels = [str(v) for v in data.index]
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

        for idx, tick in enumerate(ax.get_yticklabels()):
            if idx >= len(raw_row_labels):
                continue
            label = raw_row_labels[idx]
            category = function_category_from_row_label(label)
            tick.set_color(FUNCTION_CATEGORY_COLORS.get(category, FUNCTION_CATEGORY_COLORS["Unknown"]))
            tick.set_fontweight("bold")

        present_categories = []
        for label in raw_row_labels:
            category = function_category_from_row_label(label)
            if category not in present_categories:
                present_categories.append(category)

        legend_handles = []
        for category in FUNCTION_CATEGORY_ORDER + ["Unknown"]:
            if category not in present_categories:
                continue
            color = FUNCTION_CATEGORY_COLORS.get(category, FUNCTION_CATEGORY_COLORS["Unknown"])
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="",
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=7,
                    label=category,
                )
            )
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                title="Function",
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                frameon=False,
                borderaxespad=0.0,
            )

        ax.tick_params(axis="y", pad=2)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    # 1) Original per-protein MaxLFQ values from the source species TSV, displayed as log10(MaxLFQ+1).
    display_current = np.log10(original_plot_table + 1.0)
    save_heatmap(
        data=display_current,
        output_path=out_png,
        title=f"Plant Tissue log10(MaxLFQ+1): {plant_key}\n(y = search_protein_id / plant_protein_id / orthogroup)",
        cbar_label="log10(MaxLFQ+1)",
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
    # Exclude zeros per row when computing min/max so undetected tissues
    # do not compress the dynamic range of detected values.
    display_log_plus1_nonzero = display_log_plus1.mask(plot_table <= 0)
    row_mins = display_log_plus1_nonzero.min(axis=1)
    row_maxs = display_log_plus1_nonzero.max(axis=1)
    row_ranges = (row_maxs - row_mins).replace(0, np.nan)
    display_row_minmax = (
        display_log_plus1_nonzero.sub(row_mins, axis=0).div(row_ranges, axis=0).fillna(0.0)
    )
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
    collapsed = plot_table.copy()
    collapsed_parts = [split_y_label_parts(label) for label in table.index]
    collapsed["_search_ids"] = [search_ids for search_ids, _, _ in collapsed_parts]
    collapsed["_orthogroup"] = [orthogroup for _, _, orthogroup in collapsed_parts]
    collapsed = collapsed.drop_duplicates(subset=["_orthogroup"] + list(plot_table.columns))
    collapsed["_collapsed_label"] = collapsed["_search_ids"].astype(str) + "/" + collapsed["_orthogroup"].astype(str)
    collapsed = collapsed.set_index("_collapsed_label")
    collapsed = collapsed.drop(columns=["_search_ids", "_orthogroup"])
    collapsed["_total_intensity"] = collapsed.sum(axis=1)
    collapsed = collapsed.sort_values(by="_total_intensity", ascending=False).drop(columns=["_total_intensity"])
    display_collapsed = np.log10(collapsed.mask(collapsed <= 0))
    save_heatmap(
        data=display_collapsed,
        output_path=out_png_orthogroup_collapsed,
        title=f"Plant Tissue log10 Intensities, orthogroup deduplicated: {plant_key}\n(y = search_protein_id/orthogroup)",
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
    print(f"Saved all-hits raw maxLFQ table in FASTA order: {out_tsv_all_hits}")
    print(f"Saved heatmap (original MaxLFQ+1): {out_png}")
    print(f"Saved heatmap (Q1-Q3 capped): {out_png_q1_q3}")
    print(f"Saved heatmap (row min-max): {out_png_row_minmax}")
    print(f"Saved heatmap (orthogroup-collapsed): {out_png_orthogroup_collapsed}")


def main() -> None:
    args = parse_args()
    plant_aliases = load_plant_aliases()

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
    search_order_rank = build_tsv_order_rank(args.search_order_tsv) if args.search_order_tsv else {}

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
                    intensities_dir=args.intensities,
                    tissue_ontology_path=tissue_ontology_path,
                    outlier_tissue_keys=outlier_tissue_keys,
                    matrix_dir=matrix_dir,
                    output_dir=output_dir,
                    output_png=None,
                    output_tsv=None,
                    plant_display_name=plant_display_name,
                    input_fasta=args.input_fasta,
                    search_order_rank=search_order_rank,
                    plant_aliases=plant_aliases,
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
        intensities_dir=args.intensities,
        tissue_ontology_path=tissue_ontology_path,
        outlier_tissue_keys=outlier_tissue_keys,
        matrix_dir=matrix_dir,
        output_dir=output_dir,
        output_png=args.output_png,
        output_tsv=args.output_tsv,
        plant_display_name=plant_display_name,
        input_fasta=args.input_fasta,
        search_order_rank=search_order_rank,
        plant_aliases=plant_aliases,
        max_rows=args.max_rows,
        figscale=args.figscale,
        max_label_len=args.max_label_len,
    )


if __name__ == "__main__":
    main()
