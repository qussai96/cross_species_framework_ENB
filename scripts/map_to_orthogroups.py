#!/usr/bin/env python3

import pandas as pd
import sys
from collections import defaultdict

def split_genes(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [x.strip() for x in str(cell).split(",") if x.strip()]

def load_mapping(mapping_file):
    """
    First column = search_protein_id
    Second column = orthofinder_protein_id
    Ignore remaining columns
    """
    df = pd.read_csv(mapping_file, sep="\t", header=None, dtype=str)
    df = df.iloc[:, :2]  # keep only first two columns
    df.columns = ["search_id", "ortho_id"]

    ortho_to_search = defaultdict(list)
    for _, row in df.iterrows():
        ortho_to_search[row["ortho_id"]].append(row["search_id"])

    return ortho_to_search, set(df["ortho_id"])


def extract_orthogroups(mapping_file, orthogroups_file, output_file):
    ortho_to_search, target_proteins = load_mapping(mapping_file)

    og_df = pd.read_csv(orthogroups_file, sep="\t", dtype=str).fillna("")
    og_col = og_df.columns[0]
    species_cols = og_df.columns[1:]

    results = []

    for _, row in og_df.iterrows():
        matched_search_ids = set()
        found = False

        for col in species_cols:
            proteins = split_genes(row[col])
            for p in proteins:
                if p in target_proteins:
                    found = True
                    matched_search_ids.update(ortho_to_search[p])

        if found:
            new_row = row.copy()
            new_row["search_protein_id"] = ", ".join(sorted(matched_search_ids))
            results.append(new_row)

    if not results:
        print("No matching orthogroups found.")
        pd.DataFrame().to_csv(output_file, sep="\t", index=False)
        return

    out_df = pd.DataFrame(results)

    # place new column after Orthogroup column
    cols = list(out_df.columns)
    cols.insert(1, cols.pop(cols.index("search_protein_id")))
    out_df = out_df[cols]

    out_df.to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_orthogroups.py mapping.tsv Orthogroups.tsv output.tsv")
        sys.exit(1)

    extract_orthogroups(sys.argv[1], sys.argv[2], sys.argv[3])
