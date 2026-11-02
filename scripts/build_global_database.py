#!/usr/bin/env python3
"""Build the full global DIAMOND database and target-to-family index."""

from __future__ import annotations

import argparse
import csv
import gzip
import subprocess
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--fasta-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata, sep="\t", dtype=str).fillna("")
    required = {"Family", "FASTA"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Metadata must contain columns: {sorted(required)}")

    input_families: dict[Path, str] = {}
    for family, filename in metadata[["Family", "FASTA"]].itertuples(index=False):
        family, filename = str(family).strip(), str(filename).strip()
        if not family or not filename:
            continue
        path = args.fasta_dir / Path(filename).name
        if not path.is_file():
            raise FileNotFoundError(f"Reference FASTA missing: {path}")
        previous_family = input_families.setdefault(path, family)
        if previous_family != family:
            raise ValueError(
                f"Reference FASTA is assigned to multiple families: {path} "
                f"({previous_family}, {family})")
    inputs = sorted((family, path) for path, family in input_families.items())
    if not inputs:
        raise ValueError("Metadata contains no usable Family/FASTA rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = args.output_dir / "all_family_proteins_full.faa"
    family_map = args.output_dir / "all_family_proteins.target_to_family.tsv.gz"
    database = args.output_dir / "all_family_proteins_full"
    seen: dict[str, str] = {}

    with combined.open("w", encoding="utf-8") as output, gzip.open(
            family_map, "wt", encoding="utf-8", newline="") as map_handle:
        writer = csv.writer(map_handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["target_protein", "family"])
        for family, fasta in inputs:
            last = ""
            with fasta.open(encoding="utf-8") as source:
                for line in source:
                    last = line
                    if line.startswith(">"):
                        target = line[1:].strip().split()[0]
                        previous = seen.get(target)
                        if previous is not None:
                            raise ValueError(
                                f"Duplicate protein ID: {target} "
                                f"({previous}, {family})")
                        seen[target] = family
                        writer.writerow([target, family])
                    output.write(line)
            if last and not last.endswith("\n"):
                output.write("\n")

    # The index must contain one row per database sequence. Duplicate IDs are
    # rejected because post-search family assignment would otherwise be ambiguous.
    with gzip.open(family_map, "rt", encoding="utf-8") as handle:
        next(handle)
        row_count = sum(1 for _ in handle)
    if row_count != len(seen):
        raise ValueError(
            "Duplicate protein IDs detected in reference FASTAs: "
            f"records={row_count}, unique_ids={len(seen)}")

    subprocess.run([
        "diamond", "makedb", "--in", str(combined), "--db", str(database),
        "--threads", str(args.threads),
    ], check=True)
    print(f"Built global database with {len(seen)} proteins from {len(inputs)} FASTAs")
    print(f"Database: {database}.dmnd")
    print(f"Target-family map: {family_map}")


if __name__ == "__main__":
    main()
