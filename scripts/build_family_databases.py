#!/usr/bin/env python3
"""Build one DIAMOND protein database per plant family in the metadata."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def fasta_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [line[1:].strip().split()[0] for line in handle if line.startswith(">")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--fasta-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata, sep="\t", dtype=str).fillna("")
    if not {"Family", "FASTA"}.issubset(metadata.columns):
        raise ValueError("Metadata must contain Family and FASTA columns")
    by_family: dict[str, list[Path]] = defaultdict(list)
    for family, filename in metadata[["Family", "FASTA"]].itertuples(index=False):
        family, filename = str(family).strip(), str(filename).strip()
        if not family or not filename:
            continue
        path = args.fasta_dir / Path(filename).name
        if not path.is_file():
            raise FileNotFoundError(f"Reference FASTA missing: {path}")
        by_family[family].append(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for family in sorted(by_family):
        stem = safe_name(family)
        combined = args.output_dir / f"{stem}.faa"
        seen: set[str] = set()
        with combined.open("w", encoding="utf-8") as output:
            for fasta in by_family[family]:
                identifiers = fasta_ids(fasta)
                duplicate = seen.intersection(identifiers)
                if duplicate:
                    raise ValueError(
                        f"Duplicate protein IDs in {family}: {sorted(duplicate)[:5]}"
                    )
                seen.update(identifiers)
                text = fasta.read_text(encoding="utf-8")
                output.write(text)
                if text and not text.endswith("\n"):
                    output.write("\n")
        subprocess.run([
            "diamond", "makedb", "--in", str(combined),
            "--db", str(args.output_dir / stem), "--threads", str(args.threads),
        ], check=True)
        print(f"Built {family}: {len(seen)} proteins")


if __name__ == "__main__":
    main()
