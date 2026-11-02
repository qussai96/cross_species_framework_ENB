#!/usr/bin/env python3
"""Map queries to the best protein in every family using one global database.

The normal DIAMOND search reports every qualifying target (``-k 0``). Hits are
partitioned by target family and the highest-bitscore hit is retained for each
query/family pair. Queries missing one or more families are searched once more
globally in sensitive mode without masking; fallback fills only missing pairs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def fasta_ids(path: Path) -> list[str]:
    result = [line[1:].strip().split()[0] for line in path.open(encoding="utf-8")
              if line.startswith(">")]
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate query IDs in {path}")
    return result


def fasta_records(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    header, lines = "", []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if header:
                    records[header[1:].strip().split()[0]] = header + "".join(lines)
                header, lines = line, []
            else:
                lines.append(line)
        if header:
            records[header[1:].strip().split()[0]] = header + "".join(lines)
    return records


def write_fasta_subset(records: dict[str, str], wanted: set[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for query, record in records.items():
            if query in wanted:
                handle.write(record)


def read_target_families(path: Path) -> dict[str, str]:
    opener = gzip.open if path.suffix == ".gz" else open
    result: dict[str, str] = {}
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"target_protein", "family"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        for row in reader:
            target, family = row["target_protein"].strip(), row["family"].strip()
            previous = result.setdefault(target, family)
            if previous != family:
                raise ValueError(
                    f"Protein ID occurs in multiple families: {target} "
                    f"({previous}, {family})")
    return result


def run_search(database: Path, query: Path, output: Path, threads: int,
               query_cover: int, sensitive_unmasked: bool = False) -> None:
    command = [
        "diamond", "blastp", "--query", str(query), "--db", str(database),
        "--out", str(output), "--outfmt", "6", "qseqid", "sseqid",
        "bitscore", "evalue", "qcovhsp", "--evalue", "1e-5",
        "--query-cover", str(query_cover), "--max-target-seqs", "0",
        "--threads", str(threads), "--quiet",
    ]
    if sensitive_unmasked:
        command.extend(["--sensitive", "--masking", "0"])
    subprocess.run(command, check=True)


def update_best_hits(
    path: Path,
    target_family: dict[str, str],
    valid_families: set[str],
    best: dict[tuple[str, str], tuple[float, str, float, float]],
    targets: set[str],
    search_modes: dict[tuple[str, str], str],
    search_mode: str,
    missing_only: bool = False,
) -> int:
    added = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for query, target, bitscore, evalue, coverage in csv.reader(
                handle, delimiter="\t"):
            family = target_family.get(target)
            if family is None:
                raise ValueError(
                    f"DIAMOND target is absent from target-family map: {target}")
            if family not in valid_families:
                raise ValueError(f"Target family absent from metadata: {family}")
            key = (query, family)
            if missing_only and key in best:
                continue
            record = (float(bitscore), target, float(evalue), float(coverage))
            if key not in best:
                added += 1
            if key not in best or record[0] > best[key][0]:
                best[key] = record
                search_modes[key] = search_mode
                targets.add(target)
    return added


def protein_ogs(wanted: set[str], orthogroup_dir: Path):
    locations: dict[str, tuple[str, str]] = {}
    for filename, status in (("Orthogroups.tsv", "assigned"),
                             ("Orthogroups_UnassignedGenes.tsv", "unassigned")):
        with (orthogroup_dir / filename).open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if not row:
                    continue
                og = row[0]
                for cell in row[1:]:
                    for protein in (value.strip() for value in cell.split(",")):
                        if protein in wanted:
                            locations[protein] = (status, og)
    return locations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-fasta", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--global-db", required=True, type=Path)
    parser.add_argument("--target-family-map", required=True, type=Path)
    parser.add_argument("--orthogroup-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--best-hits-output", type=Path)
    parser.add_argument("--hit-dir", required=True, type=Path)
    parser.add_argument("--discard-raw-hits", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--query-cover", type=int, default=50)
    parser.add_argument("--disable-low-complexity-fallback", action="store_true")
    args = parser.parse_args()

    queries = fasta_ids(args.query_fasta)
    metadata = pd.read_csv(args.metadata, sep="\t", dtype=str).fillna("")
    families = sorted(value.strip() for value in metadata["Family"].unique()
                      if value.strip())
    family_set = set(families)
    target_family = read_target_families(args.target_family_map)
    map_families = set(target_family.values())
    if map_families != family_set:
        raise ValueError(
            "Family mismatch between metadata and target-family map; "
            f"metadata_only={sorted(family_set-map_families)}, "
            f"map_only={sorted(map_families-family_set)}")

    args.hit_dir.mkdir(parents=True, exist_ok=True)
    normal_output = args.hit_dir / "global_normal_all_hits.tsv"
    run_search(args.global_db, args.query_fasta, normal_output, args.threads,
               args.query_cover)

    best: dict[tuple[str, str], tuple[float, str, float, float]] = {}
    targets: set[str] = set()
    search_modes: dict[tuple[str, str], str] = {}
    update_best_hits(normal_output, target_family, family_set, best, targets,
                     search_modes, "normal")
    normal_hit_count = len(best)

    missing_pairs = {(query, family) for query in queries for family in families
                     if (query, family) not in best}
    fallback_attempted = len(missing_pairs)
    fallback_rescued = 0
    if missing_pairs and not args.disable_low_complexity_fallback:
        fallback_query = args.hit_dir / "fallback_queries.faa"
        fallback_output = args.hit_dir / "global_sensitive_unmasked_all_hits.tsv"
        write_fasta_subset(fasta_records(args.query_fasta),
                           {query for query, _family in missing_pairs},
                           fallback_query)
        run_search(args.global_db, fallback_query, fallback_output, args.threads,
                   args.query_cover, sensitive_unmasked=True)
        fallback_rescued = update_best_hits(
            fallback_output, target_family, family_set, best, targets,
            search_modes, "sensitive_unmasked_fallback", missing_only=True)

    locations = protein_ogs(targets, args.orthogroup_dir)
    rows = []
    for query in queries:
        for family in families:
            hit = best.get((query, family))
            if hit is None:
                rows.append({
                    "query": query, "family": family, "target_protein": "",
                    "status": "no_qualifying_hit", "orthogroup": "",
                    "bitscore": pd.NA, "evalue": pd.NA,
                    "query_coverage_pct": pd.NA, "diamond_search_mode": "none"})
                continue
            bitscore, target, evalue, coverage = hit
            status, orthogroup = locations.get(target, ("missing", ""))
            rows.append({
                "query": query, "family": family, "target_protein": target,
                "status": status, "orthogroup": orthogroup,
                "bitscore": bitscore, "evalue": evalue,
                "query_coverage_pct": coverage,
                "diamond_search_mode": search_modes[(query, family)]})

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="\t", index=False)
    if args.best_hits_output is not None:
        assigned_hits = output.loc[output["status"].eq("assigned")].copy()
        assigned_hits = assigned_hits.sort_values(
            ["query", "bitscore"], ascending=[True, False]).drop_duplicates("query")
        args.best_hits_output.parent.mkdir(parents=True, exist_ok=True)
        assigned_hits[["query", "target_protein", "bitscore", "evalue",
                       "query_coverage_pct"]].to_csv(
            args.best_hits_output, sep="\t", index=False, header=False)

    print(f"Wrote {args.output}")
    print(f"Queries: {len(queries)}; families: {len(families)}; rows: {len(output)}")
    print(f"Global database target IDs indexed: {len(target_family)}")
    print(f"Normal qualifying query-family hits: {normal_hit_count}")
    if not args.disable_low_complexity_fallback:
        print(f"Fallback query-family pairs searched: {fallback_attempted}")
        print(f"Fallback query-family pairs rescued: {fallback_rescued}")
    print(f"Assigned family hits: {output['status'].eq('assigned').sum()}")
    if args.discard_raw_hits:
        shutil.rmtree(args.hit_dir)


if __name__ == "__main__":
    main()
