#!/usr/bin/env python3
"""Map one query FASTA to the best protein in every crop family and its OG.

Each query/family pair is searched normally first.  Pairs without any
qualifying hit are searched a second time in sensitive mode with sequence
masking disabled.  The fallback retains the same query-coverage and E-value
requirements and can never replace a hit from the normal search.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def fasta_ids(path: Path) -> list[str]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                result.append(line[1:].strip().split()[0])
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate query IDs in {path}")
    return result


def search_family(family: str, database: Path, query: Path, output: Path,
                  threads: int, query_cover: int,
                  sensitive_unmasked: bool = False) -> tuple[str, Path]:
    command = [
        "diamond", "blastp", "--query", str(query), "--db", str(database),
        "--out", str(output), "--outfmt", "6", "qseqid", "sseqid",
        "bitscore", "evalue", "qcovhsp", "--evalue", "1e-5",
        "--query-cover", str(query_cover), "--max-target-seqs", "25",
        "--threads", str(threads), "--quiet",
    ]
    if sensitive_unmasked:
        command.extend(["--sensitive", "--masking", "0"])
    subprocess.run(command, check=True)
    return family, output


def fasta_records(path: Path) -> dict[str, str]:
    """Return FASTA records keyed by the first token of each header."""
    records: dict[str, str] = {}
    header = ""
    lines: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if header:
                    records[header[1:].strip().split()[0]] = header + "".join(lines)
                header = line
                lines = []
            else:
                lines.append(line)
        if header:
            records[header[1:].strip().split()[0]] = header + "".join(lines)
    return records


def write_fasta_subset(records: dict[str, str], wanted: list[str], path: Path) -> None:
    """Write selected records while retaining their original FASTA text."""
    with path.open("w", encoding="utf-8") as handle:
        for query in wanted:
            handle.write(records[query])


def update_best_hits(path: Path, family: str,
                     best: dict[tuple[str, str], tuple[float, str, float, float]],
                     targets: set[str], search_modes: dict[tuple[str, str], str],
                     search_mode: str, missing_only: bool = False) -> int:
    """Read DIAMOND output and update the best hit for each query/family."""
    added = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for query, target, bitscore, evalue, coverage in csv.reader(
                handle, delimiter="\t"):
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
    parser.add_argument("--family-db-dir", required=True, type=Path)
    parser.add_argument("--orthogroup-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--best-hits-output", type=Path,
                        help="Optional headerless best hit per query for heatmap labels")
    parser.add_argument("--hit-dir", required=True, type=Path)
    parser.add_argument("--discard-raw-hits", action="store_true",
                        help="Remove per-family DIAMOND TSVs after writing the summary")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-search", type=int, default=1)
    parser.add_argument("--query-cover", type=int, default=50)
    parser.add_argument(
        "--disable-low-complexity-fallback", action="store_true",
        help=("Do not retry query/family pairs lacking a normal hit with "
              "DIAMOND --sensitive --masking 0"),
    )
    args = parser.parse_args()

    queries = fasta_ids(args.query_fasta)
    metadata = pd.read_csv(args.metadata, sep="\t", dtype=str).fillna("")
    families = sorted(value for value in metadata["Family"].unique() if value.strip())
    databases = {family: args.family_db_dir / f"{safe_name(family)}.dmnd"
                 for family in families}
    missing = [str(path) for path in databases.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Family DIAMOND databases are missing; build/copy them first: "
            + ", ".join(missing[:5])
        )

    args.hit_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for family, database in databases.items():
            output = args.hit_dir / f"{safe_name(family)}.tsv"
            futures.append(pool.submit(
                search_family, family, database, args.query_fasta, output,
                args.threads_per_search, args.query_cover,
            ))
        for future in as_completed(futures):
            family, _ = future.result()
            print(f"Finished {family}", flush=True)

    best: dict[tuple[str, str], tuple[float, str, float, float]] = {}
    targets: set[str] = set()
    search_modes: dict[tuple[str, str], str] = {}
    for family in families:
        path = args.hit_dir / f"{safe_name(family)}.tsv"
        update_best_hits(
            path, family, best, targets, search_modes, "normal",
        )

    normal_hit_count = len(best)
    fallback_attempted = 0
    fallback_rescued = 0
    if not args.disable_low_complexity_fallback:
        records = fasta_records(args.query_fasta)
        fallback_dir = args.hit_dir / "sensitive_unmasked_fallback"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        missing_by_family: dict[str, list[str]] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            for family, database in databases.items():
                missing_queries = [
                    query for query in queries if (query, family) not in best
                ]
                if not missing_queries:
                    continue
                missing_by_family[family] = missing_queries
                fallback_attempted += len(missing_queries)
                query_path = fallback_dir / f"{safe_name(family)}.faa"
                output = fallback_dir / f"{safe_name(family)}.tsv"
                write_fasta_subset(records, missing_queries, query_path)
                futures.append(pool.submit(
                    search_family, family, database, query_path, output,
                    args.threads_per_search, args.query_cover, True,
                ))
            for future in as_completed(futures):
                family, _ = future.result()
                print(f"Finished low-complexity fallback: {family}", flush=True)

        for family in missing_by_family:
            path = fallback_dir / f"{safe_name(family)}.tsv"
            fallback_rescued += update_best_hits(
                path, family, best, targets, search_modes,
                "sensitive_unmasked_fallback", missing_only=True,
            )

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
                    "query_coverage_pct": pd.NA,
                    "diamond_search_mode": "none",
                })
                continue
            bitscore, target, evalue, coverage = hit
            status, orthogroup = locations.get(target, ("missing", ""))
            rows.append({
                "query": query, "family": family, "target_protein": target,
                "status": status, "orthogroup": orthogroup,
                "bitscore": bitscore, "evalue": evalue,
                "query_coverage_pct": coverage,
                "diamond_search_mode": search_modes[(query, family)],
            })
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="\t", index=False)
    if args.best_hits_output is not None:
        assigned_hits = output.loc[output["status"].eq("assigned")].copy()
        assigned_hits = assigned_hits.sort_values(
            ["query", "bitscore"], ascending=[True, False]
        ).drop_duplicates("query")
        args.best_hits_output.parent.mkdir(parents=True, exist_ok=True)
        assigned_hits[["query", "target_protein", "bitscore", "evalue",
                       "query_coverage_pct"]].to_csv(
            args.best_hits_output, sep="\t", index=False, header=False
        )
    print(f"Wrote {args.output}")
    print(f"Queries: {len(queries)}; families: {len(families)}; rows: {len(output)}")
    print(f"Normal qualifying query-family hits: {normal_hit_count}")
    if not args.disable_low_complexity_fallback:
        print(f"Fallback query-family pairs searched: {fallback_attempted}")
        print(f"Fallback query-family pairs rescued: {fallback_rescued}")
    print(f"Assigned family hits: {output['status'].eq('assigned').sum()}")
    if args.discard_raw_hits:
        shutil.rmtree(args.hit_dir)


if __name__ == "__main__":
    main()
