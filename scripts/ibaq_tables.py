#!/usr/bin/env python3
"""Compute per-tissue iBAQ tables for FragPipe search results, one plant at a time.

Port of nbs/ibaq_tissues.ipynb (originally Banana-only) to every plant under the
FinalFragger results root. For each plant four tables are written:

  1. <Plant>_raw_intensity.tsv        combined_protein.tsv "<sample> Intensity", proteins x samples
  2. <Plant>_ibaq.tsv                 intensity / observable tryptic peptides
  3. <Plant>_ibaq_tic_normalized.tsv  columns scaled so every column sum == median column sum
  4. <Plant>_ibaq_median_centered.tsv (3) further scaled so every column median == median of medians

Every table carries the combined_protein.tsv "Indistinguishable Proteins" annotation
as its first column, ahead of the sample columns.

The iBAQ denominator is the number of theoretically observable tryptic peptides:
fully cleaved (0 missed cleavages), length MIN_LEN..MAX_LEN, stricttrypsin rule
(cut after every K/R, including before proline) to match the FragPipe search.
The protein universe is the non-decoy set of combined_protein.tsv.

Examples:
    python scripts/ibaq_tables.py                       # every plant, serial
    python scripts/ibaq_tables.py --jobs 8              # every plant, 8 workers
    python scripts/ibaq_tables.py --plants Banana Rice  # named plants only
"""

import argparse
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_ROOT = Path("/cmnfs/ENB/search_results/FinalFragger")
REPO_DIR = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_DIR / "ibaq_tables"

SEARCH_SUBDIR = "FragPipeSearchResults"
PARAMS_NAME = "fragger_dda_plus.params"

MIN_LEN, MAX_LEN = 6, 30  # keep tryptic peptides in this length range (inclusive)

INDIST_COL = "Indistinguishable Proteins"  # carried through to every output table

TABLE_SUFFIXES = (
    "raw_intensity",
    "ibaq",
    "ibaq_total_sum_corrected",
    "ibaq_total_sum_corrected_median_centered",
)


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def find_plants(results_root):
    """Plant names under `results_root` that have a usable search directory."""
    plants = []
    for child in sorted(results_root.iterdir()):
        if child.is_dir() and (child / SEARCH_SUBDIR / "combined_protein.tsv").is_file():
            plants.append(child.name)
    return plants


def sample_key(name):
    """Sort samples by their trailing _<n> index (P087149_1 < P087158_10)."""
    m = re.search(r"_(\d+)$", name)
    return (0, int(m.group(1)), name) if m else (1, 0, name)


def intensity_columns(columns):
    """{sample: column} for the per-sample "<sample> Intensity" columns, sample-index order.

    MaxLFQ columns end in the same word and are excluded: iBAQ needs the plain
    summed intensity, since MaxLFQ is already a ratio-based relative quantity.
    """
    found = {}
    for col in columns:
        if col.endswith(" MaxLFQ Intensity"):
            continue
        m = re.match(r"^(.+) Intensity$", col)
        if m:
            found[m.group(1)] = col
    return {s: found[s] for s in sorted(found, key=sample_key)}


def resolve_fasta(search_dir):
    """The search FASTA named in the FragPipe params, so it always matches results."""
    params_path = search_dir / PARAMS_NAME
    if not params_path.is_file():
        raise FileNotFoundError(f"missing {params_path}")
    m = re.search(r"^database_name\s*=\s*(\S+)", params_path.read_text(), re.MULTILINE)
    if not m:
        raise ValueError(f"no database_name in {params_path}")
    fasta = Path(m.group(1))
    if not fasta.is_file():
        raise FileNotFoundError(f"FASTA from params does not exist: {fasta}")
    return fasta


# --------------------------------------------------------------------------- #
# in-silico digestion -> iBAQ denominator
# --------------------------------------------------------------------------- #
def read_fasta(path, keep):
    """{header_id: sequence} for entries whose id is in `keep`."""
    seqs, pid, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if pid in keep:
                    seqs[pid] = "".join(buf)
                pid = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if pid in keep:
        seqs[pid] = "".join(buf)
    return seqs


def digest_strict(seq):
    """Strict tryptic digest: cut C-terminal to every K/R (including before proline)."""
    cuts = [i + 1 for i, aa in enumerate(seq) if aa in "KR"]
    bounds = [0, *cuts, len(seq)]
    return [seq[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]


def n_observable(seq, lo=MIN_LEN, hi=MAX_LEN, unique=False):
    """Observable tryptic peptides; unique=False counts all peptides (original iBAQ)."""
    peps = [p for p in digest_strict(seq) if lo <= len(p) <= hi]
    return len(set(peps)) if unique else len(peps)


# --------------------------------------------------------------------------- #
# normalization
# --------------------------------------------------------------------------- #
def scale_columns(frame, stats, plant, what):
    """Divide each column by stat/median(stats) -> every column stat == the median stat."""
    target = stats.median()
    factors = stats / target
    bad = factors[~np.isfinite(factors) | (factors <= 0)]
    if len(bad):
        # Left as NaN rather than guessed at: a non-positive column stat means the
        # column is empty or degenerate, and any fill-in would invent signal.
        print(
            f"[{plant}] WARNING: non-positive/non-finite {what} normalization factor for "
            f"{len(bad)} sample(s): {', '.join(bad.index)} -> columns become NaN",
            file=sys.stderr,
        )
        factors = factors.mask(~np.isfinite(factors) | (factors <= 0))
    return frame.divide(factors, axis=1)


# --------------------------------------------------------------------------- #
# per-plant pipeline
# --------------------------------------------------------------------------- #
def build_tables(plant, results_root, min_len, max_len, unique):
    """The four tables for one plant, as {suffix: DataFrame}."""
    search_dir = results_root / plant / SEARCH_SUBDIR
    combined_path = search_dir / "combined_protein.tsv"

    header = pd.read_csv(combined_path, sep="\t", nrows=0).columns
    samples = intensity_columns(header)
    if not samples:
        raise ValueError(f"no '<sample> Intensity' columns in {combined_path}")

    fasta_path = resolve_fasta(search_dir)

    # Protein universe and intensities both come from combined_protein.tsv, keyed on
    # the full "Protein" header form, which matches the FASTA headers.
    combined = pd.read_csv(
        combined_path,
        sep="\t",
        low_memory=False,
        usecols=["Protein", INDIST_COL, *samples.values()],
    )
    combined = combined.loc[~combined["Protein"].str.startswith("rev_")]
    detected = set(combined["Protein"])
    if not detected:
        raise ValueError(f"no non-decoy proteins in {combined_path}")

    seqs = read_fasta(fasta_path, detected)
    if len(seqs) < len(detected):
        print(
            f"[{plant}] WARNING: {len(detected) - len(seqs)} detected protein(s) absent "
            f"from {fasta_path.name}; their iBAQ is NaN",
            file=sys.stderr,
        )

    pep_counts = pd.Series(
        {pid: n_observable(s, min_len, max_len, unique) for pid, s in seqs.items()},
        dtype="float64",
    )
    n_zero = int((pep_counts == 0).sum())
    if n_zero:
        print(
            f"[{plant}] WARNING: {n_zero} protein(s) have 0 observable peptides; " f"their iBAQ is NaN",
            file=sys.stderr,
        )
    pep_counts_nz = pep_counts.replace(0, np.nan)  # never divide by zero

    # Numerator: per-sample Intensity -> proteins x samples.
    intensity = combined.set_index("Protein")[list(samples.values())]
    intensity.columns = list(samples)
    intensity = intensity.reindex(sorted(detected))
    intensity.index.name = "Protein"

    # combined_protein.tsv writes 0 for a protein it did not quantify in a run, which is
    # missingness rather than a measured zero. Kept as NaN so it stays out of the column
    # sums and medians below: ~45% of entries are 0, enough that several samples per plant
    # would otherwise have a zero median and be scaled to an all-NaN column.
    intensity = intensity.replace(0, np.nan)

    ibaq = intensity.div(pep_counts_nz.reindex(intensity.index), axis=0)
    corrected_ibaq = scale_columns(ibaq, ibaq.sum(axis=0), plant, "column-sum")
    centered = scale_columns(corrected_ibaq, corrected_ibaq.median(axis=0), plant, "column-median")

    # Prepended only now, so the annotation never reaches the sums, medians or division.
    indistinguishable = combined.set_index("Protein")[INDIST_COL].reindex(intensity.index)

    def annotated(frame):
        return frame.copy().assign(**{INDIST_COL: indistinguishable})[[INDIST_COL, *frame.columns]]

    tables = (intensity, ibaq, corrected_ibaq, centered)
    return dict(zip(TABLE_SUFFIXES, (annotated(f) for f in tables)))


def run_plant(plant, results_root, out_root, min_len, max_len, unique, skip_existing):
    """Write one plant's four tables. Returns a status dict (never raises)."""
    out_dir = out_root / plant
    paths = {s: out_dir / f"{i}_{plant}_{s}.tsv" for i, s in enumerate(TABLE_SUFFIXES)}

    if skip_existing and all(p.is_file() for p in paths.values()):
        return {
            "plant": plant,
            "status": "skipped",
            "proteins": "",
            "samples": "",
            "detail": "all 4 tables already present",
        }

    try:
        tables = build_tables(plant, results_root, min_len, max_len, unique)
        out_dir.mkdir(parents=True, exist_ok=True)
        for suffix, frame in tables.items():
            frame.to_csv(paths[suffix], sep="\t")
        ibaq = tables["ibaq"]
        n_proteins = ibaq.shape[0]
        n_samples = ibaq.columns.drop(INDIST_COL).size
        print(f"[{plant}] OK  {n_proteins} proteins x {n_samples} samples -> {out_dir}")
        return {"plant": plant, "status": "ok", "proteins": n_proteins, "samples": n_samples, "detail": ""}
    except Exception as exc:
        print(f"[{plant}] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {
            "plant": plant,
            "status": "failed",
            "proteins": "",
            "samples": "",
            "detail": f"{type(exc).__name__}: {exc}",
        }


def main():
    ap = argparse.ArgumentParser(
        description="Compute per-tissue iBAQ tables for every plant in the FragPipe results root.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--results-root", type=Path, default=RESULTS_ROOT, help="root holding <Plant>/FragPipeSearchResults"
    )
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT, help="output root; one subdirectory per plant")
    ap.add_argument("--plants", nargs="*", metavar="PLANT", help="plants to process (default: all discovered)")
    ap.add_argument("--jobs", type=int, default=1, help="plants to process in parallel")
    ap.add_argument("--min-len", type=int, default=MIN_LEN, help="min tryptic peptide length")
    ap.add_argument("--max-len", type=int, default=MAX_LEN, help="max tryptic peptide length")
    ap.add_argument(
        "--unique", action="store_true", help="count unique peptides only (default: all peptides, original iBAQ)"
    )
    ap.add_argument("--skip-existing", action="store_true", help="skip plants whose four tables already exist")
    ap.add_argument("--list", action="store_true", help="list discovered plants and exit")
    args = ap.parse_args()

    if not args.results_root.is_dir():
        sys.exit(f"results root does not exist: {args.results_root}")

    available = find_plants(args.results_root)
    if args.list:
        print("\n".join(available))
        return 0

    if args.plants:
        unknown = [p for p in args.plants if p not in available]
        if unknown:
            sys.exit(
                f"unknown plant(s): {', '.join(unknown)}\n" f"run with --list to see the {len(available)} available"
            )
        plants = args.plants
    else:
        plants = available

    if not plants:
        sys.exit(f"no plants with {SEARCH_SUBDIR}/combined_protein.tsv under {args.results_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(plants)} plant(s) -> {args.out_dir} (jobs={args.jobs})")

    work = (args.results_root, args.out_dir, args.min_len, args.max_len, args.unique, args.skip_existing)
    if args.jobs > 1:
        results = []
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_plant, p, *work) for p in plants]
            for fut in as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda r: plants.index(r["plant"]))
    else:
        results = [run_plant(p, *work) for p in plants]

    summary = pd.DataFrame(results)
    summary_path = args.out_dir / "summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)

    counts = summary["status"].value_counts()
    print("\nDone: " + ", ".join(f"{n} {s}" for s, n in counts.items()))
    print(f"Summary: {summary_path}")

    failed = summary.loc[summary["status"] == "failed", "plant"].tolist()
    if failed:
        print(f"Failed plants: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
