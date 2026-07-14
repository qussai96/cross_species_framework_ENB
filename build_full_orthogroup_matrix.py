#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from protein_orthogroup_po_matrix import build_matrix_fast, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the full orthogroup × plant/tissue intensity matrices."
    )
    parser.add_argument("--orthogroups", required=True, help="Full Orthogroups.tsv path")
    parser.add_argument("--intensities", required=True, help="Directory with *.proteins.tsv intensity files")
    parser.add_argument("--tissue-ontology", required=True, help="Tissue ontology TSV path")
    parser.add_argument("--metadata", required=True, help="all_crops_metadata.tsv path")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the generated matrices (default: directory of --output or current directory)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional legacy path for the MaxLFQ matrix TSV; the filename is ignored when --output-dir is used",
    )
    parser.add_argument(
        "--output-raw",
        default=None,
        help="Optional legacy output path for the raw intensity matrix TSV",
    )
    parser.add_argument(
        "--og-annotation-table",
        default=None,
        help="Optional orthogroup annotation table used for descriptions",
    )
    parser.add_argument("--log", default=None, help="Optional log file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.output:
        output_dir = Path(args.output).resolve().parent
    else:
        output_dir = Path.cwd()

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_output = Path(args.output_raw) if args.output_raw else output_dir / "protein_orthogroup_po_matrix_raw_intensities.csv"
    maxlfq_output = output_dir / "protein_orthogroup_po_matrix_maxlfq_intensities.csv"
    og_desc_path = output_dir / "OG_Desc.tsv"

    build_matrix_fast(
        orthogroups=args.orthogroups,
        intensities_dir=args.intensities,
        tissue_ontology=args.tissue_ontology,
        metadata=args.metadata,
        output_raw=str(raw_output),
        output_maxlfq=str(maxlfq_output),
        og_desc_path=str(og_desc_path),
        og_annotation_table=args.og_annotation_table,
    )


if __name__ == "__main__":
    main()