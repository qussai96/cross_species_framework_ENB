#!/usr/bin/env python3

import sys

def process_fasta(input_fasta, output_faa):
    seen = set()
    with open(input_fasta) as fin, open(output_faa, "w") as fout:
        header = None
        seq_lines = []

        def flush():
            if header and header not in seen:
                seen.add(header)
                fout.write(f">{header}\n")
                fout.write("".join(seq_lines))

        for line in fin:
            if line.startswith(">"):
                flush()
                header = line[1:].strip().split()[0]
                seq_lines = []
            else:
                seq_lines.append(line)

        flush()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} input.fasta output.faa")

    process_fasta(sys.argv[1], sys.argv[2])
