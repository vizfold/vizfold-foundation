#!/usr/bin/env python

"""
Run intermediate representation visualization on a saved OpenFold outputs file.

Usage example:

    python scripts/run_with_intermediate_reps.py \
        --outputs-path outputs/my_protein_outputs.pt \
        --outdir outputs/intermediate_reps/my_protein \
        --protein-name 6KWC

This script does NOT run OpenFold itself. It assumes you have already
run inference (e.g., via existing scripts / notebooks) and saved the
`outputs` dict using:

    torch.save(outputs, "outputs/my_protein_outputs.pt")
"""

import argparse
import os
import torch

from visualize_intermediate_utils import visualize_from_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize msa / pair / single reps from a saved OpenFold outputs dict."
    )
    parser.add_argument(
        "--outputs-path",
        type=str,
        required=True,
        help="Path to a torch-saved outputs dict (e.g. .pt or .pth).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory where plots and summary .npy files will be written.",
    )
    parser.add_argument(
        "--protein-name",
        type=str,
        default=None,
        help="Optional protein name/ID for labeling plots and filenames.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If set, show plots interactively in addition to saving PNGs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[Info] Loading outputs from: {args.outputs_path}")
    outputs = torch.load(args.outputs_path, map_location="cpu")

    if not isinstance(outputs, dict):
        raise ValueError(
            f"Expected a dict-like object from {args.outputs_path}, "
            f"but got type {type(outputs)}"
        )

    visualize_from_outputs(
        outputs,
        out_dir=args.outdir,
        protein_name=args.protein_name,
        show=args.show,
    )


if __name__ == "__main__":
    main()
