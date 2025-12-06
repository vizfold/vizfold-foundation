#!/usr/bin/env python

"""
Small helper script to build a single multi-panel figure from the
saved numpy summaries (single_norm, pair_norm, msa_per_res).

You can use this to generate a nice example figure for your README.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Create a demo figure from saved intermediate summaries."
    )
    p.add_argument(
        "--summary-dir",
        type=str,
        required=True,
        help="Directory containing *_single_norm.npy, *_pair_norm.npy, *_msa_per_res.npy.",
    )
    p.add_argument(
        "--basename",
        type=str,
        default="sample",
        help="Base name used in the npy files (e.g., '6KWC').",
    )
    p.add_argument(
        "--out-path",
        type=str,
        default="outputs/intermediate_reps/example_panel.png",
        help="Where to save the combined panel figure.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    single_path = os.path.join(args.summary_dir, f"{args.basename}_single_norm.npy")
    pair_path = os.path.join(args.summary_dir, f"{args.basename}_pair_norm.npy")
    msa_res_path = os.path.join(args.summary_dir, f"{args.basename}_msa_per_res.npy")

    single = np.load(single_path)
    pair = np.load(pair_path)
    msa_res = np.load(msa_res_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Single
    x = np.arange(1, len(single) + 1)
    axes[0].plot(x, single, linewidth=1.2)
    axes[0].set_title("Single Rep Norm per Residue")
    axes[0].set_xlabel("Residue index")
    axes[0].set_ylabel("L2 norm")

    # Pair
    im = axes[1].imshow(pair, origin="lower", aspect="auto")
    axes[1].set_title("Pair Rep Frobenius Norm")
    axes[1].set_xlabel("Residue index")
    axes[1].set_ylabel("Residue index")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # MSA per-residue
    x2 = np.arange(1, len(msa_res) + 1)
    axes[2].plot(x2, msa_res, linewidth=1.2)
    axes[2].set_title("MSA Rep Norm per Residue (first seq)")
    axes[2].set_xlabel("Residue index")
    axes[2].set_ylabel("L2 norm")

    fig.suptitle(f"Intermediate Representations — {args.basename}", fontsize=14)
    fig.tight_layout()
    plt.savefig(args.out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] example panel → {args.out_path}")


if __name__ == "__main__":
    main()
