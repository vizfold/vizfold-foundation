"""
Utilities for summarizing and visualizing intermediate OpenFold
representations (msa / pair / single).

This file is self-contained and does NOT import `openfold`. It expects
to work with the `outputs` dict returned by `AlphaFold.forward`.
"""

from typing import Dict, Any, Optional, Tuple
import os

import numpy as np
import torch
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Helper: safe conversion to numpy
# ----------------------------------------------------------------------

def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ----------------------------------------------------------------------
# Numeric summaries (saved as .npy)
# ----------------------------------------------------------------------

def save_intermediate_summaries(
    outputs: Dict[str, Any],
    out_dir: str,
    protein_name: Optional[str] = None,
) -> None:
    """
    Save simple norm-based summaries of msa / pair / single reps as .npy.

    - msa_per_res_norm: [N_res] (averaged over seqs and channels)
    - single_norm:      [N_res]
    - pair_norm:        [N_res, N_res]
    """
    os.makedirs(out_dir, exist_ok=True)
    tag = protein_name or "sample"

    # ----- MSA -----
    if "msa" in outputs and outputs["msa"] is not None:
        msa = outputs["msa"]
        if not isinstance(msa, torch.Tensor):
            msa = torch.as_tensor(msa)

        # msa: [..., N_seq, N_res, C_m]
        # Compute per-residue norm, averaged over sequences
        # shape -> [..., N_seq, N_res, C_m]
        msa_norm = msa.norm(dim=-1)          # [..., N_seq, N_res]
        msa_per_res = msa_norm.mean(dim=-2)  # [..., N_res]

        msa_np = _to_numpy(msa_per_res.squeeze(0))
        path = os.path.join(out_dir, f"{tag}_msa_per_res_norm.npy")
        np.save(path, msa_np)
        print(f"[saved] {path}")

    # ----- SINGLE -----
    if "single" in outputs and outputs["single"] is not None:
        single = outputs["single"]
        if not isinstance(single, torch.Tensor):
            single = torch.as_tensor(single)

        # single: [..., N_res, C_s]
        single_norm = single.norm(dim=-1)  # [..., N_res]
        single_np = _to_numpy(single_norm.squeeze(0))
        path = os.path.join(out_dir, f"{tag}_single_norm.npy")
        np.save(path, single_np)
        print(f"[saved] {path}")

    # ----- PAIR -----
    if "pair" in outputs and outputs["pair"] is not None:
        pair = outputs["pair"]
        if not isinstance(pair, torch.Tensor):
            pair = torch.as_tensor(pair)

        # pair: [..., N_res, N_res, C_z]
        pair_norm = pair.norm(dim=-1)  # [..., N_res, N_res]
        pair_np = _to_numpy(pair_norm.squeeze(0))
        path = os.path.join(out_dir, f"{tag}_pair_norm.npy")
        np.save(path, pair_np)
        print(f"[saved] {path}")


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------

def plot_single_norm(
    single: torch.Tensor,
    out_path: str,
    protein_name: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Line plot of ||single[i]|| across residues.
    """
    if not isinstance(single, torch.Tensor):
        single = torch.as_tensor(single)

    single_norm = single.norm(dim=-1)  # [..., N_res]
    single_norm = single_norm.squeeze(0).cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.plot(single_norm)
    plt.xlabel("Residue index")
    plt.ylabel("||single|| (L2 norm)")
    title = f"Single representation norm per residue"
    if protein_name:
        title += f" — {protein_name}"
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    print(f"[saved] {out_path}")


def plot_pair_heatmap(
    pair: torch.Tensor,
    out_path: str,
    protein_name: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Heatmap of ||pair[i,j]|| across residue pairs (i,j).
    """
    if not isinstance(pair, torch.Tensor):
        pair = torch.as_tensor(pair)

    pair_norm = pair.norm(dim=-1).squeeze(0).cpu().numpy()  # [N_res, N_res]

    plt.figure(figsize=(6, 5))
    im = plt.imshow(pair_norm, cmap="viridis", origin="lower")
    plt.colorbar(im, label="||pair|| (L2)")
    plt.xlabel("Residue j")
    plt.ylabel("Residue i")
    title = "Pair representation norm heatmap"
    if protein_name:
        title += f" — {protein_name}"
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    print(f"[saved] {out_path}")


def plot_msa_norms(
    msa: torch.Tensor,
    out_line_path: str,
    out_seq_hist_path: str,
    protein_name: Optional[str] = None,
    mode: str = "first",
    show: bool = False,
) -> None:
    """
    Two plots:

    1) Per-residue norm (average over sequences) as a line plot
    2) Histogram of per-sequence average norms

    mode:
      - "first": use only the first example in batch
    """
    if not isinstance(msa, torch.Tensor):
        msa = torch.as_tensor(msa)

    # msa: [..., N_seq, N_res, C_m]
    msa = msa.squeeze(0)  # [N_seq, N_res, C_m]
    msa_norm = msa.norm(dim=-1)            # [N_seq, N_res]
    per_res = msa_norm.mean(dim=0).cpu().numpy()  # [N_res]
    per_seq = msa_norm.mean(dim=1).cpu().numpy()  # [N_seq]

    # Line plot per residue
    plt.figure(figsize=(8, 4))
    plt.plot(per_res)
    plt.xlabel("Residue index")
    plt.ylabel("Mean ||msa|| over sequences")
    title = "MSA representation per-residue norm"
    if protein_name:
        title += f" — {protein_name}"
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_line_path), exist_ok=True)
    plt.savefig(out_line_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    print(f"[saved] {out_line_path}")

    # Histogram over sequences
    plt.figure(figsize=(6, 4))
    plt.hist(per_seq, bins=30)
    plt.xlabel("Mean ||msa|| over residues")
    plt.ylabel("Count (sequences)")
    title = "MSA sequence-wise mean norm"
    if protein_name:
        title += f" — {protein_name}"
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_seq_hist_path), exist_ok=True)
    plt.savefig(out_seq_hist_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    print(f"[saved] {out_seq_hist_path}")


# ----------------------------------------------------------------------
# High-level helper used by your script
# ----------------------------------------------------------------------

def visualize_from_outputs(
    outputs: Dict[str, Any],
    out_dir: str,
    protein_name: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    High-level helper: given an OpenFold `outputs` dict, generate
    summary numpy arrays and plots for msa / pair / single
    representations.

    Args:
        outputs: dict produced by AlphaFold.forward
        out_dir: directory where plots and .npy files will be written
        protein_name: optional label (used in filenames / plot titles)
        show: if True, show plots interactively in addition to saving
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save numeric summaries
    save_intermediate_summaries(outputs, out_dir, protein_name=protein_name)

    # Convert to tensors if needed
    msa = outputs.get("msa", None)
    pair = outputs.get("pair", None)
    single = outputs.get("single", None)

    if msa is not None and not isinstance(msa, torch.Tensor):
        msa = torch.as_tensor(msa)
    if pair is not None and not isinstance(pair, torch.Tensor):
        pair = torch.as_tensor(pair)
    if single is not None and not isinstance(single, torch.Tensor):
        single = torch.as_tensor(single)

    tag = protein_name or "sample"

    # Single rep line plot
    if single is not None:
        plot_single_norm(
            single,
            out_path=os.path.join(out_dir, f"{tag}_single_norm.png"),
            protein_name=protein_name,
            show=show,
        )

    # Pair rep heatmap
    if pair is not None:
        plot_pair_heatmap(
            pair,
            out_path=os.path.join(out_dir, f"{tag}_pair_norm_heatmap.png"),
            protein_name=protein_name,
            show=show,
        )

    # MSA plots
    if msa is not None:
        plot_msa_norms(
            msa,
            out_line_path=os.path.join(out_dir, f"{tag}_msa_per_res.png"),
            out_seq_hist_path=os.path.join(out_dir, f"{tag}_msa_seqwise_hist.png"),
            protein_name=protein_name,
            mode="first",
            show=show,
        )
