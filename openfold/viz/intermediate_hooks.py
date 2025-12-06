"""
Utilities for recording intermediate representations from Evoformer layers
without modifying core OpenFold code.

You can import `IntermediateRecorder` and `attach_evoformer_hooks` in your own
scripts or notebooks.
"""

from typing import List, Dict, Any

import os
import numpy as np
import torch
from torch import nn


class IntermediateRecorder:
    """
    Records intermediate representations from Evoformer layers.

    For each layer (hook call) we store:
      - 'msa'   : [*, N_seq, N_res, C_m]
      - 'pair'  : [*, N_res, N_res, C_z]
      - 'single': [*, N_res, C_s] (if available)
    """

    def __init__(self) -> None:
        self.storage: List[Dict[str, torch.Tensor]] = []

    def hook_fn(self, module: nn.Module,
                inputs: Any,
                output: Any) -> None:
        """
        Forward hook attached to each Evoformer block.

        We expect output to be a tuple (m, z, s) OR (m, z).
        If 's' is not present per layer, we just skip it.
        """
        if isinstance(output, (tuple, list)):
            if len(output) == 3:
                m, z, s = output
            elif len(output) == 2:
                m, z = output
                s = None
            else:
                # Unexpected shape; don't record
                return
        else:
            # Unexpected output type
            return

        record: Dict[str, torch.Tensor] = {
            "msa": m.detach().cpu(),
            "pair": z.detach().cpu(),
        }
        if s is not None:
            record["single"] = s.detach().cpu()

        self.storage.append(record)

    def clear(self) -> None:
        self.storage = []

    def save(self, out_dir: str, protein_id: str) -> None:
        """
        Save all recorded tensors as .npy files.

        Files look like:
          {protein_id}_msa_layer{L}.npy
          {protein_id}_pair_layer{L}.npy
          {protein_id}_single_layer{L}.npy
        """
        os.makedirs(out_dir, exist_ok=True)

        for layer_idx, reps in enumerate(self.storage):
            for name, tensor in reps.items():
                path = os.path.join(
                    out_dir,
                    f"{protein_id}_{name}_layer{layer_idx}.npy"
                )
                np.save(path, tensor.numpy())
                print(f"[saved] {path}")


def attach_evoformer_hooks(model: nn.Module,
                           recorder: IntermediateRecorder) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Attach hooks to each Evoformer block in the model.

    DOES NOT modify the model code itself. You call this from your
    own script or notebook after creating the model.

    Returns:
        list of hook handles. You MUST call .remove() on each when done.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []

    evo = getattr(model, "evoformer", None)
    if evo is None:
        print("[warning] Model has no `.evoformer` attribute; no hooks attached.")
        return handles

    blocks = getattr(evo, "blocks", None)
    if blocks is None:
        print("[warning] evoformer has no `.blocks` attribute; no hooks attached.")
        return handles

    for idx, block in enumerate(blocks):
        h = block.register_forward_hook(recorder.hook_fn)
        handles.append(h)
        print(f"[hook] registered on evoformer.blocks[{idx}]")

    if not handles:
        print("[warning] No Evoformer blocks found; check model.evoformer.blocks")

    return handles

"""
Lightweight helpers for working with intermediate representations
(msa / pair / single) from OpenFold.

These utilities do NOT modify any core OpenFold modules. They are meant
to be called after a normal OpenFold inference run, using the `outputs`
dictionary returned by `AlphaFold.forward`.
"""

from typing import Dict, Any, Optional
import os

import torch

from visualize_intermediate_utils import (
    plot_single_norm,
    plot_pair_heatmap,
    plot_msa_norms,
    save_intermediate_summaries,
)


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


    save_intermediate_summaries(outputs, out_dir, protein_name=protein_name)


    msa = outputs.get("msa", None)
    pair = outputs.get("pair", None)
    single = outputs.get("single", None)

    if msa is not None and not isinstance(msa, torch.Tensor):
        msa = torch.as_tensor(msa)
    if pair is not None and not isinstance(pair, torch.Tensor):
        pair = torch.as_tensor(pair)
    if single is not None and not isinstance(single, torch.Tensor):
        single = torch.as_tensor(single)

   
    if single is not None:
        plot_single_norm(
            single,
            out_path=os.path.join(out_dir, f"{protein_name or 'sample'}_single_norm.png"),
            protein_name=protein_name,
            show=show,
        )

    if pair is not None:
        plot_pair_heatmap(
            pair,
            out_path=os.path.join(out_dir, f"{protein_name or 'sample'}_pair_norm_heatmap.png"),
            protein_name=protein_name,
            show=show,
        )


    if msa is not None:
        plot_msa_norms(
            msa,
            out_line_path=os.path.join(out_dir, f"{protein_name or 'sample'}_msa_per_res.png"),
            out_seq_hist_path=os.path.join(out_dir, f"{protein_name or 'sample'}_msa_seqwise_hist.png"),
            protein_name=protein_name,
            mode="first",
            show=show,
        )
