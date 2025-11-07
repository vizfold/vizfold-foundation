import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_all_heads(connections_file, top_k=None):
    """Parse consolidated attention dumps without importing PyMOL dependencies."""
    heads = {}
    current_head = None

    with open(connections_file, 'r') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith('layer'):
                parts = line.replace(',', '').split()
                head_idx = int(parts[-1])
                current_head = head_idx
                heads[current_head] = []
            else:
                res1, res2, weight = map(float, line.split())
                heads[current_head].append((int(res1), int(res2), weight))

    for head_idx, connections in heads.items():
        connections.sort(key=lambda x: x[2], reverse=True)
        if top_k is not None:
            heads[head_idx] = connections[:top_k]

    return heads


def _load_heads(attention_dir, attention_type, layer_idx, residue_index=None, top_k=None):
    if attention_type == "msa_row":
        attention_path = Path(attention_dir) / f"msa_row_attn_layer{layer_idx}.txt"
    else:
        if residue_index is None:
            raise ValueError("residue_index is required for triangle attention")
        attention_path = Path(attention_dir) / f"triangle_start_attn_layer{layer_idx}_residue_idx_{residue_index}.txt"

    if not attention_path.exists():
        raise FileNotFoundError(attention_path)

    return load_all_heads(str(attention_path), top_k=top_k)


def _compute_matrix(heads):
    if not heads:
        raise ValueError("No heads found in attention file")

    max_residue = 0
    for connections in heads.values():
        for res1, res2, _ in connections:
            max_residue = max(max_residue, res1, res2)

    residue_dim = max_residue + 1
    head_ids = sorted(heads.keys())
    matrix = np.zeros((len(head_ids), residue_dim), dtype=np.float32)

    for row, head_idx in enumerate(head_ids):
        for _, res2, weight in heads[head_idx]:
            matrix[row, res2] += weight

    return head_ids, matrix


def _plot_heatmap(head_ids, matrix, protein, layer_idx, output_dir, suffix=""):
    plt.figure(figsize=(max(8, matrix.shape[1] / 20), max(4, matrix.shape[0] / 2)))
    plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(label="Aggregated attention weight")

    plt.yticks(np.arange(len(head_ids)), [f"Head {h}" for h in head_ids])

    step = max(1, matrix.shape[1] // 20)
    xticks = np.arange(0, matrix.shape[1], step)
    plt.xticks(xticks, xticks)
    plt.xlabel("Residue index")
    plt.ylabel("Attention head")
    plt.title(f"{protein} Layer {layer_idx} Attention Heatmap{suffix}")

    os.makedirs(output_dir, exist_ok=True)
    fname = f"attention_heatmap_layer_{layer_idx}_{protein}{suffix}.png"
    out_path = Path(output_dir) / fname
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[Saved] {out_path}")


def generate_heatmap(attention_dir, output_dir, protein, attention_type, layer_idx, top_k=None, residue_index=None):
    heads = _load_heads(attention_dir, attention_type, layer_idx, residue_index=residue_index, top_k=top_k)
    head_ids, matrix = _compute_matrix(heads)

    suffix = ""
    if attention_type == "triangle_start" and residue_index is not None:
        suffix = f"_residue_{residue_index}"

    generate_dir = Path(output_dir) / attention_type
    _plot_heatmap(head_ids, matrix, protein, layer_idx, generate_dir, suffix=suffix)


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate attention heatmaps as an alternative visualization.")
    parser.add_argument("--attention-dir", type=Path, required=True, help="Directory containing attention text files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save heatmap plots.")
    parser.add_argument("--protein", type=str, required=True, help="Protein identifier for filenames.")
    parser.add_argument("--attention-type", choices=["msa_row", "triangle_start"], default="msa_row", help="Attention source to visualize.")
    parser.add_argument("--layer-idx", type=int, default=47, help="Layer index to visualize.")
    parser.add_argument("--top-k", type=int, default=None, help="Limit number of edges per head.")
    parser.add_argument("--residue-index", type=int, default=None, help="Residue index for triangle attention.")
    return parser.parse_args()


def _main():
    args = _parse_args()
    generate_heatmap(
        attention_dir=str(args.attention_dir),
        output_dir=str(args.output_dir),
        protein=args.protein,
        attention_type=args.attention_type,
        layer_idx=args.layer_idx,
        top_k=args.top_k,
        residue_index=args.residue_index,
    )


if __name__ == "__main__":
    _main()

