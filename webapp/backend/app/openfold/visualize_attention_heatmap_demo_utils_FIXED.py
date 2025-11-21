import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_all_heads(connections_file, top_k=None):
    """
    Parse consolidated attention dumps.

    File format per line:
        "Layer X, Head Y"  -> starts new head section
        "res1 res2 weight" -> attention edge (query->key direction)

    Args:
        connections_file: Path to attention text file
        top_k: If specified, keep only top_k edges per head (by weight)

    Returns:
        dict mapping head_idx -> List[(query_idx, key_idx, weight)]
            where query_idx (res1) = source residue (0-based)
                  key_idx (res2) = target residue (0-based)
                  weight = attention score from query to key
    """
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
    """Load attention data from text files."""
    if attention_type == "msa_row":
        attention_path = Path(attention_dir) / f"msa_row_attn_layer{layer_idx}.txt"
    else:
        if residue_index is None:
            raise ValueError("residue_index is required for triangle attention")
        attention_path = Path(attention_dir) / f"triangle_start_attn_layer{layer_idx}_residue_idx_{residue_index}.txt"

    if not attention_path.exists():
        raise FileNotFoundError(f"Attention file not found: {attention_path}")

    return load_all_heads(str(attention_path), top_k=top_k)


def _compute_matrix_for_head(connections, sequence_length):
    """
    Create L×L residue-residue attention matrix for ONE head.

    Args:
        connections: List of (query_idx, key_idx, weight) tuples
        sequence_length: Number of residues in the sequence

    Returns:
        np.ndarray: Shape (sequence_length, sequence_length) where
                    matrix[i,j] = attention from residue i (query) to residue j (key)
    """
    matrix = np.zeros((sequence_length, sequence_length), dtype=np.float32)

    for res1, res2, weight in connections:
        if res1 < sequence_length and res2 < sequence_length:
            # res1 = query (source row), res2 = key (target column)
            matrix[res1, res2] = weight

    return matrix


def _infer_sequence_length(connections):
    """Infer sequence length from maximum residue index in connections."""
    max_idx = 0
    for res1, res2, _ in connections:
        max_idx = max(max_idx, res1, res2)
    return max_idx + 1


def _plot_heatmap_single_head(matrix, protein, layer_idx, head_idx, output_dir,
                               attention_type, residue_index=None):
    """
    Plot L×L heatmap for a single attention head.

    Args:
        matrix: np.ndarray of shape (L, L)
        protein: Protein identifier string
        layer_idx: Layer number
        head_idx: Head number
        output_dir: Output directory path
        attention_type: 'msa_row' or 'triangle_start'
        residue_index: Optional residue index for triangle attention
    """
    seq_len = matrix.shape[0]

    fig, ax = plt.subplots(figsize=(10, 9))

    # Plot heatmap
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', origin='upper')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Attention Weight')
    cbar.ax.tick_params(labelsize=10)

    # Axis labels and ticks
    tick_step = max(1, seq_len // 20)
    ticks = np.arange(0, seq_len, tick_step)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, fontsize=9)
    ax.set_yticklabels(ticks, fontsize=9)

    ax.set_xlabel('Target Residue (Key)', fontsize=12)
    ax.set_ylabel('Source Residue (Query)', fontsize=12)

    # Title
    title_parts = [f"{protein}", f"Layer {layer_idx}", f"Head {head_idx}"]
    if attention_type == "triangle_start" and residue_index is not None:
        title_parts.append(f"Residue {residue_index}")

    ax.set_title(f"Attention Heatmap: {' | '.join(title_parts)}",
                 fontsize=14, pad=15)

    # Save
    os.makedirs(output_dir, exist_ok=True)

    suffix = f"_residue_{residue_index}" if residue_index is not None else ""
    fname = f"{attention_type}_heatmap_layer_{layer_idx}_head_{head_idx}_{protein}{suffix}.png"
    out_path = Path(output_dir) / fname

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[Saved] {out_path}")
    return out_path


def generate_heatmap_for_head(attention_dir, output_dir, protein, attention_type,
                               layer_idx, head_idx, top_k=None, residue_index=None):
    """
    Generate L×L attention heatmap for a SPECIFIC head.

    This is the CORRECTED version that creates proper residue-residue matrices.

    Args:
        attention_dir: Directory containing attention text files
        output_dir: Where to save output PNG
        protein: Protein identifier
        attention_type: 'msa_row' or 'triangle_start'
        layer_idx: Layer number
        head_idx: Head number (REQUIRED - we visualize ONE head at a time)
        top_k: Optional limit on number of edges to include
        residue_index: Required for triangle_start attention

    Returns:
        Path to saved heatmap PNG
    """
    # Load attention data for all heads
    heads = _load_heads(attention_dir, attention_type, layer_idx,
                       residue_index=residue_index, top_k=top_k)

    if head_idx not in heads:
        raise ValueError(f"Head {head_idx} not found in attention data. "
                        f"Available heads: {sorted(heads.keys())}")

    # Get connections for the specified head
    connections = heads[head_idx]

    if not connections:
        raise ValueError(f"No attention connections found for head {head_idx}")

    # Infer sequence length
    sequence_length = _infer_sequence_length(connections)

    # Create L×L matrix for this head
    matrix = _compute_matrix_for_head(connections, sequence_length)

    # Verify matrix shape
    assert matrix.shape == (sequence_length, sequence_length), \
        f"Matrix shape mismatch: expected ({sequence_length}, {sequence_length}), got {matrix.shape}"

    # Plot and save
    output_path = _plot_heatmap_single_head(
        matrix, protein, layer_idx, head_idx, output_dir,
        attention_type, residue_index
    )

    return output_path


# Legacy function for backward compatibility (DEPRECATED)
def _compute_matrix(heads):
    """
    DEPRECATED: This creates [heads × residues] matrix, not [residues × residues]!

    Use _compute_matrix_for_head() and generate_heatmap_for_head() instead.
    """
    import warnings
    warnings.warn("_compute_matrix() is deprecated. Use _compute_matrix_for_head() instead.",
                  DeprecationWarning)

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
    """DEPRECATED: Legacy plotting function for multi-head aggregate view."""
    import warnings
    warnings.warn("_plot_heatmap() is deprecated. Use _plot_heatmap_single_head() instead.",
                  DeprecationWarning)

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


def generate_heatmap(attention_dir, output_dir, protein, attention_type, layer_idx,
                     head_idx=None, top_k=None, residue_index=None):
    """
    Generate attention heatmap.

    If head_idx is specified, generates L×L heatmap for that specific head (RECOMMENDED).
    If head_idx is None, generates legacy multi-head aggregate view (DEPRECATED).

    Args:
        attention_dir: Directory containing attention text files
        output_dir: Where to save output PNG
        protein: Protein identifier
        attention_type: 'msa_row' or 'triangle_start'
        layer_idx: Layer number
        head_idx: Head number (if None, uses legacy multi-head view)
        top_k: Optional limit on edges
        residue_index: Required for triangle_start attention

    Returns:
        Path to saved heatmap PNG (or None for legacy mode)
    """
    if head_idx is not None:
        # New behavior: generate L×L heatmap for specific head
        return generate_heatmap_for_head(
            attention_dir, output_dir, protein, attention_type,
            layer_idx, head_idx, top_k=top_k, residue_index=residue_index
        )
    else:
        # Legacy behavior: multi-head aggregate view (DEPRECATED)
        import warnings
        warnings.warn(
            "Generating multi-head aggregate heatmap (deprecated). "
            "Specify head_idx to generate proper L×L heatmap for a single head.",
            DeprecationWarning
        )

        heads = _load_heads(attention_dir, attention_type, layer_idx,
                           residue_index=residue_index, top_k=top_k)
        head_ids, matrix = _compute_matrix(heads)

        suffix = ""
        if attention_type == "triangle_start" and residue_index is not None:
            suffix = f"_residue_{residue_index}"

        generate_dir = Path(output_dir) / attention_type
        _plot_heatmap(head_ids, matrix, protein, layer_idx, generate_dir, suffix=suffix)
        return None


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate attention heatmaps. "
                    "Now creates L×L matrices for specific heads (not multi-head aggregates)."
    )
    parser.add_argument("--attention-dir", type=Path, required=True,
                       help="Directory containing attention text files.")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Directory to save heatmap plots.")
    parser.add_argument("--protein", type=str, required=True,
                       help="Protein identifier for filenames.")
    parser.add_argument("--attention-type", choices=["msa_row", "triangle_start"],
                       default="msa_row", help="Attention source to visualize.")
    parser.add_argument("--layer-idx", type=int, default=47,
                       help="Layer index to visualize.")
    parser.add_argument("--head-idx", type=int, default=None,
                       help="Head index to visualize (REQUIRED for new behavior).")
    parser.add_argument("--top-k", type=int, default=None,
                       help="Limit number of edges per head.")
    parser.add_argument("--residue-index", type=int, default=None,
                       help="Residue index for triangle attention.")
    return parser.parse_args()


def _main():
    args = _parse_args()

    if args.head_idx is None:
        print("WARNING: No head_idx specified. Using legacy multi-head aggregate mode.")
        print("Recommend: specify --head-idx for proper L×L residue-residue heatmap.")

    generate_heatmap(
        attention_dir=str(args.attention_dir),
        output_dir=str(args.output_dir),
        protein=args.protein,
        attention_type=args.attention_type,
        layer_idx=args.layer_idx,
        head_idx=args.head_idx,
        top_k=args.top_k,
        residue_index=args.residue_index,
    )


if __name__ == "__main__":
    _main()
