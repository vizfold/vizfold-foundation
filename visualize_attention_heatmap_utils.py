import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from visualize_attention_arc_diagram_demo_utils import load_all_heads


# ========== Matrix Reconstruction ==========
def reconstruct_attention_matrix(connections, seq_length):
    """
    Reconstruct a full attention matrix from sparse top-K connections.
    """
    matrix = np.zeros((seq_length, seq_length))
    
    for res1, res2, weight in connections:
        if 0 <= res1 < seq_length and 0 <= res2 < seq_length:
            matrix[res1, res2] = weight
    
    return matrix


def get_sequence_length_from_fasta(fasta_path):
    """
    Get sequence length from FASTA file.
    """
    with open(fasta_path, 'r') as f:
        lines = f.readlines()
    
    seq_lines = [line.strip() for line in lines if not line.startswith('>')]
    sequence = ''.join(seq_lines)
    return len(sequence)


# ========== Heatmap Plotting ==========
def plot_all_heads_heatmap(
    attention_dir,
    output_dir,
    protein,
    attention_type="msa_row",
    layer_idx=47,
    seq_length=None,
    fasta_path=None,
    normalization="global",
    colormap="viridis",
    figsize_per_head=(2.0, 2.0),
    dpi=300,
    save_to_png=True,
    residue_indices=None
):
    """
    Generate heatmap grid showing all attention heads for a layer.
    
    Args:
        attention_dir: Directory containing attention text files
        output_dir: Directory to save output PNG
        protein: Protein identifier (e.g., "6KWC")
        attention_type: "msa_row" or "triangle_start"
        layer_idx: Layer number to visualize
        seq_length: Sequence length (auto-detect if None)
        fasta_path: Path to FASTA file for sequence length detection
        normalization: "global" or "per_head" normalization
        colormap: Matplotlib colormap name
        figsize_per_head: Size of each subplot (width, height)
        dpi: Output resolution
        save_to_png: Whether to save to PNG file
        residue_indices: List of residue indices for triangle_start (required for triangle_start)
    """
    
    if attention_type not in ["msa_row", "triangle_start"]:
        raise ValueError(f"Invalid attention_type: {attention_type}")
    
    if normalization not in ["global", "per_head"]:
        raise ValueError(f"Invalid normalization: {normalization}")
    
    if attention_type == "triangle_start":
        assert residue_indices is not None, "residue_indices required for triangle_start attention"
    
    if seq_length is None:
        if fasta_path and os.path.exists(fasta_path):
            seq_length = get_sequence_length_from_fasta(fasta_path)
        else:
            seq_length = 191
            print(f"[Warning] Using default sequence length: {seq_length}")
    
    if attention_type == "msa_row":
        file_path = os.path.join(attention_dir, f"msa_row_attn_layer{layer_idx}.txt")
        heads = load_all_heads(file_path, top_k=None)
        if not heads:
            print(f"[Warning] No attention data found in {file_path}")
            
    else:  # triangle_start
        heads = {}
        for res_idx in residue_indices:
            file_path = os.path.join(attention_dir, f"triangle_start_attn_layer{layer_idx}_residue_idx_{res_idx}.txt")
            if not os.path.exists(file_path):
                print(f"[Warning] Missing file for residue {res_idx}")
                continue
            
            res_heads = load_all_heads(file_path, top_k=None)
            if not res_heads:
                print(f"[Warning] No attention data found in {file_path}")
                continue
            
            heads.update(res_heads)
            break
        
        if not heads:
            print(f"[Warning] No valid attention data found for any residue in {residue_indices}")
    
    attention_matrices = {}
    for head_idx, connections in heads.items():
        if not connections:
            continue
        matrix = reconstruct_attention_matrix(connections, seq_length)
        attention_matrices[head_idx] = matrix
    
    num_heads = len(attention_matrices)
    
    if num_heads == 0:
        print(f"[Warning] No valid attention heads to visualize")
        return None
    
    if num_heads <= 4:
        rows, cols = 1, num_heads
    elif num_heads <= 8:
        rows, cols = 2, 4
    elif num_heads <= 12:
        rows, cols = 3, 4
    elif num_heads <= 16:
        rows, cols = 4, 4
    else:
        cols = min(4, int(np.ceil(np.sqrt(num_heads))))
        rows = (num_heads + cols - 1) // cols
    
    fig_width = cols * figsize_per_head[0]
    fig_height = rows * figsize_per_head[1]
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    if num_heads == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    all_values = []
    for matrix in attention_matrices.values():
        all_values.extend(matrix.flatten())
    
    all_values = np.array(all_values)
    global_min = np.min(all_values)
    global_max = np.max(all_values)
    
    for i, (head_idx, matrix) in enumerate(sorted(attention_matrices.items())):
        ax = axes[i]
        
        if normalization == "global":
            if global_max > global_min:
                normalized_matrix = (matrix - global_min) / (global_max - global_min)
            else:
                normalized_matrix = matrix
        else:  # per_head
            head_min, head_max = np.min(matrix), np.max(matrix)
            if head_max > head_min:
                normalized_matrix = (matrix - head_min) / (head_max - head_min)
            else:
                normalized_matrix = matrix
        
        im = ax.imshow(normalized_matrix, cmap=colormap, aspect='auto', interpolation='nearest')
        ax.set_title(f'Head {head_idx}', fontsize=10, weight='bold')
        ax.set_xlabel('Residue Position', fontsize=8)
        ax.set_ylabel('Residue Position', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)
    
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if rows == 1:
        plt.subplots_adjust(top=0.75)
        title_y = 0.95
    elif rows <= 2:
        plt.subplots_adjust(top=0.85)
        title_y = 0.98
    else:
        plt.subplots_adjust(top=0.90)
        title_y = 0.99
    
    title = f"{protein} {attention_type.replace('_', ' ').title()} Attention - Layer {layer_idx}"
    fig.suptitle(title, fontsize=14, weight='bold', y=title_y)
    
    if save_to_png:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{attention_type}_heatmap_layer_{layer_idx}_{protein}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[Saved] Heatmap visualization to {output_path}")
    else:
        output_path = None
    
    plt.close()
    
    return output_path


def plot_combined_attention_heatmap(
    attention_dir,
    output_dir,
    protein,
    layer_idx=47,
    seq_length=None,
    fasta_path=None,
    normalization="global",
    colormap="viridis",
    figsize_per_head=(1.5, 1.5),
    dpi=300,
    save_to_png=True,
    residue_indices=None
):
    """
    Generate combined heatmap showing both MSA Row and Triangle Start attention.
    
    Args:
        attention_dir: Directory containing attention text files
        output_dir: Directory to save output PNG
        protein: Protein identifier (e.g., "6KWC")
        layer_idx: Layer number to visualize
        seq_length: Sequence length (auto-detect if None)
        fasta_path: Path to FASTA file for sequence length detection
        normalization: "global" or "per_head" normalization
        colormap: Matplotlib colormap name
        figsize_per_head: Size of each subplot (width, height)
        dpi: Output resolution
        save_to_png: Whether to save to PNG file
        residue_indices: List of residue indices for triangle_start (defaults to [18])
    """
    
    if seq_length is None:
        if fasta_path and os.path.exists(fasta_path):
            seq_length = get_sequence_length_from_fasta(fasta_path)
        else:
            seq_length = 191
            print(f"[Warning] Using default sequence length: {seq_length}")
    
    msa_file = os.path.join(attention_dir, f"msa_row_attn_layer{layer_idx}.txt")
    msa_heads = load_all_heads(msa_file, top_k=None)
    if not msa_heads:
        print(f"[Warning] No MSA Row attention data found in {msa_file}")
    
    if residue_indices is None:
        residue_indices = [18]
    
    tri_heads = {}
    for res_idx in residue_indices:
        tri_file = os.path.join(attention_dir, f"triangle_start_attn_layer{layer_idx}_residue_idx_{res_idx}.txt")
        if not os.path.exists(tri_file):
            print(f"[Warning] Missing triangle_start file for residue {res_idx}")
            continue
        
        res_tri_heads = load_all_heads(tri_file, top_k=None)
        if not res_tri_heads:
            print(f"[Warning] No triangle_start attention data found in {tri_file}")
            continue
        
        tri_heads.update(res_tri_heads)
        break
    
    if not tri_heads:
        print(f"[Warning] No valid triangle_start attention data found for any residue in {residue_indices}")
    
    msa_matrices = {}
    tri_matrices = {}
    
    for head_idx, connections in msa_heads.items():
        if not connections:
            continue
        matrix = reconstruct_attention_matrix(connections, seq_length)
        msa_matrices[head_idx] = matrix
    
    for head_idx, connections in tri_heads.items():
        if not connections:
            continue
        matrix = reconstruct_attention_matrix(connections, seq_length)
        tri_matrices[head_idx] = matrix
    
    msa_heads_count = len(msa_matrices)
    tri_heads_count = len(tri_matrices)
    total_heads = msa_heads_count + tri_heads_count
    
    if total_heads == 0:
        print(f"[Warning] No valid attention heads to visualize")
        return None
    
    if total_heads <= 4:
        rows, cols = 1, total_heads
    elif total_heads <= 8:
        rows, cols = 2, 4
    elif total_heads <= 12:
        rows, cols = 3, 4
    elif total_heads <= 16:
        rows, cols = 4, 4
    else:
        cols = min(4, int(np.ceil(np.sqrt(total_heads))))
        rows = (total_heads + cols - 1) // cols
    fig_width = cols * figsize_per_head[0]
    fig_height = rows * figsize_per_head[1]
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()
    
    all_values = []
    for matrix in msa_matrices.values():
        all_values.extend(matrix.flatten())
    for matrix in tri_matrices.values():
        all_values.extend(matrix.flatten())
    
    all_values = np.array(all_values)
    global_min = np.min(all_values)
    global_max = np.max(all_values)
    
    plot_idx = 0
    for head_idx, matrix in sorted(msa_matrices.items()):
        ax = axes[plot_idx]
        
        if normalization == "global":
            if global_max > global_min:
                normalized_matrix = (matrix - global_min) / (global_max - global_min)
            else:
                normalized_matrix = matrix
        else:  # per_head
            head_min, head_max = np.min(matrix), np.max(matrix)
            if head_max > head_min:
                normalized_matrix = (matrix - head_min) / (head_max - head_min)
            else:
                normalized_matrix = matrix
        
        im = ax.imshow(normalized_matrix, cmap=colormap, aspect='auto', interpolation='nearest')
        ax.set_title(f'MSA Head {head_idx}', fontsize=10, weight='bold')
        ax.set_xlabel('Residue', fontsize=8)
        ax.set_ylabel('Residue', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)
        
        plot_idx += 1
    
    for head_idx, matrix in sorted(tri_matrices.items()):
        ax = axes[plot_idx]
        
        if normalization == "global":
            if global_max > global_min:
                normalized_matrix = (matrix - global_min) / (global_max - global_min)
            else:
                normalized_matrix = matrix
        else:  # per_head
            head_min, head_max = np.min(matrix), np.max(matrix)
            if head_max > head_min:
                normalized_matrix = (matrix - head_min) / (head_max - head_min)
            else:
                normalized_matrix = matrix
        
        im = ax.imshow(normalized_matrix, cmap=colormap, aspect='auto', interpolation='nearest')
        ax.set_title(f'Tri Head {head_idx}', fontsize=10, weight='bold')
        ax.set_xlabel('Residue', fontsize=8)
        ax.set_ylabel('Residue', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)
        
        plot_idx += 1
    
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if rows == 1:
        plt.subplots_adjust(top=0.75)
        title_y = 0.95
    elif rows <= 2:
        plt.subplots_adjust(top=0.85)
        title_y = 0.98
    else:
        plt.subplots_adjust(top=0.90)
        title_y = 0.99
    
    title = f"{protein} Combined Attention Heatmaps - Layer {layer_idx}"
    fig.suptitle(title, fontsize=14, weight='bold', y=title_y)
    
    if save_to_png:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"combined_attention_heatmap_layer_{layer_idx}_{protein}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[Saved] Combined heatmap visualization to {output_path}")
    else:
        output_path = None
    
    plt.close()
    
    return output_path


if __name__ == "__main__":
    # Example usage
    attention_dir = "./outputs/attention_files_6KWC_demo_tri_18"
    output_dir = "./outputs/heatmap_visualizations"
    protein = "6KWC"
    layer_idx = 47
    fasta_path = "./examples/monomer/fasta_dir_6KWC/6KWC.fasta"
    
    # Generate MSA Row heatmap
    print("Generating MSA Row attention heatmap...")
    plot_all_heads_heatmap(
        attention_dir=attention_dir,
        output_dir=output_dir,
        protein=protein,
        attention_type="msa_row",
        layer_idx=layer_idx,
        fasta_path=fasta_path,
        normalization="global",
        colormap="viridis"
    )
    
    # Generate Triangle Start heatmap
    print("Generating Triangle Start attention heatmap...")
    plot_all_heads_heatmap(
        attention_dir=attention_dir,
        output_dir=output_dir,
        protein=protein,
        attention_type="triangle_start",
        layer_idx=layer_idx,
        fasta_path=fasta_path,
        normalization="global",
        colormap="viridis"
    )
    
    # Generate combined heatmap
    print("Generating combined attention heatmap...")
    plot_combined_attention_heatmap(
        attention_dir=attention_dir,
        output_dir=output_dir,
        protein=protein,
        layer_idx=layer_idx,
        fasta_path=fasta_path,
        normalization="global",
        colormap="viridis"
    )
    
    print("Heatmap generation complete!")
