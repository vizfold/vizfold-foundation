"""
Intermediate representation visualization for OpenFold.

Extract and visualize MSA, Pair, and Structure representations from OpenFold's 
48-layer network. Supports layer-by-layer analysis and various visualization modes.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os


class IntermediateRepresentations:
    def __init__(self):
        self.msa_reps = {}
        self.pair_reps = {}
        self.single_reps = {}
        self.enabled = False
    
    def clear(self):
        self.msa_reps = {}
        self.pair_reps = {}
        self.single_reps = {}
    
    def enable(self):
        self.enabled = True
        self.clear()
    
    def disable(self):
        self.enabled = False
        self.clear()


INTERMEDIATE_REPS = IntermediateRepresentations()


def extract_msa_representations(model_output, layer_indices=None):
    """Extract MSA representations from model output or stored layers."""
    if model_output is not None and "msa" in model_output:
        return {-1: model_output["msa"]}
    
    if not INTERMEDIATE_REPS.msa_reps:
        raise ValueError("No MSA representations found. Enable hooks first.")
    
    if layer_indices is None:
        return INTERMEDIATE_REPS.msa_reps.copy()
    
    return {idx: INTERMEDIATE_REPS.msa_reps[idx] 
            for idx in layer_indices 
            if idx in INTERMEDIATE_REPS.msa_reps}


def extract_pair_representations(model_output, layer_indices=None):
    """Extract Pair representations from model output or stored layers."""
    if model_output is not None and "pair" in model_output:
        return {-1: model_output["pair"]}
    
    if not INTERMEDIATE_REPS.pair_reps:
        raise ValueError("No Pair representations found. Enable hooks first.")
    
    if layer_indices is None:
        return INTERMEDIATE_REPS.pair_reps.copy()
    
    return {idx: INTERMEDIATE_REPS.pair_reps[idx] 
            for idx in layer_indices 
            if idx in INTERMEDIATE_REPS.pair_reps}


def extract_structure_representations(model_output):
    """Extract structure module outputs from model."""
    if model_output is None or "sm" not in model_output:
        raise ValueError("No structure module outputs found.")
    
    sm_output = model_output["sm"]
    result = {
        'backbone_frames': sm_output['frames'],
        'angles': sm_output['angles'],
        'positions': sm_output['positions'],
        'single': sm_output.get('single', None),
    }
    
    if "final_atom_positions" in model_output:
        result['final_atom_positions'] = model_output['final_atom_positions']
    
    return result


def plot_msa_representation_heatmap(msa_tensor, layer_idx, save_path, 
                                   aggregate_method='mean', cmap='viridis'):
    """Create MSA heatmap visualization."""
    msa_2d = aggregate_channels(msa_tensor, method=aggregate_method, axis=-1)
    n_seq, n_res = msa_2d.shape
    
    fig, ax = plt.subplots(figsize=(max(10, n_res // 10), max(6, n_seq // 20)))
    
    im = ax.imshow(msa_2d, aspect='auto', cmap=cmap, interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{aggregate_method.capitalize()} activation', 
                   rotation=270, labelpad=20)
    
    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_ylabel('MSA Sequence', fontsize=12)
    ax.set_title(f'MSA Representation - Layer {layer_idx}\n'
                 f'Aggregation: {aggregate_method}',
                 fontsize=14, fontweight='bold')
    ax.grid(False)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved MSA heatmap to {save_path}")
    
    return fig


def plot_pair_representation_heatmap(pair_tensor, layer_idx, save_path,
                                    aggregate_method='mean', cmap='RdBu_r'):
    """Create Pair representation heatmap."""
    pair_2d = aggregate_channels(pair_tensor, method=aggregate_method, axis=-1)
    n_res = pair_2d.shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 9))
    
    if 'RdBu' in cmap:
        vmax = np.max(np.abs(pair_2d))
        vmin = -vmax
    else:
        vmin, vmax = None, None
    
    im = ax.imshow(pair_2d, aspect='equal', cmap=cmap, 
                   interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.plot([0, n_res-1], [0, n_res-1], 'k--', alpha=0.3, linewidth=0.5)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'{aggregate_method.capitalize()} activation', 
                   rotation=270, labelpad=20)
    
    ax.set_xlabel('Residue i', fontsize=12)
    ax.set_ylabel('Residue j', fontsize=12)
    ax.set_title(f'Pair Representation - Layer {layer_idx}\n'
                 f'Aggregation: {aggregate_method}',
                 fontsize=14, fontweight='bold')
    ax.grid(False)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Pair heatmap to {save_path}")
    
    return fig


def plot_representation_evolution(tensors_across_layers, residue_idx, save_path, 
                                 rep_type='msa'):
    """Show how a residue's representation evolves across layers."""
    layer_indices = sorted(tensors_across_layers.keys())
    magnitudes = []
    
    for layer_idx in layer_indices:
        tensor = tensors_across_layers[layer_idx]
        
        if rep_type == 'msa':
            # Average over all sequences for this residue
            residue_rep = tensor[:, residue_idx, :].mean(dim=0)
        elif rep_type == 'pair':
            # Average over all pairings for this residue
            residue_rep = tensor[residue_idx, :, :].mean(dim=0)
        else:
            raise ValueError(f"Unknown rep_type: {rep_type}. Must be 'msa' or 'pair'")
        
        # Compute L2 norm
        if isinstance(residue_rep, torch.Tensor):
            magnitude = torch.norm(residue_rep).item()
        else:
            magnitude = np.linalg.norm(residue_rep)
        
        magnitudes.append(magnitude)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(layer_indices, magnitudes, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Representation Magnitude (L2 Norm)', fontsize=12)
    ax.set_title(f'{rep_type.upper()} Representation Evolution - Residue {residue_idx}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved evolution plot to {save_path}")
    
    return fig


def plot_channel_specific_heatmap(
    tensor: torch.Tensor,
    layer_idx: int,
    channel_idx: int,
    save_path: str,
    rep_type: str = 'msa',
    cmap: str = 'viridis'
) -> plt.Figure:
    """
    Visualize a specific channel of a representation.
    
    Args:
        tensor: MSA tensor (n_seq, n_res, c_m) or Pair tensor (n_res, n_res, c_z)
        layer_idx: Which layer this is from
        channel_idx: Which channel to visualize
        save_path: Where to save the figure
        rep_type: 'msa' or 'pair'
        cmap: Matplotlib colormap name
        
    Returns:
        Matplotlib figure object
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    channel_data = tensor[..., channel_idx]
    
    if rep_type == 'msa':
        n_seq, n_res = channel_data.shape
        fig, ax = plt.subplots(figsize=(max(10, n_res // 10), max(6, n_seq // 20)))
        ylabel = 'MSA Sequence'
    elif rep_type == 'pair':
        n_res = channel_data.shape[0]
        fig, ax = plt.subplots(figsize=(10, 9))
        ylabel = 'Residue j'
    else:
        raise ValueError(f"Unknown rep_type: {rep_type}. Must be 'msa' or 'pair'")
    
    im = ax.imshow(channel_data, aspect='auto' if rep_type == 'msa' else 'equal',
                   cmap=cmap, interpolation='nearest')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Activation', rotation=270, labelpad=20)
    
    ax.set_xlabel('Residue Position' if rep_type == 'msa' else 'Residue i', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{rep_type.upper()} Representation - Layer {layer_idx}, Channel {channel_idx}',
                 fontsize=14, fontweight='bold')
    
    ax.grid(False)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved channel-specific heatmap to {save_path}")
    
    return fig


def aggregate_channels(
    tensor: torch.Tensor,
    method: str = 'mean',
    axis: int = -1
) -> np.ndarray:
    """
    Aggregate over channel dimension of a tensor.
    
    Args:
        tensor: Input tensor with channel dimension
        method: 'mean', 'max', 'norm', or 'sum'
        axis: Which axis is the channel dimension (default: -1)
        
    Returns:
        Aggregated tensor as numpy array
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    if method == 'mean':
        return np.mean(tensor, axis=axis)
    elif method == 'max':
        return np.max(tensor, axis=axis)
    elif method == 'norm':
        return np.linalg.norm(tensor, axis=axis)
    elif method == 'sum':
        return np.sum(tensor, axis=axis)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def save_intermediate_reps_to_disk(
    intermediate_reps: Dict,
    output_dir: str,
    protein_name: str
) -> None:
    """
    Save extracted intermediate representations to disk for later analysis.
    
    Args:
        intermediate_reps: Dictionary of extracted representations
                          Keys should be 'msa', 'pair', 'structure', etc.
                          Values are either tensors or dicts of tensors
        output_dir: Directory to save to
        protein_name: Name of the protein (for filename)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, f"{protein_name}_intermediate_reps.pt")
    
    # Convert tensors to CPU before saving
    def to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        elif isinstance(obj, dict):
            return {k: to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_cpu(item) for item in obj]
        else:
            return obj
    
    intermediate_reps_cpu = to_cpu(intermediate_reps)
    
    torch.save(intermediate_reps_cpu, save_path)
    print(f"Saved intermediate representations to {save_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"{protein_name}_metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"Protein: {protein_name}\n")
        f.write(f"Representations saved: {list(intermediate_reps.keys())}\n")
        for key, value in intermediate_reps.items():
            if isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for layer_idx, tensor in value.items():
                    if isinstance(tensor, torch.Tensor):
                        f.write(f"  Layer {layer_idx}: shape {tuple(tensor.shape)}\n")
            elif isinstance(value, torch.Tensor):
                f.write(f"{key}: shape {tuple(value.shape)}\n")
    
    print(f"Saved metadata to {metadata_path}")


def load_intermediate_reps_from_disk(
    input_path: str
) -> Dict:
    """
    Load previously saved intermediate representations.
    
    Args:
        input_path: Path to saved representations (.pt file)
        
    Returns:
        Dictionary of representations
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    
    intermediate_reps = torch.load(input_path, map_location='cpu')
    
    print(f"Loaded intermediate representations from {input_path}")
    print(f"Available keys: {list(intermediate_reps.keys())}")
    
    return intermediate_reps


def register_evoformer_hooks(model):
    """
    Register forward hooks on Evoformer blocks to capture intermediate representations.
    
    This function attaches hooks to the model's Evoformer blocks to store
    MSA and Pair representations after each layer.
    
    Args:
        model: OpenFold AlphaFold model instance
        
    Usage:
        INTERMEDIATE_REPS.enable()
        register_evoformer_hooks(model)
        output = model(batch)
        msa_reps = extract_msa_representations(None)
        pair_reps = extract_pair_representations(None)
    """
    hooks = []
    
    def make_hook(layer_idx):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input_tuple, output_tuple):
            if INTERMEDIATE_REPS.enabled:
                m, z = output_tuple
                # Store copies to avoid issues with in-place operations
                INTERMEDIATE_REPS.msa_reps[layer_idx] = m.detach().clone()
                INTERMEDIATE_REPS.pair_reps[layer_idx] = z.detach().clone()
        return hook_fn
    
    # Register hooks on each Evoformer block
    if hasattr(model, 'evoformer') and hasattr(model.evoformer, 'blocks'):
        for idx, block in enumerate(model.evoformer.blocks):
            hook = block.register_forward_hook(make_hook(idx))
            hooks.append(hook)
        print(f"Registered {len(hooks)} forward hooks on Evoformer blocks")
    else:
        raise AttributeError(
            "Model does not have expected structure. "
            "Expected model.evoformer.blocks"
        )
    
    return hooks


def remove_hooks(hooks):
    """Remove all registered hooks."""
    for hook in hooks:
        hook.remove()
    print(f"Removed {len(hooks)} hooks")


if __name__ == "__main__":
    print("Intermediate Representation Visualization Utilities")
    print("=" * 60)
    print("\nThis module provides functions for:")
    print("  - Extracting MSA, Pair, and Structure representations")
    print("  - Visualizing representations as heatmaps and line plots")
    print("\nStatus: ✓ IMPLEMENTED")
    print("\nFeatures:")
    print("  [✓] Extract MSA representations")
    print("  [✓] Extract Pair representations")
    print("  [✓] Extract Structure module outputs")
    print("  [✓] Visualize MSA as heatmaps")
    print("  [✓] Visualize Pair representations")
    print("  [✓] Track representation evolution across layers")
    print("  [✓] Channel-specific visualization")
    print("  [✓] Save/load representations to disk")
    print("\nNext steps:")
    print("  - Run unit tests: pytest tests/test_intermediate_extraction.py")
    print("  - Create demo notebook")
    print("  - Integrate with web interface")
    print("\nExample usage:")
    print("  from visualize_intermediate_reps_utils import *")
    print("  INTERMEDIATE_REPS.enable()")
    print("  hooks = register_evoformer_hooks(model)")
    print("  output = model(batch)")
    print("  msa_reps = extract_msa_representations(None)")
    print("  plot_msa_representation_heatmap(msa_reps[0], 0, 'msa_layer0.png')")