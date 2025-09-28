"""
Utility functions for extracting and visualizing intermediate representations 
from OpenFold's deep neural network.

This module provides tools to:
- Extract MSA (Multiple Sequence Alignment) representations
- Extract Pair representations  
- Extract Structure module outputs
- Visualize these representations as heatmaps and line plots

Author: Working on Issue #8
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Optional


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_msa_representations(
    model_output: Dict,
    layer_indices: Optional[List[int]] = None
) -> Dict[int, torch.Tensor]:
    """
    Extract MSA representation tensors from OpenFold forward pass.
    
    Args:
        model_output: Dictionary containing model outputs including intermediate reps
        layer_indices: Which layers to extract (None = all layers)
        
    Returns:
        Dictionary mapping layer index to MSA tensor
        Shape of each tensor: (n_seq, n_res, c_m) where:
            - n_seq: number of sequences in MSA
            - n_res: number of residues
            - c_m: MSA channel dimension (typically 256)
    
    TODO: Implement extraction logic
    """
    # TODO: Implement this function
    # Hint: Look at how attention scores are extracted in 
    # visualize_attention_general_utils.py
    raise NotImplementedError("MSA extraction not yet implemented")


def extract_pair_representations(
    model_output: Dict,
    layer_indices: Optional[List[int]] = None
) -> Dict[int, torch.Tensor]:
    """
    Extract Pair representation tensors from OpenFold forward pass.
    
    Args:
        model_output: Dictionary containing model outputs including intermediate reps
        layer_indices: Which layers to extract (None = all layers)
        
    Returns:
        Dictionary mapping layer index to Pair tensor
        Shape of each tensor: (n_res, n_res, c_z) where:
            - n_res: number of residues
            - c_z: Pair channel dimension (typically 128)
    
    TODO: Implement extraction logic
    """
    # TODO: Implement this function
    raise NotImplementedError("Pair extraction not yet implemented")


def extract_structure_representations(
    model_output: Dict
) -> Dict[str, torch.Tensor]:
    """
    Extract Structure module outputs from OpenFold forward pass.
    
    Args:
        model_output: Dictionary containing model outputs
        
    Returns:
        Dictionary with keys:
            - 'backbone_frames': Rigid body transformations for backbone
            - 'torsion_angles': Sidechain torsion angles
            - 'positions': Final atomic positions
    
    TODO: Implement extraction logic
    """
    # TODO: Implement this function
    raise NotImplementedError("Structure extraction not yet implemented")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_msa_representation_heatmap(
    msa_tensor: torch.Tensor,
    layer_idx: int,
    save_path: str,
    aggregate_method: str = 'mean',
    cmap: str = 'viridis'
) -> plt.Figure:
    """
    Visualize MSA representation as a heatmap.
    
    Args:
        msa_tensor: Shape (n_seq, n_res, c_m)
        layer_idx: Which layer this representation is from
        save_path: Where to save the figure
        aggregate_method: How to aggregate over channels ('mean', 'max', 'norm')
        cmap: Matplotlib colormap name
        
    Returns:
        Matplotlib figure object
    
    TODO: Implement visualization
    """
    # TODO: Implement this function
    # Steps:
    # 1. Aggregate over channel dimension
    # 2. Create heatmap with sequences on Y-axis, residues on X-axis
    # 3. Add labels and colorbar
    # 4. Save figure
    raise NotImplementedError("MSA heatmap visualization not yet implemented")


def plot_pair_representation_heatmap(
    pair_tensor: torch.Tensor,
    layer_idx: int,
    save_path: str,
    aggregate_method: str = 'mean',
    cmap: str = 'RdBu_r'
) -> plt.Figure:
    """
    Visualize Pair representation as a contact-map-like heatmap.
    
    Args:
        pair_tensor: Shape (n_res, n_res, c_z)
        layer_idx: Which layer this representation is from
        save_path: Where to save the figure
        aggregate_method: How to aggregate over channels ('mean', 'max', 'norm')
        cmap: Matplotlib colormap name
        
    Returns:
        Matplotlib figure object
    
    TODO: Implement visualization
    """
    # TODO: Implement this function
    # Steps:
    # 1. Aggregate over channel dimension
    # 2. Create symmetric heatmap (residue i vs residue j)
    # 3. Add diagonal line for reference
    # 4. Add labels and colorbar
    # 5. Save figure
    raise NotImplementedError("Pair heatmap visualization not yet implemented")


def plot_representation_evolution(
    tensors_across_layers: Dict[int, torch.Tensor],
    residue_idx: int,
    save_path: str,
    rep_type: str = 'msa'
) -> plt.Figure:
    """
    Show how a specific residue's representation changes across layers.
    
    Args:
        tensors_across_layers: Dictionary mapping layer index to tensor
        residue_idx: Which residue to track
        save_path: Where to save the figure
        rep_type: 'msa' or 'pair'
        
    Returns:
        Matplotlib figure object
    
    TODO: Implement visualization
    """
    # TODO: Implement this function
    # Steps:
    # 1. Extract representation for specific residue from each layer
    # 2. Compute magnitude or norm across channel dimension
    # 3. Plot line: X=layer number, Y=representation magnitude
    # 4. Add labels and grid
    # 5. Save figure
    raise NotImplementedError("Evolution plot not yet implemented")


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
    
    This allows examining individual feature channels rather than aggregating.
    
    Args:
        tensor: MSA tensor (n_seq, n_res, c_m) or Pair tensor (n_res, n_res, c_z)
        layer_idx: Which layer this is from
        channel_idx: Which channel to visualize
        save_path: Where to save the figure
        rep_type: 'msa' or 'pair'
        cmap: Matplotlib colormap name
        
    Returns:
        Matplotlib figure object
    
    TODO: Implement visualization
    """
    # TODO: Implement this function
    raise NotImplementedError("Channel-specific visualization not yet implemented")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
        output_dir: Directory to save to
        protein_name: Name of the protein (for filename)
    
    TODO: Implement saving logic
    """
    # TODO: Implement this function
    # Consider using torch.save() or numpy.save()
    raise NotImplementedError("Saving to disk not yet implemented")


def load_intermediate_reps_from_disk(
    input_path: str
) -> Dict:
    """
    Load previously saved intermediate representations.
    
    Args:
        input_path: Path to saved representations
        
    Returns:
        Dictionary of representations
    
    TODO: Implement loading logic
    """
    # TODO: Implement this function
    raise NotImplementedError("Loading from disk not yet implemented")


# ============================================================================
# MAIN ENTRY POINT (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Intermediate Representation Visualization Utilities")
    print("=" * 60)
    print("\nThis module provides functions for:")
    print("  - Extracting MSA, Pair, and Structure representations")
    print("  - Visualizing representations as heatmaps and line plots")
    print("\nStatus: Under development for Issue #8")
    print("\nTODO:")
    print("  [ ] Implement extraction functions")
    print("  [ ] Implement visualization functions")
    print("  [ ] Add unit tests")
    print("  [ ] Create demo notebook")

