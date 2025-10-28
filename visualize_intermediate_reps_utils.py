"""
OpenFold Intermediate Representation Visualization

Core functions for visualizing MSA, Pair, and Structure representations
from OpenFold's 48-layer Evoformer network.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


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


def aggregate_channels(tensor, method='mean', axis=-1):
    if method == 'mean':
        return torch.mean(tensor, dim=axis).cpu().numpy()
    elif method == 'max':
        return torch.max(tensor, dim=axis)[0].cpu().numpy()
    elif method == 'sum':
        return torch.sum(tensor, dim=axis).cpu().numpy()
    elif method == 'std':
        return torch.std(tensor, dim=axis).cpu().numpy()
    else:
        return torch.mean(tensor, dim=axis).cpu().numpy()


def plot_msa_representation_heatmap(msa_tensor, layer_idx, save_path, 
                                   aggregate_method='mean', cmap='viridis',
                                   highlight_residue=None, custom_ticks=None):
    msa_2d = aggregate_channels(msa_tensor, method=aggregate_method, axis=-1)
    n_seq, n_res = msa_2d.shape
    
    fig, ax = plt.subplots(figsize=(max(10, n_res // 10), max(6, n_seq // 20)))
    
    im = ax.imshow(msa_2d, aspect='auto', cmap=cmap, interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{aggregate_method.capitalize()} activation', 
                   rotation=270, labelpad=20)
    
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('MSA Sequence Index')
    ax.set_title(f'MSA Representation - Layer {layer_idx} ({aggregate_method})')
    
    if highlight_residue is not None:
        ax.axvline(x=highlight_residue, color='red', linestyle='--', alpha=0.7)
    
    if custom_ticks is not None:
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels(custom_ticks)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pair_representation_heatmap(pair_tensor, layer_idx, save_path,
                                    aggregate_method='mean', cmap='RdBu_r',
                                    contact_map=None, show_correlation=False):
    pair_2d = aggregate_channels(pair_tensor, method=aggregate_method, axis=-1)
    n_res = pair_2d.shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(pair_2d, aspect='equal', cmap=cmap, interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{aggregate_method.capitalize()} activation', 
                   rotation=270, labelpad=20)
    
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Residue Index')
    ax.set_title(f'Pair Representation - Layer {layer_idx} ({aggregate_method})')
    
    if contact_map is not None:
        contact_indices = np.where(contact_map > 0.5)
        ax.scatter(contact_indices[1], contact_indices[0], 
                  c='yellow', s=1, alpha=0.6, marker='.')
    
    if show_correlation and contact_map is not None:
        correlation = np.corrcoef(pair_2d.flatten(), contact_map.flatten())[0, 1]
        ax.text(0.02, 0.98, f'Contact Correlation: {correlation:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_representation_evolution(layer_representations, residue_idx, save_path,
                                 rep_type='msa', multiple_residues=None,
                                 show_confidence=False, show_differences=False):
    layers = sorted(layer_representations.keys())
    magnitudes = []
    
    for layer in layers:
        tensor = layer_representations[layer]
        if rep_type == 'msa':
            mag = torch.norm(tensor[:, residue_idx, :], dim=-1).mean().item()
        else:
            mag = torch.norm(tensor[residue_idx, :, :], dim=-1).mean().item()
        magnitudes.append(mag)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if multiple_residues is not None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(multiple_residues)))
        for i, res_idx in enumerate(multiple_residues):
            res_magnitudes = []
            for layer in layers:
                tensor = layer_representations[layer]
                if rep_type == 'msa':
                    mag = torch.norm(tensor[:, res_idx, :], dim=-1).mean().item()
                else:
                    mag = torch.norm(tensor[res_idx, :, :], dim=-1).mean().item()
                res_magnitudes.append(mag)
            
            ax.plot(layers, res_magnitudes, 'o-', color=colors[i], 
                   label=f'Residue {res_idx}', linewidth=2, markersize=4)
    else:
        ax.plot(layers, magnitudes, 'o-', color='blue', linewidth=2, markersize=4)
    
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Representation Magnitude (L2 Norm)')
    ax.set_title(f'{rep_type.upper()} Representation Evolution')
    ax.grid(True, alpha=0.3)
    
    if multiple_residues is not None:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_mock_contact_map(n_res, contact_probability=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    contact_map = np.zeros((n_res, n_res))
    for i in range(n_res):
        for j in range(i+5, n_res):
            if np.random.random() < contact_probability:
                contact_map[i, j] = 1
                contact_map[j, i] = 1
    
    return contact_map


def stratified_layer_sampling(n_layers=48, strategy='uniform', n_samples=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if strategy == 'uniform':
        if n_samples is None:
            n_samples = min(12, n_layers)
        return np.linspace(0, n_layers-1, n_samples, dtype=int).tolist()
    
    elif strategy == 'grouped':
        if n_samples is None:
            n_samples = 12
        groups = [0, 6, 12, 18, 24, 30, 36, 42, 47]
        return groups[:n_samples]
    
    elif strategy == 'logarithmic':
        if n_samples is None:
            n_samples = 10
        log_indices = np.logspace(0, np.log10(n_layers-1), n_samples, dtype=int)
        return np.unique(log_indices).tolist()
    
    elif strategy == 'dense':
        return list(range(0, n_layers, 2))
    
    elif strategy == 'random':
        if n_samples is None:
            n_samples = 15
        return sorted(np.random.choice(n_layers, n_samples, replace=False).tolist())
    
    elif strategy == 'all':
        return list(range(n_layers))
    
    else:
        return list(range(min(n_samples or 10, n_layers)))


def plot_multilayer_evolution(layer_representations, residue_indices, save_path,
                             rep_type='msa', layer_sampling='uniform'):
    layers = sorted(layer_representations.keys())
    sampled_layers = stratified_layer_sampling(len(layers), strategy=layer_sampling)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(residue_indices)))
    
    for i, res_idx in enumerate(residue_indices):
        magnitudes = []
        for layer in sampled_layers:
            if layer in layer_representations:
                tensor = layer_representations[layer]
                if rep_type == 'msa':
                    mag = torch.norm(tensor[:, res_idx, :], dim=-1).mean().item()
                else:
                    mag = torch.norm(tensor[res_idx, :, :], dim=-1).mean().item()
                magnitudes.append(mag)
        
        ax.plot(sampled_layers, magnitudes, 'o-', color=colors[i], 
               label=f'Residue {res_idx}', linewidth=2, markersize=3)
    
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Representation Magnitude (L2 Norm)')
    ax.set_title(f'{rep_type.upper()} Evolution Across {len(sampled_layers)} Layers')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_stratified_layer_comparison(layer_representations, layer_indices, save_path,
                                    rep_type='msa', aggregate_method='mean'):
    n_layers = len(layer_indices)
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, layer_idx in enumerate(layer_indices):
        if layer_idx in layer_representations:
            tensor = layer_representations[layer_idx]
            if rep_type == 'msa':
                data_2d = aggregate_channels(tensor, method=aggregate_method, axis=-1)
                cmap = 'viridis'
            else:
                data_2d = aggregate_channels(tensor, method=aggregate_method, axis=-1)
                cmap = 'RdBu_r'
            
            im = axes[i].imshow(data_2d, aspect='auto', cmap=cmap, interpolation='nearest')
            axes[i].set_title(f'Layer {layer_idx}')
            axes[i].set_xlabel('Residue Index')
            if rep_type == 'msa':
                axes[i].set_ylabel('MSA Index')
            else:
                axes[i].set_ylabel('Residue Index')
    
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{rep_type.upper()} Stratified Layer Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_layer_convergence_analysis(layer_representations, save_path, rep_type='msa'):
    layers = sorted(layer_representations.keys())
    convergence_metrics = []
    
    for i in range(1, len(layers)):
        prev_tensor = layer_representations[layers[i-1]]
        curr_tensor = layer_representations[layers[i]]
        
        if rep_type == 'msa':
            prev_flat = prev_tensor.mean(dim=0).flatten()
            curr_flat = curr_tensor.mean(dim=0).flatten()
        else:
            prev_flat = prev_tensor.flatten()
            curr_flat = curr_tensor.flatten()
        
        diff = torch.norm(curr_flat - prev_flat).item()
        convergence_metrics.append(diff)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(layers[1:], convergence_metrics, 'o-', color='red', linewidth=2, markersize=4)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Representation Change (L2 Norm)')
    ax.set_title(f'{rep_type.upper()} Layer Convergence Analysis')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_structure_module_evolution(structure_outputs, save_path):
    frames = structure_outputs['backbone_frames']
    angles = structure_outputs['angles']
    positions = structure_outputs['positions']
    
    n_recycles = len(frames)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # RMSD evolution
    rmsd_values = []
    for i in range(1, n_recycles):
        pos1 = positions[i-1]
        pos2 = positions[i]
        rmsd = torch.sqrt(torch.mean(torch.sum((pos1 - pos2)**2, dim=-1))).item()
        rmsd_values.append(rmsd)
    
    axes[0, 0].plot(range(1, n_recycles), rmsd_values, 'o-', color='blue')
    axes[0, 0].set_xlabel('Recycle Index')
    axes[0, 0].set_ylabel('RMSD (Å)')
    axes[0, 0].set_title('Structure Convergence (RMSD)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Angle evolution
    phi_angles = angles[:, :, 0].mean(dim=1).cpu().numpy()
    psi_angles = angles[:, :, 1].mean(dim=1).cpu().numpy()
    
    axes[0, 1].plot(range(n_recycles), phi_angles, 'o-', label='Phi', color='red')
    axes[0, 1].plot(range(n_recycles), psi_angles, 'o-', label='Psi', color='green')
    axes[0, 1].set_xlabel('Recycle Index')
    axes[0, 1].set_ylabel('Angle (radians)')
    axes[0, 1].set_title('Backbone Angle Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Position displacement
    displacements = []
    for i in range(1, n_recycles):
        disp = torch.norm(positions[i] - positions[i-1], dim=-1).mean().item()
        displacements.append(disp)
    
    axes[0, 2].plot(range(1, n_recycles), displacements, 'o-', color='purple')
    axes[0, 2].set_xlabel('Recycle Index')
    axes[0, 2].set_ylabel('Mean Displacement (Å)')
    axes[0, 2].set_title('Atomic Displacement')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 3D trajectory
    final_pos = positions[-1].cpu().numpy()
    axes[1, 0].plot(final_pos[:, 0], final_pos[:, 1], 'o-', color='blue', markersize=2)
    axes[1, 0].set_xlabel('X (Å)')
    axes[1, 0].set_ylabel('Y (Å)')
    axes[1, 0].set_title('Final Structure (XY projection)')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Frame evolution
    frame_norms = []
    for i in range(n_recycles):
        frame_norm = torch.norm(frames[i], dim=-1).mean().item()
        frame_norms.append(frame_norm)
    
    axes[1, 1].plot(range(n_recycles), frame_norms, 'o-', color='orange')
    axes[1, 1].set_xlabel('Recycle Index')
    axes[1, 1].set_ylabel('Frame Norm')
    axes[1, 1].set_title('Backbone Frame Evolution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Angle distribution
    final_phi = angles[-1, :, 0].cpu().numpy()
    final_psi = angles[-1, :, 1].cpu().numpy()
    
    axes[1, 2].scatter(final_phi, final_psi, alpha=0.6, s=10)
    axes[1, 2].set_xlabel('Phi (radians)')
    axes[1, 2].set_ylabel('Psi (radians)')
    axes[1, 2].set_title('Final Ramachandran Plot')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_layer_importance_metrics(layer_representations, metric='variance'):
    layers = sorted(layer_representations.keys())
    metrics = []
    
    for layer in layers:
        tensor = layer_representations[layer]
        
        if metric == 'variance':
            metric_val = torch.var(tensor).item()
        elif metric == 'entropy':
            probs = torch.softmax(tensor.flatten(), dim=0)
            metric_val = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        elif metric == 'sparsity':
            metric_val = torch.mean((tensor == 0).float()).item()
        elif metric == 'norm':
            metric_val = torch.norm(tensor).item()
        else:
            metric_val = torch.var(tensor).item()
        
        metrics.append(metric_val)
    
    return dict(zip(layers, metrics))


def plot_layer_importance_ranking(layer_representations, save_path, 
                                 metrics=['variance', 'entropy', 'norm']):
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        importance_scores = compute_layer_importance_metrics(layer_representations, metric)
        layers = list(importance_scores.keys())
        scores = list(importance_scores.values())
        
        sorted_indices = np.argsort(scores)[::-1]
        sorted_layers = [layers[idx] for idx in sorted_indices]
        sorted_scores = [scores[idx] for idx in sorted_indices]
        
        axes[i].bar(range(len(sorted_layers)), sorted_scores, color='skyblue')
        axes[i].set_xlabel('Layer Rank')
        axes[i].set_ylabel(f'{metric.capitalize()} Score')
        axes[i].set_title(f'Layer Importance ({metric.capitalize()})')
        axes[i].set_xticks(range(0, len(sorted_layers), max(1, len(sorted_layers)//10)))
        axes[i].set_xticklabels([sorted_layers[j] for j in range(0, len(sorted_layers), max(1, len(sorted_layers)//10))])
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residue_feature_analysis(tensor, residue_idx, save_path, rep_type='msa'):
    if rep_type == 'msa':
        features = tensor[:, residue_idx, :].cpu().numpy()
        n_seq, n_features = features.shape
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Feature distribution
        axes[0, 0].hist(features.flatten(), bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Feature Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'Feature Distribution (Residue {residue_idx})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Sequence-wise variation
        seq_vars = np.var(features, axis=1)
        axes[0, 1].plot(seq_vars, 'o-', color='red')
        axes[0, 1].set_xlabel('Sequence Index')
        axes[0, 1].set_ylabel('Feature Variance')
        axes[0, 1].set_title('Sequence-wise Feature Variation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature correlation
        corr_matrix = np.corrcoef(features.T)
        im = axes[1, 0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 0].set_title('Feature Correlation Matrix')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Top features
        feature_means = np.mean(np.abs(features), axis=0)
        top_features = np.argsort(feature_means)[-20:]
        axes[1, 1].bar(range(len(top_features)), feature_means[top_features], color='green')
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('Mean Absolute Value')
        axes[1, 1].set_title('Top 20 Most Active Features')
        axes[1, 1].grid(True, alpha=0.3)
    
    else:  # pair representation
        features = tensor[residue_idx, :, :].cpu().numpy()
        n_res, n_features = features.shape
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Feature distribution
        axes[0, 0].hist(features.flatten(), bins=50, alpha=0.7, color='purple')
        axes[0, 0].set_xlabel('Feature Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'Feature Distribution (Residue {residue_idx})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residue-wise variation
        res_vars = np.var(features, axis=1)
        axes[0, 1].plot(res_vars, 'o-', color='orange')
        axes[0, 1].set_xlabel('Residue Index')
        axes[0, 1].set_ylabel('Feature Variance')
        axes[0, 1].set_title('Residue-wise Feature Variation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature correlation
        corr_matrix = np.corrcoef(features.T)
        im = axes[1, 0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 0].set_title('Feature Correlation Matrix')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Top features
        feature_means = np.mean(np.abs(features), axis=0)
        top_features = np.argsort(feature_means)[-20:]
        axes[1, 1].bar(range(len(top_features)), feature_means[top_features], color='red')
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('Mean Absolute Value')
        axes[1, 1].set_title('Top 20 Most Active Features')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_layer_clustering_dendrogram(layer_representations, save_path, method='ward'):
    layers = sorted(layer_representations.keys())
    representations = []
    
    for layer in layers:
        tensor = layer_representations[layer]
        flat_repr = tensor.flatten().cpu().numpy()
        representations.append(flat_repr)
    
    representations = np.array(representations)
    
    # Compute distance matrix
    distances = pdist(representations, metric='euclidean')
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(distances, method=method)
    
    # Plot dendrogram
    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=layers, ax=ax, leaf_rotation=90)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Distance')
    ax.set_title(f'Layer Clustering Dendrogram ({method.capitalize()} linkage)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_comprehensive_visualization_report(layer_representations, structure_outputs, 
                                             output_dir='demo_outputs'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate contact map
    n_res = list(layer_representations.values())[0].shape[1]
    contact_map = generate_mock_contact_map(n_res, contact_probability=0.15, seed=42)
    
    # Multi-layer evolution
    plot_multilayer_evolution(layer_representations, [10, 50, 100, 150, 200], 
                             f'{output_dir}/multilayer_evolution.png', 
                             rep_type='msa', layer_sampling='all')
    
    # Stratified comparisons
    sample_layers = stratified_layer_sampling(48, strategy='all')
    plot_stratified_layer_comparison(layer_representations, sample_layers,
                                   f'{output_dir}/stratified_msa_comparison.png',
                                   rep_type='msa')
    
    # Convergence analysis
    plot_layer_convergence_analysis(layer_representations, 
                                   f'{output_dir}/msa_convergence_analysis.png',
                                   rep_type='msa')
    
    # Enhanced pair heatmap with contacts
    final_layer = max(layer_representations.keys())
    plot_pair_representation_heatmap(layer_representations[final_layer], final_layer,
                                    f'{output_dir}/pair_layer{final_layer}_with_contacts.png',
                                    contact_map=contact_map, show_correlation=True)
    
    # Structure module analysis
    plot_structure_module_evolution(structure_outputs, 
                                   f'{output_dir}/structure_evolution.png')
    
    # Layer importance ranking
    plot_layer_importance_ranking(layer_representations,
                                 f'{output_dir}/layer_importance.png')
    
    # Layer clustering
    plot_layer_clustering_dendrogram(layer_representations,
                                    f'{output_dir}/layer_clustering.png')
    
    print(f"Comprehensive visualization report generated in {output_dir}/")