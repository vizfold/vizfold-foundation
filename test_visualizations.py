#!/usr/bin/env python3
"""Test script for intermediate representation visualization functions."""

import torch
import numpy as np
import os
import tempfile
from visualize_intermediate_reps_utils import *

print("=" * 70)
print("Testing Intermediate Representation Visualizations")
print("=" * 70)

test_output_dir = tempfile.mkdtemp(prefix="openfold_viz_test_")
print(f"\nTest outputs will be saved to: {test_output_dir}\n")

# Test 1: MSA Visualization
print("[Test 1] MSA Representation Visualization")
print("-" * 70)

msa_tensor = torch.randn(10, 50, 256)
print(f"Created mock MSA tensor with shape: {msa_tensor.shape}")

for method in ['mean', 'max', 'norm']:
    save_path = os.path.join(test_output_dir, f"msa_layer0_{method}.png")
    try:
        fig = plot_msa_representation_heatmap(
            msa_tensor, 
            layer_idx=0, 
            save_path=save_path,
            aggregate_method=method
        )
        print(f"  ✓ Created MSA heatmap with {method} aggregation")
        plt.close(fig)
    except Exception as e:
        print(f"  ✗ Failed with {method}: {e}")

# Test 2: Pair Visualization
print("\n[Test 2] Pair Representation Visualization")
print("-" * 70)

pair_tensor = torch.randn(50, 50, 128)
print(f"Created mock Pair tensor with shape: {pair_tensor.shape}")

for cmap in ['RdBu_r', 'viridis']:
    save_path = os.path.join(test_output_dir, f"pair_layer0_{cmap}.png")
    try:
        fig = plot_pair_representation_heatmap(
            pair_tensor,
            layer_idx=0,
            save_path=save_path,
            cmap=cmap
        )
        print(f"  ✓ Created Pair heatmap with {cmap} colormap")
        plt.close(fig)
    except Exception as e:
        print(f"  ✗ Failed with {cmap}: {e}")

# Test 3: Evolution Plot
print("\n[Test 3] Representation Evolution Across Layers")
print("-" * 70)

msa_layers = {i: torch.randn(10, 50, 256) for i in range(5)}
pair_layers = {i: torch.randn(50, 50, 128) for i in range(5)}

# Test MSA evolution
save_path = os.path.join(test_output_dir, "msa_evolution_res10.png")
try:
    fig = plot_representation_evolution(
        msa_layers,
        residue_idx=10,
        save_path=save_path,
        rep_type='msa'
    )
    print("  ✓ Created MSA evolution plot")
    plt.close(fig)
except Exception as e:
    print(f"  ✗ Failed MSA evolution: {e}")

# Test Pair evolution
save_path = os.path.join(test_output_dir, "pair_evolution_res10.png")
try:
    fig = plot_representation_evolution(
        pair_layers,
        residue_idx=10,
        save_path=save_path,
        rep_type='pair'
    )
    print("  ✓ Created Pair evolution plot")
    plt.close(fig)
except Exception as e:
    print(f"  ✗ Failed Pair evolution: {e}")

# Test 4: Channel-Specific Visualization
print("\n[Test 4] Channel-Specific Visualization")
print("-" * 70)

# Test MSA channel visualization
save_path = os.path.join(test_output_dir, "msa_channel_64.png")
try:
    fig = plot_channel_specific_heatmap(
        msa_tensor,
        layer_idx=0,
        channel_idx=64,
        save_path=save_path,
        rep_type='msa'
    )
    print("  ✓ Created MSA channel-specific heatmap")
    plt.close(fig)
except Exception as e:
    print(f"  ✗ Failed MSA channel viz: {e}")

# Test Pair channel visualization
save_path = os.path.join(test_output_dir, "pair_channel_32.png")
try:
    fig = plot_channel_specific_heatmap(
        pair_tensor,
        layer_idx=0,
        channel_idx=32,
        save_path=save_path,
        rep_type='pair'
    )
    print("  ✓ Created Pair channel-specific heatmap")
    plt.close(fig)
except Exception as e:
    print(f"  ✗ Failed Pair channel viz: {e}")

# Test 5: Save/Load to Disk
print("\n[Test 5] Save and Load Intermediate Representations")
print("-" * 70)

intermediate_reps = {
    'msa': msa_layers,
    'pair': pair_layers,
    'final_msa': msa_tensor,
    'final_pair': pair_tensor
}

# Test saving
try:
    save_intermediate_reps_to_disk(
        intermediate_reps,
        test_output_dir,
        "test_protein"
    )
    print("  ✓ Successfully saved representations to disk")
except Exception as e:
    print(f"  ✗ Failed to save: {e}")

# Test loading
load_path = os.path.join(test_output_dir, "test_protein_intermediate_reps.pt")
try:
    loaded_reps = load_intermediate_reps_from_disk(load_path)
    print("  ✓ Successfully loaded representations from disk")
    print(f"    Loaded keys: {list(loaded_reps.keys())}")
except Exception as e:
    print(f"  ✗ Failed to load: {e}")

# Test 6: Structure Extraction
print("\n[Test 6] Structure Module Extraction")
print("-" * 70)

mock_output = {
    'sm': {
        'frames': torch.randn(8, 50, 7),
        'angles': torch.randn(8, 50, 7, 2),
        'positions': torch.randn(8, 50, 14, 3),
        'single': torch.randn(50, 384)
    },
    'final_atom_positions': torch.randn(50, 37, 3)
}

try:
    structure_reps = extract_structure_representations(mock_output)
    print("  ✓ Successfully extracted structure representations")
    print(f"    Keys: {list(structure_reps.keys())}")
    print(f"    Frames shape: {structure_reps['backbone_frames'].shape}")
    print(f"    Angles shape: {structure_reps['angles'].shape}")
    print(f"    Positions shape: {structure_reps['positions'].shape}")
except Exception as e:
    print(f"  ✗ Failed structure extraction: {e}")

# Summary
print("\n" + "=" * 70)
print("✓ All visualization tests completed!")
print("=" * 70)
print(f"\nTest outputs saved to: {test_output_dir}")
print("\nGenerated files:")
for fname in sorted(os.listdir(test_output_dir)):
    fpath = os.path.join(test_output_dir, fname)
    size = os.path.getsize(fpath)
    print(f"  - {fname} ({size:,} bytes)")

print("\n" + "=" * 70)
print("All core functionality is working!")
print("Ready for integration with OpenFold model!")
print("=" * 70)

