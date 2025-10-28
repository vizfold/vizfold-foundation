#!/usr/bin/env python3

import torch
import numpy as np
import os
from visualize_intermediate_reps_utils import *

print("OpenFold Intermediate Representation Demo")
print("-" * 40)

# Create mock protein data
print("Creating mock protein data...")
n_seq = 15
n_res = 100
c_m = 256
c_z = 128

msa_tensor = torch.randn(n_seq, n_res, c_m)
pair_tensor = torch.randn(n_res, n_res, c_z)

mock_output = {
    'msa': msa_tensor,
    'pair': pair_tensor,
    'sm': {
        'frames': torch.randn(8, n_res, 7),
        'angles': torch.randn(8, n_res, 7, 2),
        'positions': torch.randn(8, n_res, 14, 3),
        'single': torch.randn(n_res, 384)
    },
    'final_atom_positions': torch.randn(n_res, 37, 3)
}

print(f"MSA shape: {msa_tensor.shape}")
print(f"Pair shape: {pair_tensor.shape}")

# Extract representations
print("\nExtracting representations...")
msa_final = extract_msa_representations(mock_output)
pair_final = extract_pair_representations(mock_output)
structure = extract_structure_representations(mock_output)
print("Done")

# Create visualizations
print("\nCreating visualizations...")
output_dir = "demo_outputs"
os.makedirs(output_dir, exist_ok=True)

# MSA heatmap
plot_msa_representation_heatmap(msa_final[-1], -1, 
                                f"{output_dir}/msa_final.png",
                                highlight_residue=50, custom_ticks=[0, 25, 50, 75])
print("  MSA heatmap saved")

# Pair heatmap with contact map
mock_contact_map = generate_mock_contact_map(n_res, contact_probability=0.15, seed=42)
plot_pair_representation_heatmap(pair_final[-1], -1,
                                f"{output_dir}/pair_final.png",
                                contact_map=mock_contact_map, show_correlation=True)
print("  Pair heatmap with contact overlay saved")

# Layer-by-layer analysis
print("\nCreating mock layer data...")
msa_layers = {}
pair_layers = {}

for i in range(5):
    noise = torch.randn_like(msa_tensor) * 0.1
    msa_layers[i] = msa_tensor + noise
    noise = torch.randn_like(pair_tensor) * 0.1
    pair_layers[i] = pair_tensor + noise

# Evolution plot with multiple residues
plot_representation_evolution(msa_layers, 50, 
                             f"{output_dir}/evolution.png", 'msa',
                             multiple_residues=[10, 25, 50, 75, 90])
print("  Evolution plot with multiple residues saved")

# Comprehensive analysis
print("\nRunning comprehensive analysis...")
create_comprehensive_visualization_report(msa_layers, structure, output_dir)
print("  Comprehensive analysis complete")

print(f"\nDemo complete! Files in: {output_dir}")
files = os.listdir(output_dir)
print(f"Generated {len(files)} files")