#!/usr/bin/env python3
"""Quick demo showing how to use the intermediate representation visualization system."""

import torch
import numpy as np
import os
from visualize_intermediate_reps_utils import *

print("🚀 OpenFold Intermediate Representation Visualization Demo")
print("=" * 60)

# Create mock data similar to real OpenFold outputs
print("\n📊 Creating mock protein data...")
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

print(f"✅ Created mock data:")
print(f"   MSA shape: {msa_tensor.shape}")
print(f"   Pair shape: {pair_tensor.shape}")
print(f"   Structure frames: {mock_output['sm']['frames'].shape}")

# Demo 1: Extract Final Representations
print("\n🔍 Demo 1: Extracting Final Representations")
print("-" * 40)

msa_final = extract_msa_representations(mock_output)
pair_final = extract_pair_representations(mock_output)
structure = extract_structure_representations(mock_output)

print(f"✅ Extracted MSA representations: {list(msa_final.keys())}")
print(f"✅ Extracted Pair representations: {list(pair_final.keys())}")
print(f"✅ Extracted Structure outputs: {list(structure.keys())}")

# Demo 2: Create Visualizations
print("\n🎨 Demo 2: Creating Visualizations")
print("-" * 40)

output_dir = "demo_outputs"
os.makedirs(output_dir, exist_ok=True)

print("📈 Creating MSA heatmap...")
fig = plot_msa_representation_heatmap(
    msa_final[-1],
    layer_idx=-1,
    save_path=f"{output_dir}/demo_msa_final.png"
)
print("   ✅ MSA heatmap saved!")

print("📈 Creating Pair heatmap...")
fig = plot_pair_representation_heatmap(
    pair_final[-1],
    layer_idx=-1,
    save_path=f"{output_dir}/demo_pair_final.png"
)
print("   ✅ Pair heatmap saved!")

# Demo 3: Mock Layer-by-Layer Analysis
print("\n🔄 Demo 3: Mock Layer-by-Layer Analysis")
print("-" * 40)

print("📊 Creating mock layer data...")
msa_layers = {}
pair_layers = {}

for layer_idx in range(5):
    noise = torch.randn_like(msa_tensor) * 0.1
    msa_layers[layer_idx] = msa_tensor + noise
    
    noise = torch.randn_like(pair_tensor) * 0.1
    pair_layers[layer_idx] = pair_tensor + noise

print(f"✅ Created representations for {len(msa_layers)} layers")

print("📈 Creating evolution plot...")
fig = plot_representation_evolution(
    msa_layers,
    residue_idx=50,
    save_path=f"{output_dir}/demo_evolution.png",
    rep_type='msa'
)
print("   ✅ Evolution plot saved!")

# Demo 4: Channel Analysis
print("\n🔬 Demo 4: Channel-Specific Analysis")
print("-" * 40)

print("📈 Creating channel-specific heatmap...")
fig = plot_channel_specific_heatmap(
    msa_tensor,
    layer_idx=0,
    channel_idx=64,
    save_path=f"{output_dir}/demo_channel_64.png",
    rep_type='msa'
)
print("   ✅ Channel heatmap saved!")

# Demo 5: Save and Load
print("\n💾 Demo 5: Save and Load")
print("-" * 40)

demo_data = {
    'msa_layers': msa_layers,
    'pair_layers': pair_layers,
    'final_msa': msa_tensor,
    'final_pair': pair_tensor,
    'metadata': {
        'protein_name': 'DEMO_PROTEIN',
        'n_residues': n_res,
        'n_sequences': n_seq
    }
}

print("💾 Saving representations...")
save_intermediate_reps_to_disk(
    demo_data,
    output_dir,
    "demo_protein"
)
print("   ✅ Data saved!")

print("📂 Loading representations...")
loaded_data = load_intermediate_reps_from_disk(
    f"{output_dir}/demo_protein_intermediate_reps.pt"
)
print(f"   ✅ Data loaded! Keys: {list(loaded_data.keys())}")

# Demo 6: Aggregation Methods
print("\n⚙️  Demo 6: Different Aggregation Methods")
print("-" * 40)

methods = ['mean', 'max', 'norm', 'sum']
for method in methods:
    print(f"📈 Testing {method} aggregation...")
    result = aggregate_channels(msa_tensor, method=method)
    print(f"   ✅ {method}: shape {result.shape}, range [{result.min():.3f}, {result.max():.3f}]")

# Summary
print("\n" + "=" * 60)
print("🎉 Demo Complete! All features working!")
print("=" * 60)

print(f"\n📁 Generated files in '{output_dir}/':")
for fname in sorted(os.listdir(output_dir)):
    fpath = os.path.join(output_dir, fname)
    size = os.path.getsize(fpath)
    print(f"   📄 {fname} ({size:,} bytes)")

print(f"\n✨ Ready to use with real OpenFold data!")
print(f"   Just replace 'mock_output' with your actual model output")
print(f"   and follow the same patterns shown above.")

print("\n🚀 Next steps:")
print("   1. Try with real OpenFold model")
print("   2. Explore different proteins")
print("   3. Create custom visualizations")
print("   4. Integrate into your analysis pipeline")

print("\n📚 Documentation:")
print("   - QUICK_START_GUIDE.md")
print("   - INTERMEDIATE_REPS_README.md")
print("   - IMPLEMENTATION_SUMMARY.md")
