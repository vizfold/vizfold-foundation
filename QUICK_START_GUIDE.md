# Quick Start Guide: Intermediate Representation Visualization

## Implementation Complete!

All core functionality for visualizing intermediate representations from OpenFold's 48-layer network is now ready to use.

## What's Available

**Extraction Functions (3)**
- `extract_msa_representations()` - Get MSA representations from any layer
- `extract_pair_representations()` - Get pair representations  
- `extract_structure_representations()` - Get structure module outputs

**Visualization Functions (4)**
- `plot_msa_representation_heatmap()` - MSA heatmaps with flexible aggregation
- `plot_pair_representation_heatmap()` - Contact-map-like pair visualizations
- `plot_representation_evolution()` - Track changes across layers
- `plot_channel_specific_heatmap()` - Individual channel analysis

**Utility Functions (5)**
- `aggregate_channels()` - Flexible channel aggregation
- `save_intermediate_reps_to_disk()` - Save for later analysis
- `load_intermediate_reps_from_disk()` - Load saved data
- `register_evoformer_hooks()` - Automatic layer capture
- `remove_hooks()` - Clean up

---

## 🚀 5-Minute Tutorial

### Step 1: Import and Run Model

```python
from visualize_intermediate_reps_utils import *

# Your existing OpenFold code
output = model(batch)
```

### Step 2: Extract Final Representations

```python
# Get final layer outputs
msa_final = extract_msa_representations(output)
pair_final = extract_pair_representations(output)
structure = extract_structure_representations(output)
```

### Step 3: Create Your First Visualization

```python
# MSA heatmap
plot_msa_representation_heatmap(
    msa_final[-1],
    layer_idx=-1,
    save_path='outputs/msa_final.png'
)

# Pair contact map
plot_pair_representation_heatmap(
    pair_final[-1],
    layer_idx=-1,
    save_path='outputs/pair_final.png'
)
```

**Done!** You now have publication-quality visualizations.

---

## 🔬 Advanced: Layer-by-Layer Analysis

Want to see how representations evolve through all 48 layers?

```python
# Enable intermediate capture
INTERMEDIATE_REPS.enable()
hooks = register_evoformer_hooks(model)

# Run model (hooks automatically capture each layer)
output = model(batch)

# Extract all layers
msa_all = extract_msa_representations(None)  # Returns dict: {0: tensor, 1: tensor, ...}
pair_all = extract_pair_representations(None)

# Visualize specific layers
for layer in [0, 12, 24, 36, 47]:
    plot_msa_representation_heatmap(
        msa_all[layer],
        layer_idx=layer,
        save_path=f'outputs/msa_layer_{layer}.png'
    )

# Track a specific residue across layers
plot_representation_evolution(
    msa_all,
    residue_idx=50,
    save_path='outputs/residue_50_evolution.png',
    rep_type='msa'
)

# Cleanup when done
remove_hooks(hooks)
INTERMEDIATE_REPS.disable()
```

---

## 📊 Visualization Options

### Aggregation Methods

```python
# Try different aggregation methods
for method in ['mean', 'max', 'norm', 'sum']:
    plot_msa_representation_heatmap(
        msa_tensor,
        layer_idx=0,
        save_path=f'msa_{method}.png',
        aggregate_method=method
    )
```

### Colormaps

```python
# Different colormaps for different insights
plot_pair_representation_heatmap(
    pair_tensor,
    layer_idx=0,
    save_path='pair_diverging.png',
    cmap='RdBu_r'  # Diverging colormap (good for seeing +/- patterns)
)

plot_pair_representation_heatmap(
    pair_tensor,
    layer_idx=0,
    save_path='pair_sequential.png',
    cmap='viridis'  # Sequential colormap (good for magnitude)
)
```

### Channel-Specific Analysis

```python
# Examine individual feature channels
plot_channel_specific_heatmap(
    msa_tensor,
    layer_idx=12,
    channel_idx=64,
    save_path='msa_layer12_channel64.png',
    rep_type='msa'
)
```

---

## 💾 Save and Load

Working with large proteins? Save representations for later:

```python
# After running model with hooks
intermediate_data = {
    'msa': msa_all,
    'pair': pair_all,
    'structure': structure,
    'metadata': {
        'protein_name': '6KWC',
        'n_residues': 100,
        'timestamp': '2025-10-01'
    }
}

# Save (~5-10 GB for typical protein with all 48 layers)
save_intermediate_reps_to_disk(
    intermediate_data,
    output_dir='saved_reps',
    protein_name='6KWC'
)

# Load later (doesn't require model)
loaded = load_intermediate_reps_from_disk('saved_reps/6KWC_intermediate_reps.pt')
msa_all = loaded['msa']
```

---

## 📁 Files Created

```
visualize_intermediate_reps_utils.py   (642 lines) - Main implementation
test_visualizations.py                 (223 lines) - Comprehensive tests  
INTERMEDIATE_REPS_README.md           (275 lines) - Full documentation
IMPLEMENTATION_SUMMARY.md             (367 lines) - Technical details
QUICK_START_GUIDE.md                   (this file) - Getting started
```

---

## ✅ Verification

Run the test suite to verify everything works:

```bash
python3 test_visualizations.py
```

This creates mock data and tests all functions, generating 11 sample visualizations.

**Expected output:**
```
✓ All visualization tests completed!
Generated files:
  - msa_layer0_mean.png
  - msa_layer0_max.png
  - pair_layer0_RdBu_r.png
  - msa_evolution_res10.png
  - pair_evolution_res10.png
  ... and 6 more
```

---

## 🎯 Common Use Cases

### 1. Compare Representations Across Layers

```python
# See how layer 0 vs layer 47 differ
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for idx, layer in enumerate([0, 47]):
    msa_2d = aggregate_channels(msa_all[layer])
    axes[idx].imshow(msa_2d, cmap='viridis')
    axes[idx].set_title(f'Layer {layer}')
plt.savefig('layer_comparison.png')
```

### 2. Find Most Important Layers

```python
# Track representation magnitude across layers
magnitudes = []
for layer_idx in range(48):
    msa = msa_all[layer_idx]
    mag = torch.norm(msa).item()
    magnitudes.append(mag)

# Plot
plt.plot(range(48), magnitudes)
plt.xlabel('Layer')
plt.ylabel('Total Representation Magnitude')
plt.savefig('layer_importance.png')
```

### 3. Residue-Specific Analysis

```python
# Compare multiple residues
residues_of_interest = [10, 25, 50, 75]
for res_idx in residues_of_interest:
    plot_representation_evolution(
        msa_all,
        residue_idx=res_idx,
        save_path=f'residue_{res_idx}_evolution.png'
    )
```

---

## 🐛 Troubleshooting

### "No MSA representations found"
**Solution:** Enable intermediate storage and register hooks before running model:
```python
INTERMEDIATE_REPS.enable()
hooks = register_evoformer_hooks(model)
output = model(batch)
```

### Memory Issues
**Solution:** Don't store all 48 layers. Extract specific layers:
```python
# After running with hooks
important_layers = [0, 12, 24, 47]
msa_subset = {i: msa_all[i] for i in important_layers}
```

### Slow Visualization
**Solution:** Use lower DPI or smaller figure sizes:
```python
# In any plot function, edit the figure size
fig, ax = plt.subplots(figsize=(6, 4))  # Smaller figure
plt.savefig(path, dpi=150)  # Lower DPI (default is 300)
```

---

## 📚 Next Steps

1. **Read Full Documentation**: See `INTERMEDIATE_REPS_README.md`
2. **Run Tests**: Execute `python3 test_visualizations.py`
3. **Try With Real Data**: Use with your OpenFold model
4. **Create Notebook**: Build a Jupyter notebook for interactive analysis
5. **Integrate Web UI**: Add to web interface for point-and-click visualization

---

## 🎓 Understanding the Output

### MSA Heatmaps
- **X-axis**: Residue position in protein
- **Y-axis**: Sequence in MSA
- **Color**: Aggregated activation (brighter = stronger)
- **Interpretation**: Shows which sequences/residues are most active

### Pair Heatmaps
- **X/Y-axes**: Residue positions
- **Color**: Interaction strength between residues
- **Diagonal**: Self-interactions
- **Off-diagonal patterns**: Residue-residue relationships

### Evolution Plots
- **X-axis**: Layer number (0-47)
- **Y-axis**: Representation magnitude
- **Interpretation**: How representation changes through processing

---

## 💡 Tips

1. **Start Simple**: Begin with final layer visualizations before exploring all layers
2. **Use Aggregation Wisely**: `mean` for general view, `norm` for magnitude, `max` for strongest features
3. **Memory Management**: Clear storage after each protein: `INTERMEDIATE_REPS.clear()`
4. **Batch Processing**: Save all representations to disk, then create visualizations offline
5. **Publication Quality**: All outputs are 300 DPI and suitable for papers

---

## 🤝 Contributing

Found a bug or want to add features? 

1. Check `IMPLEMENTATION_SUMMARY.md` for architecture details
2. Add tests to `test_visualizations.py`
3. Update documentation
4. Follow existing code style

---

## 📞 Questions?

- **Documentation**: `INTERMEDIATE_REPS_README.md` (complete API reference)
- **Technical Details**: `IMPLEMENTATION_SUMMARY.md` (implementation notes)
- **Code**: `visualize_intermediate_reps_utils.py` (all functions have docstrings)

---

## 🎉 You're Ready!

You now have everything you need to visualize and analyze OpenFold's intermediate representations. Happy exploring! 🚀

**Status**: ✅ All core functionality implemented and tested
**Version**: 1.0
**Last Updated**: October 2025

