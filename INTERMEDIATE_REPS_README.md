# Intermediate Representation Visualization for OpenFold

## Overview

This module provides comprehensive tools for extracting and visualizing intermediate representations from OpenFold's 48-layer deep neural network. It enables researchers to examine how protein representations evolve through the network layers.

## Features

### ✓ Extraction Functions
- **MSA Representations**: Extract Multiple Sequence Alignment representations from any layer
- **Pair Representations**: Extract pairwise residue interaction representations
- **Structure Module Outputs**: Extract backbone frames, torsion angles, and atomic positions

### ✓ Visualization Functions
- **MSA Heatmaps**: Visualize MSA representations as sequence vs. residue heatmaps
- **Pair Heatmaps**: Contact-map-like visualizations of pairwise representations
- **Evolution Plots**: Track how specific residues change across layers
- **Channel-Specific Views**: Examine individual feature channels
- **Multiple Aggregation Methods**: mean, max, norm, sum

### ✓ Utilities
- **Save/Load**: Persist representations to disk for later analysis
- **Hook Registration**: Automatically capture layer-by-layer outputs
- **Flexible API**: Works with model outputs or global storage

## Installation

Dependencies are already installed:
```bash
# torch, numpy, matplotlib
```

## Quick Start

### 1. Extract Final Representations (Simplest)

```python
from visualize_intermediate_reps_utils import *

# Run OpenFold model
output = model(batch)

# Extract final representations
msa_reps = extract_msa_representations(output)
pair_reps = extract_pair_representations(output)
structure_reps = extract_structure_representations(output)

# Visualize
plot_msa_representation_heatmap(
    msa_reps[-1], 
    layer_idx=-1, 
    save_path='outputs/msa_final.png'
)
```

### 2. Extract Layer-by-Layer Representations

```python
from visualize_intermediate_reps_utils import *

# Enable intermediate storage
INTERMEDIATE_REPS.enable()

# Register hooks to capture each layer
hooks = register_evoformer_hooks(model)

# Run model
output = model(batch)

# Extract all layers
msa_all_layers = extract_msa_representations(None)
pair_all_layers = extract_pair_representations(None)

# Visualize specific layers
for layer_idx in [0, 12, 24, 47]:
    plot_msa_representation_heatmap(
        msa_all_layers[layer_idx],
        layer_idx=layer_idx,
        save_path=f'outputs/msa_layer_{layer_idx}.png'
    )

# Track evolution
plot_representation_evolution(
    msa_all_layers,
    residue_idx=25,
    save_path='outputs/msa_evolution_res25.png',
    rep_type='msa'
)

# Cleanup
remove_hooks(hooks)
INTERMEDIATE_REPS.disable()
```

### 3. Channel-Specific Analysis

```python
# Visualize specific channels
plot_channel_specific_heatmap(
    msa_all_layers[12],
    layer_idx=12,
    channel_idx=64,
    save_path='outputs/msa_layer12_channel64.png',
    rep_type='msa'
)
```

### 4. Save and Load for Later

```python
# Save representations
intermediate_data = {
    'msa': msa_all_layers,
    'pair': pair_all_layers,
    'structure': structure_reps
}
save_intermediate_reps_to_disk(
    intermediate_data,
    output_dir='saved_reps',
    protein_name='6KWC'
)

# Load later
loaded_reps = load_intermediate_reps_from_disk('saved_reps/6KWC_intermediate_reps.pt')
```

## API Reference

### Extraction Functions

#### `extract_msa_representations(model_output, layer_indices=None)`
Extract MSA representations from model output or global storage.

**Parameters:**
- `model_output` (Dict): Model output dictionary or None
- `layer_indices` (List[int]): Specific layers to extract, or None for all

**Returns:** Dict[int, torch.Tensor] mapping layer index to MSA tensor (n_seq, n_res, c_m)

#### `extract_pair_representations(model_output, layer_indices=None)`
Extract Pair representations from model output or global storage.

**Returns:** Dict[int, torch.Tensor] mapping layer index to Pair tensor (n_res, n_res, c_z)

#### `extract_structure_representations(model_output)`
Extract Structure module outputs.

**Returns:** Dict with keys: 'backbone_frames', 'angles', 'positions', 'final_atom_positions'

### Visualization Functions

#### `plot_msa_representation_heatmap(msa_tensor, layer_idx, save_path, aggregate_method='mean', cmap='viridis')`
Create MSA heatmap visualization.

**Parameters:**
- `msa_tensor` (torch.Tensor): Shape (n_seq, n_res, c_m)
- `layer_idx` (int): Layer number for title
- `save_path` (str): Output file path
- `aggregate_method` (str): 'mean', 'max', 'norm', or 'sum'
- `cmap` (str): Matplotlib colormap name

#### `plot_pair_representation_heatmap(pair_tensor, layer_idx, save_path, aggregate_method='mean', cmap='RdBu_r')`
Create Pair contact-map-like heatmap.

**Parameters:**
- `pair_tensor` (torch.Tensor): Shape (n_res, n_res, c_z)
- Other parameters same as MSA heatmap

#### `plot_representation_evolution(tensors_across_layers, residue_idx, save_path, rep_type='msa')`
Plot how a residue's representation changes across layers.

**Parameters:**
- `tensors_across_layers` (Dict[int, torch.Tensor]): Layer index to tensor mapping
- `residue_idx` (int): Which residue to track
- `rep_type` (str): 'msa' or 'pair'

#### `plot_channel_specific_heatmap(tensor, layer_idx, channel_idx, save_path, rep_type='msa', cmap='viridis')`
Visualize a single feature channel.

### Utility Functions

#### `aggregate_channels(tensor, method='mean', axis=-1)`
Aggregate tensor over channel dimension.

#### `save_intermediate_reps_to_disk(intermediate_reps, output_dir, protein_name)`
Save representations to disk as PyTorch file.

#### `load_intermediate_reps_from_disk(input_path)`
Load previously saved representations.

#### `register_evoformer_hooks(model)`
Register forward hooks to capture layer outputs. Returns list of hooks.

#### `remove_hooks(hooks)`
Remove registered hooks.

### Global Storage

```python
# Enable/disable collection
INTERMEDIATE_REPS.enable()
INTERMEDIATE_REPS.disable()

# Clear stored data
INTERMEDIATE_REPS.clear()

# Access directly
msa_layer_10 = INTERMEDIATE_REPS.msa_reps[10]
```

## Testing

Run comprehensive tests:

```bash
python3 test_visualizations.py
```

This creates mock data and tests all visualization functions, generating sample outputs.

## Integration with Existing Code

This module follows the same patterns as existing OpenFold visualization tools:
- Similar to `visualize_attention_general_utils.py` for attention maps
- Uses global storage like `ATTENTION_METADATA`
- Compatible with existing demo notebooks

## Next Steps

1. **Unit Tests**: Add comprehensive pytest tests in `tests/test_intermediate_extraction.py`
2. **Demo Notebook**: Create Jupyter notebook demonstrating all features
3. **Web Interface**: Integrate visualization functions into web interface
4. **Performance**: Optimize memory usage for large proteins
5. **Documentation**: Add to main OpenFold documentation

## File Structure

```
attention-jay-venn/
├── visualize_intermediate_reps_utils.py   # Main implementation (643 lines)
├── test_visualizations.py                  # Comprehensive test script
├── INTERMEDIATE_REPS_README.md            # This file
└── tests/
    └── test_intermediate_extraction.py     # Unit tests (to be completed)
```

## Examples Gallery

After running `test_visualizations.py`, you'll find examples of:
- MSA heatmaps with different aggregation methods
- Pair representation contact maps
- Evolution plots showing layer-by-layer changes
- Channel-specific visualizations
- Saved/loaded representation files

## Contributing

When adding new features:
1. Follow the existing function signature patterns
2. Include comprehensive docstrings
3. Add tests to `test_visualizations.py`
4. Update this README

## License

Same as OpenFold - Apache License 2.0

## Author

Developed for Issue #8: Intermediate Representation Visualization

## Version

v1.0 - Full implementation complete (October 2025)

