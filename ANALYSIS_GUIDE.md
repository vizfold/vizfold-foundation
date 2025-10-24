# OpenFold Comprehensive 48-Layer Analysis Guide

## Overview

This guide provides instructions for running a comprehensive analysis of OpenFold's 48 Evoformer layers, including intermediate representation visualization, convergence analysis, and structure module evaluation.

---

## Configuration

### Analysis Parameters

The notebook is configured for detailed protein analysis with the following parameters:

```python
n_seq = 32          # MSA sequences
n_res = 256         # Residues
n_layers = 48       # Evoformer layers
n_recycles = 12     # Structure module recycles
```

### Layer Sampling Strategies

Multiple sampling strategies are available for different analysis needs:

| Strategy | Coverage | Layers | Use Case |
|----------|----------|--------|----------|
| `uniform` | 19% (9) | Evenly spaced | Quick exploratory analysis |
| `grouped` | 27% (13) | Early/mid/late groups | Standard workflow |
| `random` | 70% (33) | Randomized selection | Unbiased sampling |
| `dense` | 75% (36) | Dense uniform | High-detail analysis |
| `all` | 100% (48) | Complete coverage | Comprehensive analysis |

---

## Running the Analysis

### Step 1: Environment Setup

Ensure you have the required packages:
```bash
conda activate openfold-env
```

Or install minimal requirements:
```bash
pip install -r requirements_visualization.txt
```

### Step 2: Launch Notebook

```bash
jupyter notebook OpenFold_Comprehensive_Analysis.ipynb
```

### Step 3: Execute Analysis

In Jupyter:
1. Select the appropriate kernel (openfold-env or base)
2. **Kernel → Restart & Clear Output**
3. **Cell → Run All**

### Expected Runtime

- **CPU**: 15-20 minutes
- **GPU**: 5-8 minutes

---

## Generated Visualizations

### 1. Multi-Layer Evolution
- **File**: `multilayer_evolution.png`
- **Description**: Tracks 15 residues across all 48 layers
- **Shows**: Magnitude evolution and layer-to-layer changes

### 2. Stratified MSA Comparison
- **File**: `stratified_msa.png`
- **Description**: Grid visualization of MSA representations
- **Shows**: Layer-by-layer transformation of MSA features

### 3. Stratified Pair Comparison
- **File**: `stratified_pair.png`
- **Description**: Complete pair representation analysis
- **Shows**: 256×256 residue interactions across layers

### 4. MSA Convergence Analysis
- **File**: `msa_convergence.png`
- **Description**: Correlation matrix showing layer similarity
- **Shows**: Convergence patterns and stability points

### 5. Pair Convergence Analysis
- **File**: `pair_convergence.png`
- **Description**: Pair representation stability tracking
- **Shows**: Rate of change across network depth

### 6. Layer Importance Ranking
- **File**: `layer_importance.png`
- **Description**: Multi-metric layer importance assessment
- **Shows**: Variance, entropy, and norm-based rankings

### 7. Structure Module Evolution
- **File**: `structure_evolution.png`
- **Description**: 6-panel structure refinement analysis
- **Shows**: Frames, angles, RMSD, trajectory, displacement, changes

### 8. Residue Feature Analysis
- **File**: `residue_50_analysis.png`
- **Description**: Detailed single-residue feature examination
- **Shows**: Heatmap, distribution, statistics, correlations, activations

### 9. Hierarchical Clustering
- **File**: `layer_clustering.png`
- **Description**: Layer similarity dendrogram
- **Shows**: Natural groupings and layer relationships

### 10. Contact Map Integration
- **File**: `pair_with_contacts.png`
- **Description**: Contact prediction validation
- **Shows**: Predicted contacts overlaid on pair representations

---

## Customization

### Adjusting Protein Size

Edit Cell 4 in the notebook:

```python
n_seq = 16      # Reduce for faster processing
n_res = 128     # Adjust based on protein size
```

### Changing Layer Sampling

Edit Cell 8:

```python
# Use different sampling strategy
sampled_layers = stratified_layer_sampling(48, strategy='dense')  # 75% coverage
sampled_layers = stratified_layer_sampling(48, strategy='grouped')  # 27% coverage
```

### Selecting Tracked Residues

Edit Cell 6:

```python
# Customize which residues to track
tracked_residues = [0, 50, 100, 150, 200, 255]  # Specific positions
# Or use automatic spacing
tracked_residues = list(range(0, n_res, 20))  # Every 20th residue
```

---

## Performance Considerations

### Memory Requirements

- **Minimal (n_res=100)**: ~500 MB RAM
- **Standard (n_res=128)**: ~2 GB RAM
- **Full (n_res=256)**: ~8 GB RAM

### Optimization Tips

1. **Reduce Residues**: Lower `n_res` for faster processing
2. **Sample Layers**: Use `'dense'` or `'grouped'` instead of `'all'`
3. **Limit Tracked Residues**: Track fewer residues in evolution plots
4. **Use GPU**: Enable CUDA if available for 3x speedup

---

## Output Files

All visualizations are saved to `notebook_outputs/`:

```
notebook_outputs/
├── multilayer_evolution.png          (~5 MB)
├── stratified_msa.png                (~15 MB)
├── stratified_pair.png               (~25 MB)
├── msa_convergence.png               (~3 MB)
├── pair_convergence.png              (~3 MB)
├── layer_importance.png              (~2 MB)
├── structure_evolution.png           (~8 MB)
├── residue_50_analysis.png           (~4 MB)
├── layer_clustering.png              (~2 MB)
└── pair_with_contacts.png            (~10 MB)
```

**Total**: ~77 MB (high-resolution, publication-quality)

---

## Integration with Real OpenFold

To use with actual OpenFold inference:

```python
import openfold
from visualize_intermediate_reps_utils import *

# Load model
model = openfold.load_model(...)

# Register hooks
hooks = register_evoformer_hooks(model)

# Run inference
output = model(batch)

# Extract representations
msa_reps = extract_msa_representations()
pair_reps = extract_pair_representations()
structure_output = extract_structure_representations()

# Visualize using all functions from the notebook
plot_multilayer_evolution(msa_reps, ...)
plot_stratified_layer_comparison(pair_reps, ...)
# ... etc
```

---

## Troubleshooting

### Out of Memory

**Solution**: Reduce parameters
```python
n_res = 128
n_seq = 16
sampled_layers = stratified_layer_sampling(48, strategy='grouped')
```

### Slow Performance

**Solution**: Use sampling
```python
layer_sampling = 'dense'  # Instead of 'all'
tracked_residues = [10, 50, 100, 150, 200]  # Fewer residues
```

### Module Import Errors

**Solution**: Ensure module reload is enabled (already in Cell 2)
```python
import importlib
import visualize_intermediate_reps_utils
importlib.reload(visualize_intermediate_reps_utils)
```

### Figures Not Displaying

**Solution**: Check matplotlib backend
```python
%matplotlib inline
import matplotlib.pyplot as plt
```

---

## Technical Details

### Grid Layouts

Layer count determines grid configuration:
- **≤9 layers**: 3 columns
- **10-25 layers**: 5 columns  
- **26-48 layers**: 8 columns

### Figure Sizing

Automatically adapts based on content:
- **Evolution plots**: Scale with residue count
- **Grid plots**: Scale with layer count
- **Resolution**: 300 DPI (publication quality)

### Color Schemes

- **MSA/Pair heatmaps**: viridis, RdBu
- **Evolution tracking**: tab10 (≤10 residues), tab20 (>10 residues)
- **Clustering**: viridis
- **Importance**: plasma

---

## Next Steps

1. **Run Analysis**: Execute the notebook with current settings
2. **Review Outputs**: Examine generated visualizations
3. **Customize**: Adjust parameters for your specific needs
4. **Integrate**: Apply to real OpenFold inference
5. **Publish**: Use high-resolution outputs in presentations/papers

---

## Additional Resources

- **Function Reference**: See `COMPREHENSIVE_ANALYSIS_GUIDE.md`
- **Environment Setup**: See `NOTEBOOK_SETUP.md`
- **Feature Details**: See `MAXED_OUT_FEATURES.md` (technical specifications)

---

## Support

For questions or issues:
1. Check inline documentation in the notebook
2. Review function docstrings in `visualize_intermediate_reps_utils.py`
3. Consult the comprehensive guides listed above

---

**Ready to analyze OpenFold's 48 layers? Start by opening the notebook and running all cells!**

