# OpenFold Intermediate Representation Visualization

A comprehensive toolkit for visualizing intermediate representations from OpenFold's 48-layer Evoformer network. Extract and analyze MSA, Pair, and Structure module outputs with advanced visualization techniques.

## Features

- **MSA Representations**: Heatmaps, evolution plots, multi-residue analysis
- **Pair Representations**: Contact map overlays, correlation analysis
- **Structure Module**: Backbone frames, angles, RMSD tracking
- **Multi-Layer Analysis**: Stratified sampling, convergence analysis, layer importance ranking
- **Comprehensive Reports**: PDF generation, clustering analysis

## Quick Start

```bash
# Install dependencies
pip install torch numpy matplotlib scipy

# Run demo
python demo_usage.py

# Interactive analysis
jupyter notebook OpenFold_Comprehensive_Analysis.ipynb
```

## Core Functions

### Basic Visualizations
- `plot_msa_representation_heatmap()` - MSA heatmaps with residue highlighting
- `plot_pair_representation_heatmap()` - Pair representations with contact overlays
- `plot_representation_evolution()` - Layer-by-layer evolution tracking

### Advanced Analysis
- `plot_multilayer_evolution()` - Evolution across all 48 layers
- `plot_stratified_layer_comparison()` - Side-by-side layer comparisons
- `plot_layer_convergence_analysis()` - Convergence metrics
- `plot_structure_module_evolution()` - Structure module analysis

### Utility Functions
- `stratified_layer_sampling()` - Smart layer selection strategies
- `compute_layer_importance_metrics()` - Layer importance ranking
- `create_comprehensive_visualization_report()` - Full analysis pipeline

## Usage Examples

```python
from visualize_intermediate_reps_utils import *

# Extract representations from OpenFold output
msa_reps = extract_msa_representations(model_output)
pair_reps = extract_pair_representations(model_output)

# Create visualizations
plot_msa_representation_heatmap(msa_reps[-1], -1, "msa_heatmap.png")
plot_multilayer_evolution(msa_reps, [10, 50, 100], "evolution.png")

# Comprehensive analysis
create_comprehensive_visualization_report(msa_reps, structure_outputs)
```

## HPC Setup (PACE ICE)

For running on PACE ICE cluster with real OpenFold data:

```bash
# Get GPU node
salloc -p ice-gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G -t 02:00:00

# Clone and setup
git clone https://github.com/vizfold/attention-viz-demo.git ~/scratch/attention-viz-demo
cd ~/scratch/attention-viz-demo

# Install OpenFold and dependencies
module load mamba
mamba env create -p ~/scratch/envs/openfold_env -f environment.yml
conda activate ~/scratch/envs/openfold_env

# Start Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

## Files

- `visualize_intermediate_reps_utils.py` - Core visualization functions
- `demo_usage.py` - Basic demo script
- `OpenFold_Comprehensive_Analysis.ipynb` - Interactive Jupyter notebook
- `requirements_visualization.txt` - Minimal dependencies

## License

Apache License 2.0 - See [LICENSE](./LICENSE) for details.

## Acknowledgments

Based on [OpenFold](https://github.com/aqlaboratory/openfold) - an open-source AlphaFold implementation.