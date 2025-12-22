# Heatmap Visualization

The heatmap visualization provides a complementary view to arc diagrams and PyMOL overlays by showing all attention heads simultaneously in a grid format. This enables cross-head comparison and spatial pattern analysis that reveals insights not visible in individual head visualizations.

## Overview

### Key Features

- **Cross-head comparison**: Compare attention patterns across all heads in a single view
- **Spatial pattern analysis**: Identify hotspots and structural patterns in attention matrices
- **Global normalization**: Consistent color scaling across heads for fair comparison
- **Per-head normalization**: Alternative mode preserves individual head characteristics
- **Multiple colormaps**: Support for viridis, plasma, inferno, and more
- **High-resolution output**: 300 DPI PNG files suitable for publication

### Attention Types

The heatmap visualization supports two main attention types:

1. **MSA Row Attention**: Shows pairwise attention between residues as inferred from multiple sequence alignments
2. **Triangle Start Attention**: Focuses on attention from a single residue to others, as part of triangle-based geometric reasoning

## Usage

### Primary Method: Python Functions (Recommended)

Use the visualization functions directly in Python, same as arc diagrams and PyMOL. This is the recommended approach for notebook-based workflows and interactive analysis:

```python
from visualize_attention_heatmap_utils import plot_all_heads_heatmap, plot_combined_attention_heatmap

# Generate MSA Row heatmap
output_path = plot_all_heads_heatmap(
    attention_dir='./outputs/attention_files_6KWC_demo_tri_18',
    output_dir='./outputs/heatmaps',
    protein='6KWC',
    attention_type='msa_row',
    layer_idx=47,
    fasta_path='./examples/monomer/fasta_dir_6KWC/6KWC.fasta',  # Auto-detect seq_length
    normalization='global',
    colormap='viridis',
    figsize_per_head=(2.0, 2.0),
    dpi=300,
    save_to_png=True,
    residue_indices=None  # Not needed for msa_row
)

# Generate Triangle Start heatmap
output_path = plot_all_heads_heatmap(
    attention_dir='./outputs/attention_files_6KWC_demo_tri_18',
    output_dir='./outputs/heatmaps',
    protein='6KWC',
    attention_type='triangle_start',
    layer_idx=47,
    fasta_path='./examples/monomer/fasta_dir_6KWC/6KWC.fasta',  # Auto-detect seq_length
    normalization='global',
    colormap='viridis',
    figsize_per_head=(2.0, 2.0),
    dpi=300,
    save_to_png=True,
    residue_indices=[18, 39, 51]  # Required for triangle_start
)

# Generate combined heatmap
output_path = plot_combined_attention_heatmap(
    attention_dir='./outputs/attention_files_6KWC_demo_tri_18',
    output_dir='./outputs/heatmaps',
    protein='6KWC',
    layer_idx=47,
    fasta_path='./examples/monomer/fasta_dir_6KWC/6KWC.fasta',  # Auto-detect seq_length
    normalization='global',
    colormap='viridis',
    figsize_per_head=(1.5, 1.5),
    dpi=300,
    save_to_png=True,
    residue_indices=[18, 39]  # For triangle_start component
)
```

### Optional Method: Command Line Interface

For batch processing or shell-based workflows, use the CLI script:

```bash
# MSA Row attention heatmap
python scripts/generate_attention_heatmaps.py \
    --attention_dir ./outputs/attention_files_6KWC_demo_tri_18 \
    --output_dir ./outputs/heatmaps \
    --protein 6KWC \
    --layer 47 \
    --attention_type msa_row

# Triangle Start attention heatmap
python scripts/generate_attention_heatmaps.py \
    --attention_dir ./outputs/attention_files_6KWC_demo_tri_18 \
    --output_dir ./outputs/heatmaps \
    --protein 6KWC \
    --layer 47 \
    --attention_type triangle_start \
    --residue_indices 18 39 51

# Combined heatmap (both attention types)
python scripts/generate_attention_heatmaps.py \
    --attention_dir ./outputs/attention_files_6KWC_demo_tri_18 \
    --output_dir ./outputs/heatmaps \
    --protein 6KWC \
    --layer 47 \
    --attention_type combined \
    --residue_indices 18 39
```

## Parameters

### Required Parameters

- `attention_dir`: Directory containing attention text files
- `output_dir`: Directory to save output PNG files
- `protein`: Protein identifier (e.g., "6KWC")
- `attention_type`: "msa_row", "triangle_start", or "combined"
- `residue_indices`: List of residue indices for triangle_start attention (required for triangle_start and combined)

### Optional Parameters

- `layer`: Layer number to visualize (default: 47)
- `seq_length`: Sequence length (auto-detect if None)
- `fasta_path`: Path to FASTA file for sequence length detection
- `normalization`: "global" or "per_head" normalization (default: "global")
- `colormap`: Matplotlib colormap name (default: "viridis")
- `figsize_per_head`: Size of each subplot as (width, height) (default: (2.0, 2.0))
- `dpi`: Output resolution (default: 300)
- `verbose`: Enable verbose output

## Output Format

### File Naming Convention

- MSA Row: `msa_row_heatmap_layer_{layer}_{protein}.png`
- Triangle Start: `triangle_start_heatmap_layer_{layer}_{protein}.png`
- Combined: `combined_attention_heatmap_layer_{layer}_{protein}.png`

### Grid Layouts

- **MSA Row**: 2×4 grid (8 heads)
- **Triangle Start**: 1×4 grid (4 heads)
- **Combined**: 3×4 grid (8 MSA Row + 4 Triangle Start heads)

## Normalization Methods

### Global Normalization

All heads use the same color scale based on the global minimum and maximum values across all heads. This enables fair comparison between heads but may compress the dynamic range of individual heads.

### Per-Head Normalization

Each head is normalized independently to its own minimum and maximum values. This preserves the full dynamic range of each head but makes cross-head comparison more difficult.

## Scientific Applications

### Cross-Head Analysis

Heatmaps reveal how different attention heads specialize:

- **Self-attention patterns**: Diagonal elements show how much each residue attends to itself
- **Cross-attention patterns**: Off-diagonal elements show inter-residue attention
- **Head specialization**: Different heads may focus on different structural or functional aspects

### Spatial Pattern Recognition

Heatmaps enable identification of:

- **Hotspots**: Regions of high attention activity
- **Structural motifs**: Patterns corresponding to secondary structures
- **Functional sites**: Attention patterns around active sites or binding regions

### Comparative Analysis

Compare attention patterns across:

- **Different layers**: How attention evolves through the network
- **Different proteins**: Conservation of attention patterns
- **Different conditions**: Effects of mutations or modifications

## Integration with Existing Visualizations

Heatmaps complement existing visualization methods:

- **Arc diagrams**: Show sequence-level attention patterns
- **PyMOL overlays**: Show 3D structural context
- **Heatmaps**: Show spatial patterns and cross-head comparison

Use all three visualization types together for comprehensive analysis of OpenFold's attention mechanisms.

## Troubleshooting

### Common Issues

1. **File not found**: Ensure attention files exist in the specified directory
2. **Sequence length mismatch**: Provide correct sequence length or FASTA file
3. **Memory issues**: Reduce `figsize_per_head` or `dpi` for large proteins
4. **Title overlap**: Adjust `figsize_per_head` or use different normalization

### Performance Tips

- Use `global` normalization for cross-head comparison
- Use `per_head` normalization for individual head analysis
- Reduce `dpi` for faster generation during development
- Use `combined` type to generate all visualizations at once

## Examples

See the [README](../README.md) for example outputs and usage demonstrations.
