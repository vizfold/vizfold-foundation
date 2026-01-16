# OpenFold-based Attention Visualization Demo

This is a lightweight extension of [OpenFold](https://github.com/aqlaboratory/openfold) that enables interactive visualization of attention mechanisms in protein structure prediction. It provides tools to render MSA row and Triangle attention scores as:

- Arc diagrams (sequence space)
- 3D PyMOL overlays (structure space)
- Heatmap grids (all heads at once)

---

## Key Features

- Compatible with OpenFold outputs (`.pdb`, attention text dumps)
- Support for layer- and head-specific visualizations
- Integrated residue highlighting
- Cross-head comparison with heatmap grids
- Notebook-friendly and HPC-friendly workflow

---

## Installation

This repo assumes you have already installed [OpenFold and its dependencies](https://openfold.readthedocs.io/en/latest/Installation.html), or you are using CyberShuttle (see `cybershuttle.yml`)
You will also need:
- `PyMOL` (open-source version is sufficient)
- `matplotlib`, `numpy`, `scipy`, `pandas`
- `biopython` (for sequence parsing)

You can also install the full set of dendencies (including those for OpenFold) from our `cybershuttle.yml` file directly.
Beyond confirming the proper installation of OpenFold, you can test the specific dependendices for our repo by using:
```
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from pymol import cmd
from pymol.cgo import CYLINDER, SPHERE
```

---

## Interactive Demo: `viz_attention_demo_base.ipynb`

The notebook `viz_attention_demo_base.ipynb` demonstrates the full visualization pipeline using OpenFold.

It performs the following steps:

1. **Runs inference** using OpenFold with precomputed alignments
2. **Extracts top-k residue–residue attention scores** from each layer and head
3. **Saves these scores** to text files
4. **Visualizes attention**:
   - As **arc diagrams** (residue–residue attention on the sequence)
   - As **3D PyMOL overlays** (on the predicted structure)
   - As **heatmap grids** (all heads at once for cross-head comparison)

We focus on two attention types:
- **MSA Row Attention**
- **Triangle Start Attention**

> The **thickness of the lines** (in both arc diagrams and 3D renderings) indicates the **strength of the attention score**.

(If using Cybershuttle, then please use `viz_attention_demo.ipynb`)

---

### MSA Row Attention (Layer 47, Protein 6KWC)

- Shows pairwise attention between residues as inferred from multiple sequence alignments
- Visualized across all attention heads at a selected model layer

Arc diagram (Head 2):

![msa_row_arc](./outputs/attention_images_6KWC_demo_tri_18/msa_row_attention_plots/msa_row_head_2_layer_47_6KWC_arc.png)

All heads subplot:

![msa_row_subplot](./outputs/attention_images_6KWC_demo_tri_18/msa_row_attention_plots/msa_row_heads_layer_47_6KWC_subplot.png)

---

### Triangle Start Attention (Layer 47, Residue 18)

- Focuses on attention **from a single residue to others**, as part of triangle-based geometric reasoning
- The **selected residue is highlighted**:
  - In arc diagrams: using a **blue label**
  - In 3D visualizations: using a **sphere**

Arc diagram (Head 0):

![triangle_start_arc](./outputs/attention_images_6KWC_demo_tri_18/tri_start_attention_plots/tri_start_res_18_head_0_layer_47_6KWC_arc.png)

All heads subplot:

![triangle_start_subplot](./outputs/attention_images_6KWC_demo_tri_18/tri_start_attention_plots/triangle_start_residue_18_layer_47_6KWC_subplot.png)

---

## Heatmap Visualization

### New Feature: All Heads at Once

The heatmap visualization provides a complementary view to arc diagrams and PyMOL overlays by showing all attention heads simultaneously in a grid format. This enables:

- **Cross-head comparison**: Compare attention patterns across all heads in a single view
- **Spatial pattern analysis**: Identify hotspots and structural patterns in attention matrices
- **Global normalization**: Consistent color scaling across heads for fair comparison

### Usage

#### Primary Method: Python Functions

Use the visualization functions directly in Python, same as arc diagrams and PyMOL:

```python
from visualize_attention_heatmap_utils import plot_all_heads_heatmap, plot_combined_attention_heatmap

# Generate MSA Row attention heatmap
plot_all_heads_heatmap(
    attention_dir='./outputs/attention_files_6KWC_demo_tri_18',
    output_dir='./outputs/heatmaps',
    protein='6KWC',
    attention_type='msa_row',
    layer_idx=47,
    fasta_path='./examples/monomer/fasta_dir_6KWC/6KWC.fasta',
    normalization='global',
    colormap='viridis'
)

# Generate Triangle Start attention heatmap
plot_all_heads_heatmap(
    attention_dir='./outputs/attention_files_6KWC_demo_tri_18',
    output_dir='./outputs/heatmaps',
    protein='6KWC',
    attention_type='triangle_start',
    layer_idx=47,
    fasta_path='./examples/monomer/fasta_dir_6KWC/6KWC.fasta',
    normalization='global',
    residue_indices=[18, 39, 51]
)

# Generate combined heatmap (both attention types)
plot_combined_attention_heatmap(
    attention_dir='./outputs/attention_files_6KWC_demo_tri_18',
    output_dir='./outputs/heatmaps',
    protein='6KWC',
    layer_idx=47,
    fasta_path='./examples/monomer/fasta_dir_6KWC/6KWC.fasta',
    normalization='global',
    residue_indices=[18, 39]
)
```

#### Optional Method: Command Line Interface

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

### Example Outputs

MSA Row Attention Heatmap (Layer 47):

![msa_row_heatmap](./outputs/heatmaps/msa_row_heatmap_layer_47_6KWC.png)

Triangle Start Attention Heatmap (Layer 47, Residue 18):

![triangle_start_heatmap](./outputs/heatmaps/triangle_start_heatmap_layer_47_6KWC.png)

Combined Attention Heatmap (Layer 47):

![combined_heatmap](./outputs/heatmaps/combined_attention_heatmap_layer_47_6KWC.png)

---

## Acknowledgements

This project is based on [**OpenFold**](https://github.com/aqlaboratory/openfold), an open-source reimplementation of AlphaFold, distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

We have extended OpenFold with:
- Custom visualization tools for attention maps (3D + arc diagrams + heatmaps)
- Demo scripts and configuration for interactive analysis
- Modifications to the inference pipeline for simplified usage
- Heatmap visualization for cross-head attention analysis

This repository includes source code originally developed by the OpenFold contributors. All original rights and attributions are retained in accordance with the Apache 2.0 License.

---

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
See the [LICENSE](./LICENSE) file for details.

---
