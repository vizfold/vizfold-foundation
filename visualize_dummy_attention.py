import os
import numpy as np
import visualize_attention_heatmap_utils
from visualize_attention_heatmap_utils import plot_all_heads_heatmap

# -------------------------------
# 1. Create dummy attention matrix
# -------------------------------
num_tokens = 10
matrix = np.random.rand(num_tokens, num_tokens)
tokens = [f"T{i}" for i in range(num_tokens)]
protein_name = "dummy_protein"

# -------------------------------
# 2. Define output directory as a string
# -------------------------------
attention_dir = f"outputs/attention_images_{protein_name}/msa_row_attention_plots"
os.makedirs(attention_dir, exist_ok=True)

# -------------------------------
# 3. Override module-level variable
# -------------------------------
visualize_attention_heatmap_utils.attention_dir = attention_dir
