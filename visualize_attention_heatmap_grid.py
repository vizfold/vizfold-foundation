import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_all_heads(connections_file):
    heads = {}
    current_head = None
    with open(connections_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith('layer'):
                parts = line.replace(',', '').split()
                head_idx = int(parts[-1])
                current_head = head_idx
                heads[current_head] = []
            else:
                res1, res2, weight = map(float, line.split())
                heads[current_head].append((int(res1), int(res2), weight))
    return heads


def reconstruct_matrix(connections, seq_len):
    matrix = np.zeros((seq_len, seq_len))
    for res1, res2, weight in connections:
        if res1 < seq_len and res2 < seq_len:
            matrix[res1, res2] = weight
    return matrix


def create_heatmap_grid(attention_file, seq_len, layer_idx=47, attention_type="msa_row", output_html="heatmap_grid.html"):
    heads = load_all_heads(attention_file)
    num_heads = len(heads)

    if num_heads == 0:
        print(f"No heads found in {attention_file}")
        return

    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    all_weights = [w for head_idx in sorted(heads.keys()) for _, _, w in heads[head_idx]]
    global_min = min(all_weights) if all_weights else 0
    global_max = max(all_weights) if all_weights else 1

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Head {i}" for i in sorted(heads.keys())],
        horizontal_spacing=0.05,
        vertical_spacing=0.08
    )

    for idx, head_idx in enumerate(sorted(heads.keys())):
        row = idx // cols + 1
        col = idx % cols + 1

        matrix = reconstruct_matrix(heads[head_idx], seq_len)

        fig.add_trace(
            go.Heatmap(
                z=matrix,
                colorscale='Blues',
                zmin=global_min,
                zmax=global_max,
                showscale=(idx == 0),
                colorbar=dict(x=1.02, len=0.3, title="Weight") if idx == 0 else None
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Residue", row=row, col=col, showticklabels=False)
        fig.update_yaxes(title_text="Residue", row=row, col=col, showticklabels=False)

    fig.update_layout(
        title_text=f"{attention_type.upper()} Layer {layer_idx} - All Heads (Globally Normalized)",
        title_x=0.5,
        height=300 * rows,
        width=1200,
        showlegend=False
    )

    fig.write_html(output_html)
    print(f"Saved: {output_html}")
    return fig


def visualize_layer_attention(attention_dir, seq_len, layer_idx=47, attention_type="msa_row", residue_idx=None, output_dir="./outputs/attention_heatmaps"):
    os.makedirs(output_dir, exist_ok=True)

    if attention_type == "msa_row":
        attention_file = os.path.join(attention_dir, f"msa_row_attn_layer{layer_idx}.txt")
        output_html = os.path.join(output_dir, f"msa_row_layer{layer_idx}_heatmap_grid.html")
    elif attention_type == "triangle_start":
        if residue_idx is None:
            raise ValueError("residue_idx required for triangle_start")
        attention_file = os.path.join(attention_dir, f"triangle_start_attn_layer{layer_idx}_residue_idx_{residue_idx}.txt")
        output_html = os.path.join(output_dir, f"triangle_start_layer{layer_idx}_res{residue_idx}_heatmap_grid.html")
    else:
        raise ValueError(f"Unknown attention_type: {attention_type}")

    if not os.path.exists(attention_file):
        print(f"File not found: {attention_file}")
        return None

    print(f"Processing: {attention_file}")
    return create_heatmap_grid(attention_file, seq_len, layer_idx, attention_type, output_html)


if __name__ == "__main__":
    attention_dir = "./outputs/attention_files_6KWC_demo_tri_18"
    seq_len = 400
    layer_idx = 47

    visualize_layer_attention(
        attention_dir=attention_dir,
        seq_len=seq_len,
        layer_idx=layer_idx,
        attention_type="msa_row"
    )

    visualize_layer_attention(
        attention_dir=attention_dir,
        seq_len=seq_len,
        layer_idx=layer_idx,
        attention_type="triangle_start",
        residue_idx=18
    )
