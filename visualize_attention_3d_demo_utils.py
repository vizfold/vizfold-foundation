import argparse
import csv
from pathlib import Path

import numpy as np
from pymol import cmd
from pymol.cgo import CYLINDER, SPHERE

# Initialize cmd.stored manually
cmd.stored = type('stored', (object,), {})()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


# ========== Attention File I/O ==========
def load_all_heads(connections_file, top_k=None):
    """
    Loads all heads' connections from a combined text file.
    Returns a dict mapping head_index -> list of (res1, res2, weight).
    """
    heads = {}
    current_head = None

    with open(connections_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith('layer'):
                # New head section
                parts = line.replace(',', '').split()
                head_idx = int(parts[-1])
                current_head = head_idx
                heads[current_head] = []
            else:
                # Residue-residue-weight line
                res1, res2, weight = map(float, line.split())
                heads[current_head].append((int(res1), int(res2), weight))

    # Sort each head's connections
    for head_idx, conns in heads.items():
        conns.sort(key=lambda x: x[2], reverse=True)
        if top_k is not None:
            heads[head_idx] = conns[:top_k]

    return heads


def load_connections(connections_file, top_k=None):
    """
    Loads connections (res1, res2, weight) from a text file.
    Sorts by descending weight and selects top_k if specified.
    """
    connections = []

    with open(connections_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            res1 = int(row[0])
            res2 = int(row[1])
            weight = float(row[2])
            connections.append((res1, res2, weight))

    # Sort by weight descending
    connections.sort(key=lambda x: x[2], reverse=True)

    if top_k is not None:
        connections = connections[:top_k]

    return connections


def extract_head_number(filename):
    parts = filename.replace('.', '_').replace('-', '_').split('_')
    for i, part in enumerate(parts):
        if part.lower() == 'head' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
        if part.lower().startswith('head'):
            try:
                return int(part.lower().replace('head', ''))
            except ValueError:
                pass
    return -1


# ========== Residue Indexing and Geometry ==========
def check_residue_numbering(selection='all'):
    """
    Automatically checks if residue numbering matches array indices.
    """
    model = cmd.get_model(f"({selection}) and name CA")
    stored_residues = sorted(list(set(int(atom.resi) for atom in model.atom)))

    if not stored_residues:
        print("No residues found in selection!")
        return None

    # print(f"Detected residues: {stored_residues[:10]} ... (total {len(stored_residues)})")

    expected = list(range(1, len(stored_residues) + 1))
    if stored_residues == expected:
        # print("Residues are sequential starting at 1 — simple +1 mapping.")
        return 'plus_one'
    elif stored_residues[0] != 0 and stored_residues[0] != 1:
        # print(f"Residues start at {stored_residues[0]} — building index-to-residue mapping.")
        index_to_resi = {i: resi for i, resi in enumerate(stored_residues)}
        return index_to_resi
    else:
        print("Residue numbering unexpected — manual check recommended.")
        return None


def get_backbone_center(resi, selection='all'):
    coords = []
    for atom in ['N', 'CA', 'C']:
        try:
            coord = cmd.get_atom_coords(f"({selection}) and resi {resi} and name {atom}")
            coords.append(coord)
        except:
            continue
    if coords:
        return [sum(x)/len(x) for x in zip(*coords)]
    return None


def offset_point_pair(p1, p2, offset=0.1):
    v = np.array(p2) - np.array(p1)
    norm = np.linalg.norm(v)
    if norm == 0:
        return p1, p2
    unit = v / norm
    return (list(np.array(p1) + offset * unit), list(np.array(p2) - offset * unit))


# ========== Drawing Utilities ==========
def normalize_weight(w, w_min, w_max, r_min=0.15, r_max=0.7):
    if w_max != w_min:
        return r_min + ((w - w_min) / (w_max - w_min)) * (r_max - r_min)
    return (r_min + r_max) / 2


def color_from_weight_monochrome(w, w_min, w_max, base_color=(0.0, 0.0, 1.0), invert=False):
    if w_max != w_min:
        norm_w = (w - w_min) / (w_max - w_min)
    else:
        norm_w = 0.5
    factor = norm_w if invert else (1.0 - norm_w)
    return [c * factor for c in base_color]


def draw_connections(connections, mapping, selection='all', base_color=(0.0, 0.0, 1.0)):
    obj = []
    weights = [w for _, _, w in connections]
    w_min, w_max = min(weights), max(weights)

    for res1_index, res2_index, weight in connections:
        if mapping == 'plus_one':
            res1 = res1_index + 1
            res2 = res2_index + 1
        elif isinstance(mapping, dict):
            res1 = mapping.get(res1_index)
            res2 = mapping.get(res2_index)
        else:
            res1, res2 = res1_index, res2_index

        coord1 = get_backbone_center(res1, selection)
        coord2 = get_backbone_center(res2, selection)

        if not coord1 or not coord2:
            continue

        coord1, coord2 = offset_point_pair(coord1, coord2, offset=0.2)
        radius = normalize_weight(weight, w_min, w_max)
        color = color_from_weight_monochrome(weight, w_min, w_max, base_color=base_color)

        obj.extend([
            CYLINDER,
            *coord1,
            *coord2,
            radius,
            *color,
            *color
        ])

    cmd.load_cgo(obj, 'connections')


def draw_highlight_residue(res_index, mapping, selection='all', radius=1.0, color=(1.0, 0.0, 0.0)):
    """
    Draw a sphere at the specified residue index.
    """
    if mapping == 'plus_one':
        resi = res_index + 1
    elif isinstance(mapping, dict):
        resi = mapping.get(res_index, res_index)
    else:
        resi = res_index

    coord = get_backbone_center(resi, selection)
    if coord:
        sphere_obj = [SPHERE, *coord, radius, *color]
        cmd.load_cgo(sphere_obj, f"highlight_residue_{resi}")
        print(f"Marked residue {resi} with sphere.")
    else:
        print(f"Could not find coords for residue {resi}")


# ========== Plotting Utilities ==========
def plot_attention_grid(image_paths, titles, rows, cols, figure_title, output_file):
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows), constrained_layout=True)
    axes = axes.flatten()

    for ax, img_path, title in zip(axes, image_paths, titles):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    for ax in axes[len(image_paths):]:
        ax.axis('off')

    fig.suptitle(figure_title, fontsize=14, weight='bold', y=1.02)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # plt.show()
    print(f"Saved summary grid to {output_file}")


# ========== PyMOL Visualization ==========
def master_plot(pdb_file, connections, save_path,
                selection='all', base_color=(0.0, 0.0, 1.0),
                highlight_res_index=None):

    cmd.reinitialize()
    cmd.load(pdb_file, 'structure')

    # Check residue mapping for indexing
    mapping = check_residue_numbering(selection=selection)

    print(f"Drawing {len(connections)} connections")

    cmd.bg_color('white')
    cmd.show('cartoon', 'structure')
    cmd.color('gray80', 'structure')
    cmd.set('cartoon_transparency', 0.0)
    cmd.set('cartoon_side_chain_helper', 1)
    cmd.hide('lines', 'structure')

    cmd.set('ray_trace_mode', 1)
    cmd.set('ray_trace_gain', 0.1)
    cmd.set('specular', 0.3)
    cmd.set('ambient', 0.4)
    cmd.set('direct', 0.8)
    cmd.set('reflect', 0.1)

    draw_connections(connections, mapping, selection=selection,
                     base_color=base_color)
    
    if highlight_res_index is not None:
        draw_highlight_residue(highlight_res_index, mapping, selection=selection)

    cmd.orient()
    cmd.viewport(1920, 1080)
    cmd.ray(1920, 1080)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cmd.png(save_path, dpi=300)

    print(f"Saved snapshot to {save_path}")


def plot_pymol_attention_heads(
    pdb_file,
    attention_dir,
    output_dir,
    protein,
    attention_type="msa_row",
    residue_indices=None,
    top_k=50,
    layer_idx=47,
    ):
    """
    Generate PyMOL visualizations of MSA row or triangle start attention.
    
    Args:
        pdb_file (str): Path to input PDB.
        attention_dir (str): Directory with attention text files.
        output_dir (str): Where to save output PNGs.
        attention_type (str): 'msa_row' or 'triangle_start'.
        residue_indices (List[int]): Only for triangle_start.
        top_k (int): Max number of attention edges to show.
        layer_idx (int): Which layer's attention to visualize.
    """

    os.makedirs(output_dir, exist_ok=True)

    if attention_type == "msa_row":
        msa_file = os.path.join(attention_dir, f"msa_row_attn_layer{layer_idx}.txt")
        msa_heads = load_all_heads(msa_file, top_k=top_k)

        image_paths = []
        for head_idx, connections in msa_heads.items():
            output_png = os.path.join(output_dir, f"msa_row_head_{head_idx}_layer_{layer_idx}_{protein}.png")
            master_plot(pdb_file, connections, output_png, base_color=(0.0, 0.0, 1.0))
            image_paths.append(output_png)

        # Subplot
        titles = [f"MSA Row Head {extract_head_number(p)}" for p in image_paths]
        subplot_path = os.path.join(output_dir, f"msa_row_heads_layer_{layer_idx}_{protein}_subplot.png")
        plot_attention_grid(image_paths, titles, rows=2, cols=4,
                            figure_title=f"{protein} MSA Row Attention Heads Layer {layer_idx}", output_file=subplot_path)

    elif attention_type == "triangle_start":
        assert residue_indices is not None, "Must supply residue_indices for triangle attention"

        for res_idx in residue_indices:
            tri_file = os.path.join(attention_dir, f"triangle_start_attn_layer{layer_idx}_residue_idx_{res_idx}.txt")
            if not os.path.exists(tri_file):
                print(f"[Warning] Missing attention file for residue {res_idx}")
                continue

            tri_heads = load_all_heads(tri_file, top_k=top_k)
            res_pngs = []
            for head_idx, connections in tri_heads.items():
                output_png = os.path.join(output_dir, f"tri_start_residue_{res_idx}_head_{head_idx}_layer_{layer_idx}_{protein}.png")
                master_plot(pdb_file, connections, output_png,
                            base_color=(0.0, 0.0, 1.0),
                            highlight_res_index=res_idx)
                res_pngs.append(output_png)

            # Subplot for this residue
            subplot_path = os.path.join(output_dir, f"triangle_start_residue_{res_idx}_layer_{layer_idx}_{protein}_subplot.png")
            titles = [f"Head {extract_head_number(p)}" for p in res_pngs]
            plot_attention_grid(res_pngs, titles, rows=1, cols=len(res_pngs),
                                figure_title=f"Triangle Start Attention Heads Layer {layer_idx} — Residue {res_idx}",
                                output_file=subplot_path)

def _parse_comma_separated_ints(raw):
    if raw is None or raw.strip() == "":
        return None
    return [int(x.strip()) for x in raw.split(',') if x.strip()]


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate PyMOL renders for OpenFold attention heads.")
    parser.add_argument("--pdb-file", type=Path, required=True, help="Path to relaxed PDB structure.")
    parser.add_argument("--attention-dir", type=Path, required=True, help="Directory containing attention text dumps.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for rendered PNG outputs.")
    parser.add_argument("--protein", type=str, required=True, help="Protein identifier used in filenames.")
    parser.add_argument("--attention-type", type=str, default="msa_row", choices=["msa_row", "triangle_start", "both"], help="Attention family to render.")
    parser.add_argument("--layer-idx", type=int, default=47, help="Evoformer layer index to visualize.")
    parser.add_argument("--top-k", type=int, default=50, help="Maximum number of edges per head.")
    parser.add_argument("--residue-indices", type=str, default="", help="Comma separated residue indices for triangle attention.")
    return parser.parse_args()


def _main():
    args = _parse_args()
    residue_indices = _parse_comma_separated_ints(args.residue_indices)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.attention_type in ("msa_row", "both"):
        print("Making visuals for MSA Row Attention...")
        plot_pymol_attention_heads(
            pdb_file=str(args.pdb_file),
            attention_dir=str(args.attention_dir),
            output_dir=str(output_dir / "msa_row"),
            protein=args.protein,
            attention_type="msa_row",
            top_k=args.top_k,
            layer_idx=args.layer_idx,
        )

    if args.attention_type in ("triangle_start", "both"):
        if residue_indices is None:
            raise ValueError("--residue-indices is required when rendering triangle_start attention")

        print("Making visuals for Triangle Start Attention...")
        plot_pymol_attention_heads(
            pdb_file=str(args.pdb_file),
            attention_dir=str(args.attention_dir),
            output_dir=str(output_dir / "triangle_start"),
            protein=args.protein,
            attention_type="triangle_start",
            residue_indices=residue_indices,
            top_k=args.top_k,
            layer_idx=args.layer_idx,
        )


if __name__ == "__main__":
    _main()
