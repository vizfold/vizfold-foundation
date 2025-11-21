import argparse
import os
from pathlib import Path

from pymol import cmd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from visualize_attention_3d_demo_utils import extract_head_number


def make_fasta_file(FASTA_PATH, FASTA_DIR, FASTA_SEQUENCE):
    # Make sure output directory exists
    os.makedirs(FASTA_DIR, exist_ok=True)

    # Write FASTA to disk
    with open(FASTA_PATH, "w") as f:
        f.write(FASTA_SEQUENCE)

    print(f"Saved FASTA to {FASTA_PATH}")


def render_pdb_to_image(pdb_file, output_image_path, fname):
    os.makedirs(output_image_path, exist_ok=True)
    output_image_path = os.path.join(output_image_path, fname)

    cmd.reinitialize()
    cmd.load(pdb_file, 'protein')

    cmd.bg_color('white')
    cmd.show('cartoon', 'protein')
    cmd.color('gray80', 'protein')
    cmd.hide('lines', 'protein')
    cmd.orient()

    cmd.viewport(800, 600)
    cmd.ray(800, 600)
    cmd.png(output_image_path, dpi=300)
    print(f"Saved image to {output_image_path}")
    
    show_image(output_image_path, 'Predicted Structure')


def show_image(image_path, title=None):
    img = mpimg.imread(image_path)
    plt.figure(figsize=(6, 5))
    plt.imshow(img)
    if title:
        plt.title(title, fontsize=14)
    plt.axis('off')
    plt.show()


def combine_3d_and_arc_images(
    structure_img_path,
    arc_img_path,
    output_path,
    title_top="3D Attention",
    title_bottom="Arc Diagram",
    fig_title=None,
    show_plot=True
):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)

    for ax, img_path, title in zip(axes, [structure_img_path, arc_img_path], [title_top, title_bottom]):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=12, weight='bold')
        ax.axis('off')

    if fig_title:
        fig.suptitle(fig_title, fontsize=16, weight='bold', y=1.03)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"[Saved] Combined panel to {output_path}")


def generate_combined_attention_panels(
    attention_type,  # "msa_row" or "triangle_start"
    protein,
    layer_idx,
    output_dir_3d,
    output_dir_arc,
    combined_output_dir,
    residue_indices=None  # for triangle_start
):
    os.makedirs(combined_output_dir, exist_ok=True)

    if attention_type == "msa_row":
        for fname in os.listdir(output_dir_3d):
            if fname.startswith("msa_row_head_") and fname.endswith(f"_{protein}.png") and f"layer_{layer_idx}_" in fname:
                head = extract_head_number(fname)
                prefix = f"msa_row_head_{head}_layer_{layer_idx}_{protein}"
                struct_path = os.path.join(output_dir_3d, f"{prefix}.png")
                arc_path = os.path.join(output_dir_arc, f"{prefix}_arc.png")
                out_path = os.path.join(combined_output_dir, f"{prefix}_combo.png")
                print('\n')
                print(struct_path)
                print(arc_path)
                print(out_path)
                if os.path.exists(struct_path) and os.path.exists(arc_path):
                    generate_title = f"MSA Row — Head {head}, Layer {layer_idx}, {protein}"
                    combine_3d_and_arc_images(struct_path, arc_path, out_path, fig_title=generate_title)
                else:
                    print(f"[Skipped] Missing image for {prefix}")

    elif attention_type == "triangle_start":
        assert residue_indices is not None
        for res_idx in residue_indices:
            for fname in os.listdir(output_dir_3d):
                if fname.startswith(f"tri_start_residue_{res_idx}_head_") and fname.endswith(f"_{protein}.png") and f"layer_{layer_idx}_" in fname:
                    head = extract_head_number(fname)
                    prefix = f"tri_start_residue_{res_idx}_head_{head}_layer_{layer_idx}_{protein}"
                    struct_path = os.path.join(output_dir_3d, f"{prefix}.png")
                    arc_path = os.path.join(output_dir_arc, f"tri_start_res_{res_idx}_head_{head}_layer_{layer_idx}_{protein}_arc.png")
                    out_path = os.path.join(combined_output_dir, f"{prefix}_combo.png")
                    print('\n')
                    print(struct_path)
                    print(arc_path)
                    print(out_path)
                    if os.path.exists(struct_path) and os.path.exists(arc_path):
                        generate_title = f"Triangle Start — Head {head}, Res {res_idx}, Layer {layer_idx}, {protein}"
                        combine_3d_and_arc_images(struct_path, arc_path, out_path, fig_title=generate_title)
                    else:
                        print(f"[Skipped] Missing image for {prefix}")


def _parse_comma_separated_ints(raw):
    if raw is None or raw.strip() == "":
        return None
    return [int(x.strip()) for x in raw.split(',') if x.strip()]


def _parse_args():
    parser = argparse.ArgumentParser(description="Combine 3D and arc attention visualizations into composite panels.")
    parser.add_argument("--attention-type", type=str, required=True, choices=["msa_row", "triangle_start"], help="Attention family to combine.")
    parser.add_argument("--protein", type=str, required=True, help="Protein identifier used in filenames.")
    parser.add_argument("--layer-idx", type=int, required=True, help="Layer index to target.")
    parser.add_argument("--output-dir-3d", type=Path, required=True, help="Directory containing PyMOL renders.")
    parser.add_argument("--output-dir-arc", type=Path, required=True, help="Directory containing arc diagrams.")
    parser.add_argument("--combined-output-dir", type=Path, required=True, help="Directory to write combined panels.")
    parser.add_argument("--residue-indices", type=str, default="", help="Comma separated residues for triangle attention.")
    return parser.parse_args()


def _main():
    args = _parse_args()
    residue_indices = _parse_comma_separated_ints(args.residue_indices)

    if args.attention_type == "triangle_start" and residue_indices is None:
        raise ValueError("--residue-indices is required for triangle_start attention panels")

    generate_combined_attention_panels(
        attention_type=args.attention_type,
        protein=args.protein,
        layer_idx=args.layer_idx,
        output_dir_3d=str(args.output_dir_3d),
        output_dir_arc=str(args.output_dir_arc),
        combined_output_dir=str(args.combined_output_dir),
        residue_indices=residue_indices,
    )


if __name__ == "__main__":
    _main()