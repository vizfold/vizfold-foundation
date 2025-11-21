"""
Service that uses the REAL visualization utilities from parent directory
This generates the actual visualizations shown in attention-viz-demo
"""
import sys
import os
from pathlib import Path

# Add parent directory to path to import visualization utilities
parent_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import the real visualization utilities
from visualize_attention_arc_diagram_demo_utils import (
    load_all_heads,
    parse_fasta_sequence,
    plot_arc_diagram_with_labels
)
from visualize_attention_heatmap_demo_utils import (
    load_all_heads as load_heads_heatmap,
    generate_heatmap
)
from visualize_attention_3d_demo_utils import (
    plot_pymol_attention_heads
)


class RealVisualizationService:
    """
    Service that generates visualizations using the original utilities
    from attention-viz-demo, ensuring accuracy and consistency
    """

    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_arc_diagram(
        self,
        attention_file: str,
        sequence: str,
        layer: int,
        head: int,
        output_path: str,
        top_k: int = 50,
        highlight_residue: int = None
    ):
        """
        Generate arc diagram using the real visualization utility

        Args:
            attention_file: Path to attention dump file (format: "Layer X, Head Y\\nsource target weight\\n...")
            sequence: Amino acid sequence string
            layer: Layer number
            head: Head number
            output_path: Where to save the PNG
            top_k: Number of top connections to show
            highlight_residue: Optional residue to highlight
        """
        # Check if attention file exists
        if not os.path.exists(attention_file):
            raise FileNotFoundError(
                f"Attention file not found at {attention_file}. "
                "Please run inference on this protein first to generate attention weights."
            )

        # Load attention data using real utility
        heads = load_all_heads(attention_file, top_k=top_k)

        # Get connections for this specific head
        if head not in heads:
            available_heads = list(heads.keys())
            raise ValueError(
                f"Head {head} not found in attention file. "
                f"Available heads: {available_heads}"
            )

        connections = heads[head]

        # Generate arc diagram using real utility
        title = f"MSA Row Attention - Arc Diagram (Head {head}, Layer {layer})"
        plot_arc_diagram_with_labels(
            connections=connections,
            residue_sequence=sequence,
            output_file=output_path,
            highlight_residue_index=highlight_residue,
            save_to_png=True,
            plt_title=title
        )

        return output_path

    def generate_heatmap(
        self,
        attention_dir: str,
        protein: str,
        layer: int,
        attention_type: str,
        output_dir: str,
        top_k: int = None,
        residue_index: int = None
    ):
        """
        Generate heatmap using the real visualization utility

        Args:
            attention_dir: Directory containing attention dump files
            protein: Protein identifier string
            layer: Layer number
            attention_type: 'msa_row' or 'triangle_start'
            output_dir: Where to save the PNG
            top_k: Optional top-k filtering
            residue_index: Optional residue index for triangle attention
        """
        # Check if attention directory exists
        if not os.path.exists(attention_dir):
            raise FileNotFoundError(
                f"Attention directory not found at {attention_dir}. "
                "Please run inference on this protein first to generate attention weights."
            )

        # Use the real heatmap generator with correct parameters
        generate_heatmap(
            attention_dir=attention_dir,
            output_dir=output_dir,
            protein=protein,
            attention_type=attention_type,
            layer_idx=layer,
            top_k=top_k,
            residue_index=residue_index
        )

        # The function creates files in output_dir/attention_type/ subdirectory
        output_subdir = os.path.join(output_dir, attention_type)
        output_file = f"attention_heatmap_layer_{layer}_{protein}.png"
        return os.path.join(output_subdir, output_file)

    def generate_3d_visualization(
        self,
        pdb_file: str,
        attention_dir: str,
        protein: str,
        layer: int,
        head: int,
        attention_type: str,
        output_dir: str,
        top_k: int = 50,
        residue_index: int = None
    ):
        """
        Generate 3D PyMOL visualization using the real visualization utility

        Args:
            pdb_file: Path to PDB structure file
            attention_dir: Directory containing attention dump files
            protein: Protein identifier string
            layer: Layer number
            head: Head number
            attention_type: 'msa_row' or 'triangle_start'
            output_dir: Where to save the PNG
            top_k: Number of top connections to show
            residue_index: Optional residue index for triangle attention
        """
        # Check if PDB file exists
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(
                f"PDB file not found at {pdb_file}. "
                "Please upload a PDB file for this protein or run structure prediction first."
            )

        os.makedirs(output_dir, exist_ok=True)

        if attention_type == "msa_row":
            # For a single head, we need to manually call the utility
            msa_file = os.path.join(attention_dir, f"msa_row_attn_layer{layer}.txt")

            if not os.path.exists(msa_file):
                raise FileNotFoundError(
                    f"Attention file not found at {msa_file}. "
                    "Please run inference on this protein first to generate attention weights."
                )

            heads_data = load_all_heads(msa_file, top_k=top_k)

            if head not in heads_data:
                available_heads = list(heads_data.keys())
                raise ValueError(
                    f"Head {head} not found in attention file. "
                    f"Available heads: {available_heads}"
                )

            connections = heads_data[head]
            output_path = os.path.join(output_dir, f"msa_row_head_{head}_layer_{layer}_{protein}.png")

            # Import and use master_plot directly
            from visualize_attention_3d_demo_utils import master_plot
            master_plot(pdb_file, connections, output_path, base_color=(0.0, 0.0, 1.0))

            return output_path

        elif attention_type == "triangle_start":
            if residue_index is None:
                raise ValueError("residue_index is required for triangle_start attention")

            tri_file = os.path.join(attention_dir, f"triangle_start_attn_layer{layer}_residue_idx_{residue_index}.txt")

            if not os.path.exists(tri_file):
                raise FileNotFoundError(
                    f"Attention file not found at {tri_file}. "
                    "Please run inference on this protein first to generate attention weights."
                )

            heads_data = load_all_heads(tri_file, top_k=top_k)

            if head not in heads_data:
                available_heads = list(heads_data.keys())
                raise ValueError(
                    f"Head {head} not found in attention file. "
                    f"Available heads: {available_heads}"
                )

            connections = heads_data[head]
            output_path = os.path.join(output_dir, f"tri_start_residue_{residue_index}_head_{head}_layer_{layer}_{protein}.png")

            # Import and use master_plot directly
            from visualize_attention_3d_demo_utils import master_plot
            master_plot(pdb_file, connections, output_path,
                       base_color=(0.0, 0.0, 1.0),
                       highlight_res_index=residue_index)

            return output_path

    def load_attention_data(self, attention_file: str, head: int, top_k: int = None):
        """
        Load attention data in the format expected by frontend

        Returns:
            List of {source, target, weight} dicts for API/frontend
        """
        heads = load_all_heads(attention_file, top_k=top_k)

        if head not in heads:
            return []

        connections = heads[head]
        return [
            {"source": int(src), "target": int(tgt), "weight": float(weight)}
            for src, tgt, weight in connections
        ]
