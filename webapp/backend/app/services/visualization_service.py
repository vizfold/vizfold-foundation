import os
import subprocess
from typing import Tuple, Optional
from app.core.config import settings


class VisualizationService:
    """Service for generating visualizations"""

    def __init__(self):
        self.output_dir = settings.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    async def generate_visualization(
        self,
        protein_id: int,
        viz_type: str,
        layer: int,
        head: Optional[int],
        attention_type: str,
        residue_index: Optional[int] = None,
        top_k: Optional[int] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Generate visualization and return paths to image and thumbnail

        This integrates with your existing visualization utilities from
        attention-viz-demo repository
        """

        # Construct output filename
        filename_parts = [
            f"protein{protein_id}",
            attention_type,
            f"layer{layer}"
        ]

        if head is not None:
            filename_parts.append(f"head{head}")

        if residue_index is not None:
            filename_parts.append(f"res{residue_index}")

        filename = f"{viz_type}_{'_'.join(filename_parts)}.png"
        image_path = os.path.join(self.output_dir, filename)

        # Generate visualization based on type
        if viz_type == "heatmap":
            await self._generate_heatmap(
                protein_id, layer, attention_type, image_path, top_k
            )
        elif viz_type == "arc_diagram":
            await self._generate_arc_diagram(
                protein_id, layer, head, attention_type, residue_index, image_path
            )
        elif viz_type == "3d":
            await self._generate_3d(
                protein_id, layer, head, attention_type, residue_index, image_path
            )
        elif viz_type == "combined":
            await self._generate_combined(
                protein_id, layer, head, attention_type, residue_index, image_path
            )
        else:
            raise ValueError(f"Unknown visualization type: {viz_type}")

        # Generate thumbnail (optional)
        thumbnail_path = None
        # TODO: Implement thumbnail generation using PIL/Pillow

        return image_path, thumbnail_path

    async def _generate_heatmap(
        self,
        protein_id: int,
        layer: int,
        attention_type: str,
        output_path: str,
        top_k: Optional[int]
    ):
        """Generate heatmap visualization using your existing utility"""

        # This would call your visualize_attention_heatmap_demo_utils.py
        # You'll need to adapt this to your actual implementation

        attention_dir = os.path.join(settings.OUTPUT_DIR, f"protein_{protein_id}", "attention")

        cmd = [
            "python",
            "visualize_attention_heatmap_demo_utils.py",
            "--attention-dir", attention_dir,
            "--output-dir", os.path.dirname(output_path),
            "--protein-id", str(protein_id),
            "--layer", str(layer),
            "--attention-type", attention_type
        ]

        if top_k:
            cmd.extend(["--top-k", str(top_k)])

        # Run the visualization script
        # subprocess.run(cmd, check=True)

        # For now, create a placeholder file
        self._create_placeholder(output_path, f"Heatmap: Layer {layer}")

    async def _generate_arc_diagram(
        self,
        protein_id: int,
        layer: int,
        head: Optional[int],
        attention_type: str,
        residue_index: Optional[int],
        output_path: str
    ):
        """Generate arc diagram using your existing utility"""
        # TODO: Integrate with visualize_attention_arc_diagram_demo_utils.py
        self._create_placeholder(output_path, f"Arc Diagram: Layer {layer}, Head {head}")

    async def _generate_3d(
        self,
        protein_id: int,
        layer: int,
        head: Optional[int],
        attention_type: str,
        residue_index: Optional[int],
        output_path: str
    ):
        """Generate 3D visualization using your existing utility"""
        # TODO: Integrate with visualize_attention_3d_demo_utils.py
        self._create_placeholder(output_path, f"3D View: Layer {layer}, Head {head}")

    async def _generate_combined(
        self,
        protein_id: int,
        layer: int,
        head: Optional[int],
        attention_type: str,
        residue_index: Optional[int],
        output_path: str
    ):
        """Generate combined visualization panel"""
        # TODO: Combine multiple visualization types
        self._create_placeholder(output_path, f"Combined: Layer {layer}, Head {head}")

    def _create_placeholder(self, path: str, text: str):
        """Create placeholder image for development"""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=20)
        ax.axis('off')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
