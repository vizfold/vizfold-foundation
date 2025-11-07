#!/usr/bin/env python3
"""
Display existing attention visualizations for protein 6KWC.
This script shows the pre-generated visualizations and explains what they represent.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

def display_visualization(image_path, title, description):
    """Display a single visualization with explanation."""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    print(f"📁 File: {os.path.basename(image_path)}")
    print(f"📏 Size: {os.path.getsize(image_path) / 1024:.1f} KB")
    print(f"📝 Description: {description}")
    
    # Load and display image
    img = mpimg.imread(image_path)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    print("🧬 OpenFold Attention Visualization Demo")
    print("=" * 50)
    print("This demo shows attention patterns captured from OpenFold")
    print("during protein structure prediction for protein 6KWC.")
    print("\nProtein 6KWC:")
    print("- Length: 159 amino acids")
    print("- Function: Unknown (hypothetical protein)")
    print("- Sequence: GSTIQPGTGYNNGYFYSYWNDGHGGVTYTNGPGGQFSVNWSNSGEFVGGKGWQPGTKNKVINFSGSYNPNGNSYLSVYGWSRNPLIEYYIVENFGTYNPSTGATKLGEVTSDGSVYDIYRTQRVNQPSIIGTATFYQYWSVRRNHRSSGSVNTANHFNAWAQQGLTLGTMDYQIVAVQGYFSSGSASITVS")
    
    # Define visualizations to show
    visualizations = [
        {
            "path": "outputs/attention_images_6KWC_demo_tri_18/msa_row_attention_plots/msa_row_head_2_layer_47_6KWC_arc.png",
            "title": "MSA Row Attention - Arc Diagram (Head 2, Layer 47)",
            "description": """
🔍 MSA Row Attention shows how residues attend to each other based on evolutionary relationships.

📈 What you see:
- X-axis: Residue positions (1-159)
- Y-axis: Attention strength (arc height)
- Arcs: Connect residues that strongly attend to each other
- Thickness: Attention weight (thicker = stronger attention)

🧬 Biological meaning:
- High attention between distant residues suggests evolutionary co-evolution
- Local attention patterns may indicate structural constraints
- This is from layer 47 (deep in the network) - captures complex relationships
            """
        },
        {
            "path": "outputs/attention_images_6KWC_demo_tri_18/msa_row_attention_plots/msa_row_heads_layer_47_6KWC_subplot.png",
            "title": "MSA Row Attention - All Heads Comparison (Layer 47)",
            "description": """
🔍 This shows all 8 attention heads side-by-side for comparison.

📈 What you see:
- 8 subplots (one per attention head)
- Each head learns different attention patterns
- Some heads focus on local interactions, others on long-range
- Head diversity allows the model to capture multiple relationship types

🧬 Biological insight:
- Different heads may specialize in different types of relationships
- Comparing heads reveals which patterns are consistent vs. head-specific
            """
        },
        {
            "path": "outputs/attention_images_6KWC_demo_tri_18/tri_start_attention_plots/tri_start_res_18_head_0_layer_47_6KWC_arc.png",
            "title": "Triangle Start Attention - Arc Diagram (Residue 18, Head 0, Layer 47)",
            "description": """
🔍 Triangle Start Attention shows what residue 18 "attends to" during geometric reasoning.

📈 What you see:
- Blue highlight: Residue 18 (the "source" residue)
- Arcs: Show what residue 18 attends to
- Height: Attention strength from residue 18 to other residues
- This is geometric attention (spatial relationships)

🧬 Biological meaning:
- Residue 18 is learning about its spatial neighborhood
- High attention to nearby residues suggests local structure
- Attention to distant residues may indicate functional relationships
- Residue 18: Glycine (G) - often important for flexibility
            """
        },
        {
            "path": "outputs/attention_images_6KWC_demo_tri_18/tri_start_attention_plots/triangle_start_residue_18_layer_47_6KWC_subplot.png",
            "title": "Triangle Start Attention - All Heads (Residue 18, Layer 47)",
            "description": """
🔍 All attention heads for residue 18, showing different geometric perspectives.

📈 What you see:
- 4 subplots (triangle attention has 4 heads)
- Each head shows residue 18's attention from a different geometric viewpoint
- Some heads may focus on local structure, others on long-range contacts

🧬 Biological insight:
- Different geometric heads capture different spatial relationship types
- Consistent patterns across heads suggest important structural features
            """
        }
    ]
    
    print(f"\n🎨 Found {len(visualizations)} visualizations to display:")
    
    for i, viz in enumerate(visualizations, 1):
        print(f"\n{i}. {viz['title']}")
        if os.path.exists(viz['path']):
            print(f"   ✅ Available")
        else:
            print(f"   ❌ Missing")
    
    # Display each visualization
    for viz in visualizations:
        if os.path.exists(viz['path']):
            display_visualization(viz['path'], viz['title'], viz['description'])
        else:
            print(f"\n❌ Skipping {viz['title']} - file not found")
    
    print(f"\n{'='*60}")
    print("🎉 Visualization Demo Complete!")
    print("=" * 60)
    print("\n📚 What you just saw:")
    print("• MSA Row Attention: Evolutionary relationships between residues")
    print("• Triangle Start Attention: Geometric relationships from a specific residue")
    print("• Multiple heads: Different learned attention patterns")
    print("• Layer 47: Deep network layer capturing complex relationships")
    
    print("\n🔬 Key Insights:")
    print("• Attention patterns reveal how AlphaFold 'thinks' about protein structure")
    print("• Different attention types capture different biological relationships")
    print("• Multiple heads allow the model to learn diverse interaction patterns")
    print("• Deep layers (like 47) capture complex, long-range relationships")
    
    print("\n🚀 Next Steps:")
    print("• Try different proteins by running the full inference pipeline")
    print("• Explore different layers (0-47) to see how attention evolves")
    print("• Build interactive web interface for real-time exploration")
    print("• Develop new visualization methods (heatmaps, graphs, animations)")

if __name__ == "__main__":
    main()

