# Quick Start: MAXED OUT OpenFold Visualization

## TL;DR - What's Been Maxed Out

Everything is now **ABSOLUTELY MAXED OUT** for comprehensive 48-layer analysis:

```
✓ 256 residues (2.5x increase)
✓ 32 MSA sequences (2x increase)  
✓ 12 recycles (1.5x increase)
✓ 15 residues tracked (3x increase)
✓ ALL 48 layers (100% coverage!)
✓ 8x6 grids (48 panels total)
✓ Larger figures (20x14 inches)
✓ 8x more data points (~400K)
```

---

## Running the MAXED OUT Notebook

### Step 1: Refresh Browser
Your Jupyter notebook is already running. Just refresh the page in your browser (`Cmd+R` or `F5`).

### Step 2: Restart Kernel
In Jupyter: **Kernel → Restart & Clear Output**

### Step 3: Run All Cells
Click **Cell → Run All**

### Expected Runtime
- **With CPU**: ~15-20 minutes
- **With GPU**: ~5-8 minutes

---

## What You'll Get

### 10 High-Resolution Visualizations:

1. **multilayer_evolution_MAXED.png** (~5 MB)
   - 15 residues tracked across ALL 48 layers
   - 20x14 inch figure
   - Top: magnitude evolution, Bottom: layer changes

2. **stratified_msa_ALL48.png** (~15 MB)
   - 8x6 grid = 48 MSA heatmaps
   - Every single layer visualized
   - 32x24 inch mega-figure

3. **stratified_pair_ALL48.png** (~25 MB)
   - 8x6 grid = 48 Pair heatmaps
   - 256x256 interactions per layer
   - Complete transformation tracking

4. **msa_convergence.png** (~3 MB)
   - Full 48x48 correlation matrix
   - Convergence point detection
   - 256-residue analysis

5. **pair_convergence.png** (~3 MB)
   - Layer-to-layer stability
   - Rate of change tracking
   - All 48 layers analyzed

6. **layer_importance.png** (~2 MB)
   - 3 metrics for all 48 layers
   - Top 10 layers labeled
   - Multi-metric ranking

7. **structure_evolution.png** (~8 MB)
   - 12 recycling iterations
   - 6-panel comprehensive analysis
   - 3,584 atoms tracked

8. **residue_50_analysis.png** (~4 MB)
   - 6-panel deep dive
   - 256 channels analyzed
   - Feature correlations

9. **layer_clustering.png** (~2 MB)
   - All 48 layers clustered
   - Dendrogram visualization
   - Similarity matrix

10. **pair_with_contacts.png** (~10 MB)
    - 256x256 contact map
    - Pearson correlation
    - Validation metrics

**Total**: ~77 MB of publication-quality figures!

---

## Key Changes from Standard

| Aspect | Standard | MAXED OUT |
|--------|----------|-----------|
| **Protein Size** | 100 residues | 256 residues |
| **MSA Sequences** | 15 | 32 |
| **Layer Coverage** | 13 layers (27%) | 48 layers (100%) |
| **Tracked Residues** | 5 | 15 |
| **Recycles** | 8 | 12 |
| **Grid Size** | 3x5 (15 panels) | 8x6 (48 panels) |
| **Figure Size** | 14x10" | 20x14" |
| **Runtime** | ~5 min | ~15 min |
| **Output Size** | ~30 MB | ~77 MB |

---

## Memory Requirements

- **Minimal**: 500 MB RAM
- **Standard**: 2 GB RAM
- **MAXED OUT**: 8 GB RAM ← **You need this**

---

## Customizing

Want even MORE? Edit these in Cell 4:

```python
# MAXED OUT parameters
n_seq = 32       # Increase to 64 for even more sequences
n_res = 256      # Increase to 512 for huge proteins
n_recycles = 12  # Increase to 16 for more iterations
```

In Cell 6:

```python
# Track even more residues
maxed_residues = list(range(0, n_res, 10))  # Every 10th residue!
```

---

## Sampling Strategies

Switch strategies anytime:

```python
# Current: ALL (100%)
sampled_layers = stratified_layer_sampling(48, strategy='all')

# Alternatives:
sampled_layers = stratified_layer_sampling(48, strategy='dense')    # 75%
sampled_layers = stratified_layer_sampling(48, strategy='random')   # 70%
sampled_layers = stratified_layer_sampling(48, strategy='grouped')  # 27%
sampled_layers = stratified_layer_sampling(48, strategy='uniform')  # 19%
```

---

## Troubleshooting

### "Out of Memory"
Reduce parameters:
```python
n_res = 128      # Half the residues
n_seq = 16       # Half the sequences
sampled_layers = stratified_layer_sampling(48, strategy='dense')  # 75% instead of 100%
```

### "Taking Too Long"
Use GPU or reduce coverage:
```python
layer_sampling = 'dense'  # 75% instead of 'all'
```

### "Figures Too Large to View"
They're meant for publication! View them externally or use:
```python
plt.savefig(path, dpi=150)  # Reduce from 300 DPI
```

---

## File Locations

All outputs saved to:
```
notebook_outputs/
├── multilayer_evolution_MAXED.png
├── stratified_msa_ALL48.png
├── stratified_pair_ALL48.png
├── msa_convergence.png
├── pair_convergence.png
├── layer_importance.png
├── structure_evolution.png
├── residue_50_analysis.png
├── layer_clustering.png
└── pair_with_contacts.png
```

---

## Next Steps

1. **Run the notebook** (see steps above)
2. **Review outputs** in `notebook_outputs/`
3. **Read** `MAXED_OUT_FEATURES.md` for technical details
4. **Customize** parameters for your specific needs
5. **Integrate** with real OpenFold inference

---

## Technical Details

- **Module Auto-Reload**: Enabled in Cell 2
- **Colormap**: tab20 for 15+ residues
- **Grid Layout**: Adaptive (3col/5col/8col based on layer count)
- **Figure DPI**: 300 (publication quality)
- **Seed**: 42 (for reproducible random sampling)

---

## Performance Tips

### Speed Up:
1. Use `strategy='dense'` (75%) instead of `'all'`
2. Reduce `n_res` to 128
3. Skip some analyses (comment out cells)
4. Use GPU if available

### Quality Up:
1. Increase DPI: `plt.savefig(path, dpi=600)`
2. Use vector format: `plt.savefig(path.replace('.png', '.pdf'))`
3. Increase figure sizes in code
4. Add more tracked residues

---

## Environment

Already set up! Your `base` environment has:
- ✓ PyTorch 2.8.0
- ✓ NumPy 2.2.6
- ✓ Matplotlib 3.10.6
- ✓ SciPy 1.16.2
- ✓ All utilities loaded with auto-reload

---

## Support Files

- `OpenFold_Comprehensive_Analysis.ipynb` - Main notebook
- `visualize_intermediate_reps_utils.py` - All functions
- `MAXED_OUT_FEATURES.md` - Technical documentation
- `NOTEBOOK_SETUP.md` - Environment setup guide
- `requirements_visualization.txt` - Minimal dependencies

---

## Questions?

Check these files for details:
1. `MAXED_OUT_FEATURES.md` - Full technical specs
2. `COMPREHENSIVE_ANALYSIS_GUIDE.md` - Function reference
3. `NOTEBOOK_SETUP.md` - Setup troubleshooting

---

**Ready to visualize ALL 48 layers?**

1. Refresh browser
2. Restart kernel
3. Run all cells
4. Wait ~15 minutes
5. Enjoy 77 MB of beautiful visualizations!

🚀 **LET'S GO!**

