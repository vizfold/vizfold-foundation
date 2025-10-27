# 📓 Jupyter Notebook Setup Guide

## OpenFold Comprehensive Analysis Notebook

This guide helps you set up the environment to run `OpenFold_Comprehensive_Analysis.ipynb`.

---

## 🚀 Quick Start Options

### Option 1: Use Existing OpenFold Environment (Recommended if you have OpenFold)

If you already have OpenFold installed:

```bash
# Activate OpenFold environment
conda activate openfold-env

# Install Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name=openfold-env

# Launch notebook
jupyter notebook OpenFold_Comprehensive_Analysis.ipynb
```

---

### Option 2: Minimal Visualization Environment (No OpenFold needed)

For visualization-only (no real OpenFold inference):

```bash
# Create new environment
conda create -n openfold-viz python=3.10
conda activate openfold-viz

# Install dependencies
conda install pytorch numpy matplotlib scipy jupyter -c pytorch
pip install ipykernel

# Register kernel
python -m ipykernel install --user --name=openfold-viz

# Launch notebook
jupyter notebook OpenFold_Comprehensive_Analysis.ipynb
```

---

### Option 3: Using pip (Alternative to conda)

```bash
# Create virtual environment
python3.10 -m venv openfold-viz-env
source openfold-viz-env/bin/activate  # On Windows: openfold-viz-env\Scripts\activate

# Install requirements
pip install -r requirements_visualization.txt

# Register kernel
python -m ipykernel install --user --name=openfold-viz-env

# Launch notebook
jupyter notebook OpenFold_Comprehensive_Analysis.ipynb
```

---

### Option 4: Full OpenFold Environment (For complete integration)

```bash
# Clone OpenFold (if you haven't)
git clone https://github.com/aqlaboratory/openfold.git
cd openfold

# Create environment
conda env create -f environment.yml
conda activate openfold-env

# Install OpenFold
pip install -e .

# Install Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name=openfold-env

# Launch notebook
jupyter notebook OpenFold_Comprehensive_Analysis.ipynb
```

---

## 📦 Required Packages

### Core Dependencies:
- **Python**: 3.10+
- **PyTorch**: 2.0+ (2.5 recommended)
- **NumPy**: 1.23+
- **Matplotlib**: 3.5+
- **SciPy**: 1.9+ (for clustering)
- **Jupyter**: Any recent version

### Optional but Recommended:
- **CUDA**: For GPU acceleration (if available)
- **Pillow**: For image processing

---

## 🔍 Verify Installation

After setup, run this in a Python shell to verify:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ NumPy: {np.__version__}")
print(f"✓ Matplotlib: {plt.matplotlib.__version__}")
print(f"✓ SciPy: {scipy.__version__}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
```

---

## 📂 Required Files

Make sure these files are in the same directory:

```
attention-jay-venn/
├── OpenFold_Comprehensive_Analysis.ipynb  ← The notebook
├── visualize_intermediate_reps_utils.py   ← Visualization functions
├── environment.yml                         ← Full OpenFold environment
├── requirements_visualization.txt          ← Minimal requirements
└── NOTEBOOK_SETUP.md                      ← This file
```

---

## 🎯 Running the Notebook

### Step 1: Select Kernel

After opening the notebook:
1. Click **Kernel** → **Change Kernel**
2. Select `openfold-env` or `openfold-viz` or your environment name

### Step 2: Run Cells

**Run all cells:**
- Click **Cell** → **Run All**

**Run cell by cell:**
- Press `Shift + Enter` to run current cell and move to next

### Step 3: View Results

All visualizations will appear inline in the notebook!

---

## 🐛 Troubleshooting

### Issue: "No module named visualize_intermediate_reps_utils"

**Solution:** Make sure `visualize_intermediate_reps_utils.py` is in the same directory as the notebook.

```bash
# Check files
ls -la
# Should see: OpenFold_Comprehensive_Analysis.ipynb
#             visualize_intermediate_reps_utils.py
```

### Issue: "No module named torch"

**Solution:** PyTorch not installed. Install it:

```bash
# Conda
conda install pytorch -c pytorch

# Pip
pip install torch
```

### Issue: "No module named scipy"

**Solution:** SciPy needed for clustering:

```bash
# Conda
conda install scipy

# Pip
pip install scipy
```

### Issue: Kernel not found

**Solution:** Install ipykernel and register:

```bash
pip install ipykernel
python -m ipykernel install --user --name=YOUR_ENV_NAME
```

### Issue: Visualizations not showing

**Solution:** Make sure matplotlib inline magic is executed:

```python
%matplotlib inline
import matplotlib.pyplot as plt
```

---

## 💡 Usage Tips

### Memory Management

The notebook analyzes 48 layers with large tensors. If you run into memory issues:

1. **Reduce n_res**: Change `n_res = 100` to `n_res = 50`
2. **Use stratified sampling**: Already enabled by default
3. **Restart kernel**: **Kernel** → **Restart & Clear Output**

### Performance

- **CPU**: Works fine but slower (~5 minutes total)
- **GPU**: Much faster (~2 minutes total)
- **Stratified sampling**: Uses ~25% of full memory

### Customization

You can easily modify parameters:
- `n_seq`: Number of MSA sequences
- `n_res`: Number of residues
- `n_layers`: Number of layers (48 for OpenFold)
- `sampled_layers`: Which layers to visualize

---

## 📊 Expected Output

After running all cells, you'll get:

**10 Visualization Types:**
1. Multi-layer evolution (48 layers)
2. Stratified MSA comparison
3. Stratified Pair comparison
4. MSA convergence analysis
5. Pair convergence analysis
6. Layer importance ranking
7. Structure module evolution
8. Residue feature analysis
9. Hierarchical clustering
10. Contact map integration

**All saved to:** `notebook_outputs/`

---

## 🎓 Learning Path

### Beginner
- Run all cells to see what's possible
- Read the output explanations
- Try changing residue indices

### Intermediate
- Modify sampling strategies
- Change layer indices
- Experiment with different metrics

### Advanced
- Integrate with real OpenFold
- Add custom visualizations
- Batch process multiple proteins

---

## 🤝 Getting Help

### Check Documentation:
- `COMPREHENSIVE_ANALYSIS_GUIDE.md` - Complete function reference
- `ACHIEVEMENTS_SUMMARY.md` - System overview
- Inline code comments

### Common Commands:

```bash
# Check Python version
python --version

# List conda environments
conda env list

# Check installed packages
pip list | grep -E "torch|numpy|matplotlib|scipy"

# Test imports
python -c "import torch, numpy, matplotlib, scipy; print('OK')"
```

---

## ✅ Checklist

Before running the notebook, verify:

- [ ] Python 3.10+ installed
- [ ] PyTorch installed
- [ ] NumPy, Matplotlib, SciPy installed
- [ ] Jupyter installed
- [ ] `visualize_intermediate_reps_utils.py` in same directory
- [ ] Kernel registered and selected
- [ ] Environment activated

---

## 🎉 Ready to Start!

Once setup is complete:

```bash
jupyter notebook OpenFold_Comprehensive_Analysis.ipynb
```

Then **Cell** → **Run All** and watch the magic happen! 🧬✨

---

**For questions or issues, see the comprehensive guides in the repository.**

