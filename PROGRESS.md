# Progress Log for Issue #8: Intermediate Representation Visualization

## Project Goal
Add functionality to extract and visualize intermediate representations (MSA, Pair, Structure module) from OpenFold's 48-layer neural network.

---

## Week 1: Setup and Initial Implementation

### ✅ Completed
- [x] Forked repository to `jayvenn21/attention-jay-venn`
- [x] Created feature branch: `feature/intermediate-rep-viz`
- [x] Set up local development environment
- [x] Created initial file structure:
  - `visualize_intermediate_reps_utils.py` - Main utility functions (skeleton)
  - `tests/test_intermediate_extraction.py` - Unit test framework
  - `PROGRESS.md` - This progress log

### 🚧 In Progress
- [ ] Explore OpenFold codebase to identify extraction points
- [ ] Implement MSA extraction function
- [ ] Implement Pair extraction function
- [ ] Create first visualization (MSA heatmap)

### 📋 Next Steps
1. Study existing attention extraction code in `visualize_attention_general_utils.py`
2. Identify where MSA and Pair tensors are stored in `openfold/model/evoformer.py`
3. Implement extraction hooks
4. Create simple heatmap visualization
5. Test on example protein (6KWC)

---

## Week 2: Visualization and Demo (Planned)

### 📋 TODO
- [ ] Complete all visualization functions
- [ ] Create demo notebook: `notebooks/viz_intermediate_reps_demo.ipynb`
- [ ] Test on multiple proteins
- [ ] Add documentation and docstrings
- [ ] Create example outputs

---

## Week 3: Testing and PR Finalization (Planned)

### 📋 TODO
- [ ] Complete unit tests
- [ ] Update main README with usage instructions
- [ ] Clean up code and add comments
- [ ] Run full test suite
- [ ] Move PR from Draft to Ready for Review
- [ ] Peer review 2 other PRs

---

## Questions / Blockers

### Open Questions
1. Which layers are most informative to visualize? (early/middle/late)
2. Should we aggregate over channel dimensions or provide channel-specific views?
3. How to handle memory for large proteins?

### Blockers
- None yet

---

## Key Files to Study

### Existing Codebase
- `visualize_attention_general_utils.py` - Template for extraction logic
- `visualize_attention_3d_demo_utils.py` - 3D visualization patterns
- `viz_attention_demo_base.ipynb` - Notebook structure to follow
- `openfold/model/evoformer.py` - Where MSA/Pair representations live
- `openfold/model/structure_module.py` - Structure module outputs
- `run_pretrained_openfold.py` - Inference script to modify

### New Files Created
- `visualize_intermediate_reps_utils.py` - Main implementation
- `tests/test_intermediate_extraction.py` - Unit tests
- `notebooks/viz_intermediate_reps_demo.ipynb` - Demo (to be created)

---

## Useful Commands

```bash
# Run tests
pytest tests/test_intermediate_extraction.py -v

# Run OpenFold inference (modify this to save intermediate reps)
python run_pretrained_openfold.py \
    --config_preset model_1_ptm \
    --model_device cuda:0 \
    --output_dir outputs/ \
    --fasta_paths examples/monomer/6KWC.fasta

# Start Jupyter
jupyter notebook notebooks/viz_intermediate_reps_demo.ipynb
```

---

## Updates

### October 1, 2025
- Initial repository setup complete
- File structure created
- Ready to start implementation

<!-- Add more updates as you progress -->

