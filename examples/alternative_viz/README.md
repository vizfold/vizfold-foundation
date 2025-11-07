Alternative Attention Visualization Examples
===========================================

Contents
--------

- `mock_attention/msa_row_attn_layer0.txt` — toy attention dump used to demo the heatmap generator.

Usage
-----

Run the heatmap utility with the mock file to produce an example figure:

```bash
python visualize_attention_heatmap_demo_utils.py \
  --attention-dir examples/alternative_viz/mock_attention \
  --output-dir outputs/alternative_viz_mock \
  --protein MOCK --layer-idx 0 --attention-type msa_row
```

The script will emit `outputs/alternative_viz_mock/msa_row/attention_heatmap_layer_0_MOCK.png` showing the per-head aggregated residue attention profile.

