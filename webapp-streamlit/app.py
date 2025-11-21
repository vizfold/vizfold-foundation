import os
import re
from pathlib import Path
from typing import Dict, List

import streamlit as st


DEFAULT_DATA_ROOT = Path(os.environ.get("ATTENTION_OUTPUT_ROOT", "outputs"))


MSA_ROW_PATTERN = re.compile(r"msa_row_head_(?P<head>\d+)_layer_(?P<layer>\d+)_(?P<protein>[A-Za-z0-9_]+)(?P<suffix>_arc|_combo)?\.png$")
TRI_3D_PATTERN = re.compile(r"tri_start_residue_(?P<res>\d+)_head_(?P<head>\d+)_layer_(?P<layer>\d+)_(?P<protein>[A-Za-z0-9_]+)\.png$")
TRI_ARC_PATTERN = re.compile(r"tri_start_res_(?P<res>\d+)_head_(?P<head>\d+)_layer_(?P<layer>\d+)_(?P<protein>[A-Za-z0-9_]+)_arc\.png$")
TRI_COMBO_PATTERN = re.compile(r"tri_start_residue_(?P<res>\d+)_head_(?P<head>\d+)_layer_(?P<layer>\d+)_(?P<protein>[A-Za-z0-9_]+)_combo\.png$")


def _scan_msa_row(root: Path) -> Dict[str, Dict[int, Dict[int, Dict[str, Path]]]]:
    assets: Dict[str, Dict[int, Dict[int, Dict[str, Path]]]] = {}
    for png in root.rglob("msa_row_head_*_layer_*_*.png"):
        match = MSA_ROW_PATTERN.search(png.name)
        if not match:
            continue
        head = int(match.group("head"))
        layer = int(match.group("layer"))
        protein = match.group("protein")
        suffix = match.group("suffix")

        assets.setdefault(protein, {}).setdefault(layer, {}).setdefault(head, {})
        key = "arc" if suffix == "_arc" else "combo" if suffix == "_combo" else "3d"
        assets[protein][layer][head][key] = png

    return assets


def _scan_triangle_start(root: Path):
    assets: Dict[str, Dict[int, Dict[int, Dict[int, Dict[str, Path]]]]] = {}
    for png in root.rglob("tri_start_*_layer_*_*.png"):
        name = png.name
        match_3d = TRI_3D_PATTERN.search(name)
        match_arc = TRI_ARC_PATTERN.search(name)
        match_combo = TRI_COMBO_PATTERN.search(name)

        match = match_3d or match_arc or match_combo
        if not match:
            continue

        residue = int(match.group("res"))
        head = int(match.group("head"))
        layer = int(match.group("layer"))
        protein = match.group("protein")

        assets.setdefault(protein, {}).setdefault(layer, {}).setdefault(residue, {}).setdefault(head, {})

        if match_3d:
            assets[protein][layer][residue][head]["3d"] = png
        elif match_arc:
            assets[protein][layer][residue][head]["arc"] = png
        elif match_combo:
            assets[protein][layer][residue][head]["combo"] = png

    return assets


@st.cache_data(show_spinner=False)
def discover_assets(data_root: Path):
    data_root = data_root.expanduser().resolve()
    return {
        "msa_row": _scan_msa_row(data_root),
        "triangle_start": _scan_triangle_start(data_root),
    }


def _select_protein(options: List[str], label: str):
    if not options:
        st.warning("No rendered outputs found.")
        st.stop()
    return st.selectbox(label, options)


def _render_image_block(title: str, path: Path):
    if not path:
        st.info(f"No {title.lower()} available for this selection.")
        return
    st.markdown(f"### {title}")
    image_path = Path(path)
    st.image(image_path.read_bytes(), clamp=True, use_column_width=True)
    st.caption(str(image_path))


def main():
    st.set_page_config(page_title="OpenFold Attention Viewer", layout="wide")
    st.title("OpenFold Attention Viewer")
    st.write("Select a protein, layer, and head to preview saved attention renders.")

    data_root = Path(st.sidebar.text_input("Outputs root", str(DEFAULT_DATA_ROOT)))
    assets = discover_assets(data_root)

    attention_type = st.sidebar.radio("Attention type", ["msa_row", "triangle_start"])

    if attention_type == "msa_row":
        msa_assets = assets["msa_row"]
        protein = _select_protein(sorted(msa_assets.keys()), "Protein")
        layers = sorted(msa_assets[protein].keys())
        layer = st.selectbox("Layer", layers)
        heads = sorted(msa_assets[protein][layer].keys())
        head = st.selectbox("Head", heads)

        entry = msa_assets[protein][layer][head]
        col1, col2, col3 = st.columns(3)
        with col1:
            _render_image_block("3D", entry.get("3d"))
        with col2:
            _render_image_block("Arc Diagram", entry.get("arc"))
        with col3:
            _render_image_block("Combined Panel", entry.get("combo"))

    else:
        tri_assets = assets["triangle_start"]
        protein = _select_protein(sorted(tri_assets.keys()), "Protein")
        layers = sorted(tri_assets[protein].keys())
        layer = st.selectbox("Layer", layers)
        residues = sorted(tri_assets[protein][layer].keys())
        residue = st.selectbox("Residue", residues)
        heads = sorted(tri_assets[protein][layer][residue].keys())
        head = st.selectbox("Head", heads)

        entry = tri_assets[protein][layer][residue][head]
        st.markdown(f"#### Residue {residue}")
        col1, col2, col3 = st.columns(3)
        with col1:
            _render_image_block("3D", entry.get("3d"))
        with col2:
            _render_image_block("Arc Diagram", entry.get("arc"))
        with col3:
            _render_image_block("Combined Panel", entry.get("combo"))


if __name__ == "__main__":
    main()

