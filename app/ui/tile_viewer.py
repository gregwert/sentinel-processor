"""
ui/tile_viewer.py

Streamlit component for browsing and exporting chipped tiles.
Renders a paginated thumbnail gallery of chips produced by the chipping
module and provides format selection and download controls.
"""

import streamlit as st
import numpy as np
from PIL import Image

from chipping.gdal_chipper import get_chip
from chipping.tile_exporter import export_chips, zip_export

COLS_PER_ROW = 4

_FORMAT_DESCRIPTIONS = {
    "png": "PNG — visual export, no georeferencing",
    "jpeg": "JPEG — compressed visual, no georeferencing",
    "geotiff": "GeoTIFF — full georeferencing preserved",
    "npy": "NPY — numpy array for ML pipelines",
}


def render_tile_viewer(grid, page_size: int = 16):
    """Render paginated chip thumbnail gallery in Streamlit."""
    if "tile_page" not in st.session_state:
        st.session_state["tile_page"] = 0
    page = st.session_state["tile_page"]

    total = grid.total
    total_pages = max((total + page_size - 1) // page_size, 1)

    # Clamp page to valid range
    page = max(0, min(page, total_pages - 1))
    st.session_state["tile_page"] = page

    start = page * page_size
    end = min(start + page_size, total)

    st.subheader(f"Chip Viewer — {total} chips ({grid.n_rows} rows × {grid.n_cols} cols)")

    # Pagination controls
    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Prev", disabled=(page == 0)):
            st.session_state["tile_page"] -= 1
            st.rerun()
    with col_info:
        st.markdown(f"Page {page + 1} of {total_pages}")
    with col_next:
        if st.button("Next →", disabled=(page == total_pages - 1)):
            st.session_state["tile_page"] += 1
            st.rerun()

    # Display grid
    cols = st.columns(COLS_PER_ROW)
    for i, chip_idx in enumerate(range(start, end)):
        chip_arr, chip_meta = get_chip(grid, chip_idx)
        # Downsample to thumbnail for display performance
        thumb = Image.fromarray(chip_arr).resize((128, 128), Image.LANCZOS)
        row_i = chip_meta["row_idx"]
        col_i = chip_meta["col_idx"]
        col = cols[i % COLS_PER_ROW]
        with col:
            st.image(thumb, caption=f"r{row_i:04d}_c{col_i:04d}", use_column_width=True)
            with st.expander("Full res"):
                st.image(chip_arr, use_column_width=True)


def render_export_controls(grid) -> "bytes | None":
    """Render format selector + export button; return zip bytes when triggered."""
    st.subheader("Export Chips")

    col_fmt, col_naming = st.columns(2)
    with col_fmt:
        fmt = st.selectbox("Format", ["png", "jpeg", "geotiff", "npy"], key="export_fmt")
    with col_naming:
        naming = st.selectbox("Naming", ["rowcol", "coords"], key="export_naming")

    st.caption(_FORMAT_DESCRIPTIONS.get(fmt, ""))

    if st.button("Export All Chips", type="primary"):
        with st.spinner(f"Exporting {grid.total} chips as {fmt.upper()}..."):
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                paths = export_chips(grid, tmp_dir, fmt=fmt, naming=naming)
                zip_bytes = zip_export(paths)
        return zip_bytes

    return None
