"""
ui/tile_viewer.py

Streamlit component for browsing chipped tiles.
Renders a paginated thumbnail gallery of chips produced by the chipping
module.
"""

import numpy as np
import streamlit as st

from chipping.gdal_chipper import get_chip

COLS_PER_ROW = 4
DEFAULT_PAGE_SIZE = 16


def _tint_chip(chip_arr: np.ndarray, colour: tuple = (200, 0, 0), alpha: float = 0.35) -> np.ndarray:
    """Blend a solid colour over chip_arr to visually mark it as rejected."""
    overlay = np.full_like(chip_arr, colour, dtype=np.float32)
    blended = chip_arr.astype(np.float32) * (1 - alpha) + overlay * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def render_tile_viewer(grid, page_size: int = DEFAULT_PAGE_SIZE, quality: dict | None = None) -> None:
    """Render a paginated chip thumbnail gallery in Streamlit.

    Parameters
    ----------
    grid : object
        Chip grid descriptor returned by the chipping module. Expected to
        expose ``grid.total`` (int), ``grid.n_rows`` (int), and
        ``grid.n_cols`` (int).
    page_size : int, optional
        Number of chip thumbnails to display per page.
        Default :data:`DEFAULT_PAGE_SIZE` (16).
    quality : dict or None, optional
        When provided, controls quality-filter overlays. Expected keys:

        accepted : list[int]
            Chip indices that passed all filters.
        rejected : list[int]
            Chip indices that failed at least one filter.
        show_rejected : bool
            If True, rejected chips are shown with a red tint.
            If False, rejected chips are hidden from the viewer.

        Default is None (no filtering applied).
    """
    if "tile_page" not in st.session_state:
        st.session_state["tile_page"] = 0
    page = st.session_state["tile_page"]

    # Determine the list of chip indices to display
    if quality is not None and quality.get("rejected"):
        rejected_set = set(quality["rejected"])
        show_rejected = quality.get("show_rejected", False)
        if show_rejected:
            display_indices = list(range(grid.total))
        else:
            display_indices = quality["accepted"]
    else:
        rejected_set = set()
        show_rejected = False
        display_indices = list(range(grid.total))

    total_display = len(display_indices)
    total_pages = max((total_display + page_size - 1) // page_size, 1)

    # Clamp page to valid range
    page = max(0, min(page, total_pages - 1))
    st.session_state["tile_page"] = page

    start = page * page_size
    end = min(start + page_size, total_display)

    st.subheader(f"Chip Viewer — {grid.total} chips ({grid.n_rows} rows × {grid.n_cols} cols)")

    # Show accepted/rejected summary when quality filtering is active
    if quality is not None:
        n_rejected = len(quality.get("rejected", []))
        if n_rejected > 0:
            st.caption(f"{grid.total - n_rejected} accepted · {n_rejected} rejected")

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
    page_chip_indices = display_indices[start:end]
    for i, chip_idx in enumerate(page_chip_indices):
        chip_arr, chip_meta = get_chip(grid, chip_idx)
        row_i = chip_meta["row_idx"]
        col_i = chip_meta["col_idx"]
        col = cols[i % COLS_PER_ROW]
        with col:
            if show_rejected and chip_idx in rejected_set:
                chip_arr = _tint_chip(chip_arr)
            st.image(chip_arr, caption=f"r{row_i:04d}_c{col_i:04d}", use_column_width=True)
