"""
ui/tile_viewer.py

Streamlit component for browsing chipped tiles.
Renders a paginated thumbnail gallery of chips produced by the chipping
module.
"""

import streamlit as st

from chipping.gdal_chipper import get_chip

COLS_PER_ROW = 4
DEFAULT_PAGE_SIZE = 16


def render_tile_viewer(grid, page_size: int = DEFAULT_PAGE_SIZE) -> None:
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
    """
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
        row_i = chip_meta["row_idx"]
        col_i = chip_meta["col_idx"]
        col = cols[i % COLS_PER_ROW]
        with col:
            st.image(chip_arr, caption=f"r{row_i:04d}_c{col_i:04d}", use_column_width=True)
