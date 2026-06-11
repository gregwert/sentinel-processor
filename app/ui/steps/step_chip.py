"""
Step 4 — Chip size configuration (pixels or metres), grid overlay preview,
run chipping to build ChipGrid.
"""
import streamlit as st
import numpy as np
from chipping.gdal_chipper import build_chip_grid, compute_chip_grid, ChipGrid
from ui.grid_overlay import render_grid_composite, chip_size_metres_to_pixels


def render(state: dict) -> bool:
    """Render chip step. Returns True when user clicks Next to advance."""

    col_back, col_spacer, col_skip = st.columns([1, 4, 1])
    with col_back:
        if st.button("← Back", key="back_chip"):
            for k in ["chip_grid", "chip_params", "chip_skipped"]:
                state.pop(k, None)
            state["step"] = "enhance"
            st.rerun()
    with col_skip:
        if st.button("Skip →", key="skip_chip"):
            state["chip_skipped"] = True
            state.pop("chip_grid", None)
            state["step"] = "review"
            st.rerun()

    st.header("Step 4 — Chipping")

    img_h, img_w = state["enhanced_image"].shape[:2]

    # Phase B — after chip_grid exists
    if "chip_grid" in state:
        grid = state["chip_grid"]
        st.success(f"Chipping complete — {grid.total} chips  ({grid.n_rows} rows × {grid.n_cols} cols)")

        cp = state.get("chip_params", {})
        overlay_img = render_grid_composite(
            state["enhanced_image"], grid.windows, img_w, img_h,
            chip_w=grid.chip_w, chip_h=grid.chip_h,
            overlap=cp.get("overlap", 0.0),
        )
        st.image(overlay_img, caption="Chip grid", use_column_width=True)

        if st.button("Re-chip", key="rechip"):
            state.pop("chip_grid", None)
            st.rerun()

        st.divider()

        # Inline tile viewer
        from ui.tile_viewer import render_tile_viewer
        render_tile_viewer(grid)

        st.divider()

        if st.button("Finalize →", type="primary", key="chip_finalize"):
            return True

        return False

    # Phase A — grid configuration and overlay (before chip_grid exists)
    unit = st.radio("Chip size unit", ["Pixels", "Metres"], horizontal=True)
    square = st.checkbox("Square chips", value=True)

    approx_warning = False

    if unit == "Pixels":
        if square:
            chip_w = st.number_input("Chip size (px)", 64, 2048, 256, 64)
            chip_h = chip_w
        else:
            chip_w = st.number_input("Chip width (px)", 64, 2048, 256, 64)
            chip_h = st.number_input("Chip height (px)", 64, 2048, 256, 64)
    else:
        if square:
            chip_size_m = st.number_input("Chip size (m)", 100.0, 50000.0, 1000.0, 100.0)
            chip_w, approx_warning = chip_size_metres_to_pixels(chip_size_m, state["source_meta"])
            chip_h = chip_w
        else:
            chip_w_m = st.number_input("Chip width (m)", 100.0, 50000.0, 1000.0, 100.0)
            chip_h_m = st.number_input("Chip height (m)", 100.0, 50000.0, 1000.0, 100.0)
            chip_w, approx_w = chip_size_metres_to_pixels(chip_w_m, state["source_meta"])
            chip_h, approx_h = chip_size_metres_to_pixels(chip_h_m, state["source_meta"])
            approx_warning = approx_w or approx_h
        st.caption(f"≈ {chip_w} × {chip_h} px")
        if approx_warning:
            st.warning("CRS is geographic — metre conversion is approximate.")

    overlap = st.slider("Overlap fraction", 0.0, 0.90, 0.0, 0.05)
    edge_mode = st.radio(
        "Edge chip handling",
        ["pad", "overlap"],
        format_func=lambda x: {
            "pad": "Pad with black (preserve exact grid)",
            "overlap": "Overlap with adjacent (no black borders)",
        }[x],
        horizontal=True,
        help="'Pad' fills edge chips that don't fit fully with black pixels. "
             "'Overlap' shifts edge chips inward so they fully overlap with their neighbour — all chips are full size, no black borders.",
    )
    naming = st.selectbox("Chip naming", ["coords", "rowcol"])

    windows = compute_chip_grid(img_w, img_h, int(chip_w), int(chip_h), overlap, edge_mode)
    n_cols = len([w for w in windows if w[1] == 0])
    n_rows = len(set(w[1] for w in windows))
    st.caption(f"Grid: {n_rows} rows × {n_cols} cols = {len(windows)} chips")

    overlay_img = render_grid_composite(
        state["enhanced_image"], windows, img_w, img_h,
        chip_w=int(chip_w), chip_h=int(chip_h), overlap=overlap,
    )
    st.image(overlay_img, caption="Chip grid preview", use_column_width=True)

    if st.button("Run Chipping", type="primary"):
        with st.spinner(f"Building {len(windows)} chips..."):
            grid = build_chip_grid(
                state["enhanced_image"],
                state["source_meta"],
                int(chip_w),
                int(chip_h),
                overlap,
                edge_mode,
            )
        state.pop("chip_skipped", None)
        state["chip_grid"] = grid
        state["chip_params"] = {
            "unit": unit,
            "square": square,
            "chip_w": int(chip_w),
            "chip_h": int(chip_h),
            "overlap": overlap,
            "naming": naming,
            "edge_mode": edge_mode,
        }
        st.rerun()

    return False
