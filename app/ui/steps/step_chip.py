"""
Step 4 — Chip size configuration (pixels or metres), grid overlay preview,
run chipping to build ChipGrid.
"""
import streamlit as st
from chipping.gdal_chipper import build_chip_grid, compute_chip_grid
from ui.grid_overlay import render_grid_composite, chip_size_metres_to_pixels
from ui.steps import render_step_nav


def render(state: dict) -> bool:
    """Render Step 4 — Chipping.

    Reads from state
    ----------------
    enhanced_image : np.ndarray
        uint8 (H, W, 3) image produced by step_enhance (or copied from
        dehazed_image when enhancement was skipped). Used as the chip
        source and for grid preview rendering.
    source_meta : dict
        Rasterio metadata dict with CRS and Affine transform. Used for
        metres-to-pixels conversion and written into chip GeoTIFFs.
    chip_grid : ChipGrid
        Previously built chip grid (if any). When present the step enters
        Phase B (result display) instead of Phase A (configuration).
    chip_params : dict
        Parameters from the previous Run Chipping call (if any). Used to
        reconstruct the overlap value for the grid overlay in Phase B.

    Writes to state
    ---------------
    chip_grid : ChipGrid
        Built on Run Chipping click. Removed on Back click, Skip click,
        and Re-chip click.
    chip_params : dict
        Grid dimensions, overlap fraction, naming scheme, and edge mode
        used for the last Run Chipping call.
    chip_skipped : bool
        Set to True on Skip click. Removed on Run Chipping click.
    step : str
        Set to 'enhance' on Back click, or 'review' on Skip click.

    Returns
    -------
    bool
        True when the user clicks 'Finalize →' (Phase B) to advance to
        the review step. False on all other renders.
    """
    back, skip = render_step_nav("chip")
    if back:
        for k in ["chip_grid", "chip_params", "chip_skipped", "chip_quality"]:
            state.pop(k, None)
        state["step"] = "enhance"
        st.rerun()
    if skip:
        state["chip_skipped"] = True
        state.pop("chip_grid", None)
        state.pop("chip_quality", None)
        state["step"] = "review"
        st.rerun()

    st.header("Step 4 — Chipping")

    img_h, img_w = state["enhanced_image"].shape[:2]

    # Phase B — after chip_grid exists
    if "chip_grid" in state:
        grid = state["chip_grid"]
        st.success(f"Chipping complete — {grid.total} chips  ({grid.n_rows} rows × {grid.n_cols} cols)")

        from chipping.chip_filter import apply_chip_filters

        with st.expander("Quality Filters", expanded=False):
            enable_cloud = st.toggle("Cloud coverage filter", value=False, key="qf_cloud_enable")
            if enable_cloud and "cloud_mask" not in state:
                st.warning("No cloud mask available — run dehazing with cloud masking enabled.")
            cloud_thresh = st.slider("Max cloud fraction", 0.0, 1.0, 0.30, 0.01,
                                     key="qf_cloud_thresh", disabled=not enable_cloud)
            st.caption("*Reject chips where more than this fraction of pixels are detected as cloud.*")
            enable_var = st.toggle("Variance filter", value=False, key="qf_var_enable")
            var_thresh = st.slider("Min pixel variance", 0.0, 2000.0, 100.0, 10.0,
                                   key="qf_var_thresh", disabled=not enable_var)
            st.caption("*Reject chips with very low pixel variance — catches black-border padding and featureless areas.*")
            show_rejected = st.checkbox("Show rejected chips in tile viewer", value=False,
                                        key="qf_show_rejected")

        if enable_cloud or enable_var:
            accepted, rejected, chip_stats = apply_chip_filters(
                state["chip_grid"],
                cloud_mask=state.get("cloud_mask"),
                cloud_thresh=cloud_thresh,
                variance_thresh=var_thresh,
                enable_cloud_filter=enable_cloud,
                enable_variance_filter=enable_var,
            )
            state["chip_quality"] = {
                "accepted": accepted,
                "rejected": rejected,
                "stats": chip_stats,
                "cloud_thresh": cloud_thresh,
                "variance_thresh": var_thresh,
                "enable_cloud_filter": enable_cloud,
                "enable_variance_filter": enable_var,
                "show_rejected": show_rejected,
            }
            if rejected:
                st.caption(f"Filters active: {len(accepted)} accepted · {len(rejected)} rejected")
        else:
            state.pop("chip_quality", None)

        cp = state.get("chip_params", {})
        overlay_img = render_grid_composite(
            state["enhanced_image"], grid.windows, img_w, img_h,
            chip_w=grid.chip_w, chip_h=grid.chip_h,
            overlap=cp.get("overlap", 0.0),
        )
        st.image(overlay_img, caption="Chip grid", use_column_width=True)

        if st.button("Re-chip", key="rechip"):
            state.pop("chip_grid", None)
            state.pop("chip_quality", None)
            st.rerun()

        st.divider()

        # Inline tile viewer
        from ui.tile_viewer import render_tile_viewer
        render_tile_viewer(grid, quality=state.get("chip_quality"))

        st.divider()

        if st.button("Finalize →", type="primary", key="chip_finalize"):
            return True

        return False

    # Phase A — grid configuration and overlay (before chip_grid exists)
    _cp = state.get("chip_params") or {}

    unit = st.radio("Chip size unit", ["Pixels", "Metres"], horizontal=True)
    square = st.checkbox("Square chips", value=True)

    approx_warning = False

    # Only seed pixel dimensions from saved params when the saved unit also was Pixels
    _seed_pixels = _cp.get("unit") == "Pixels"

    if unit == "Pixels":
        if square:
            chip_w = st.number_input(
                "Chip size (px)", 64, 2048,
                _cp.get("chip_w", 256) if _seed_pixels else 256,
                64,
            )
            chip_h = chip_w
        else:
            chip_w = st.number_input(
                "Chip width (px)", 64, 2048,
                _cp.get("chip_w", 256) if _seed_pixels else 256,
                64,
            )
            chip_h = st.number_input(
                "Chip height (px)", 64, 2048,
                _cp.get("chip_h", 256) if _seed_pixels else 256,
                64,
            )
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

    overlap = st.slider("Overlap fraction", 0.0, 0.90, _cp.get("overlap", 0.0), 0.05)

    _edge_options = ["pad", "overlap"]
    edge_mode = st.radio(
        "Edge chip handling",
        _edge_options,
        index=_edge_options.index(_cp.get("edge_mode", "pad")),
        format_func=lambda x: {
            "pad": "Pad with black (preserve exact grid)",
            "overlap": "Overlap with adjacent (no black borders)",
        }[x],
        horizontal=True,
        help="'Pad' fills edge chips that don't fit fully with black pixels. "
             "'Overlap' shifts edge chips inward so they fully overlap with their neighbour — all chips are full size, no black borders.",
    )

    _naming_options = ["coords", "rowcol"]
    naming = st.selectbox(
        "Chip naming",
        _naming_options,
        index=_naming_options.index(_cp.get("naming", "coords")),
    )

    windows = compute_chip_grid(img_w, img_h, int(chip_w), int(chip_h), overlap, edge_mode)
    # n_cols: windows whose row_off==0 are all in the first row, one per column.
    n_cols = len([w for w in windows if w[1] == 0])
    # n_rows: unique row offsets. In 'overlap' edge mode the last row may be
    # clamped to the same offset as its predecessor, so this reflects the actual
    # deduplicated grid rather than a theoretical count — intentional.
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
