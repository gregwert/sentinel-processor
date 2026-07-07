"""
Step 4 — Chip size configuration (pixels or metres), grid overlay preview,
run chipping to build ChipGrid.
"""
import streamlit as st
from sentinel_frontend import api_client as api

from sentinel_frontend.ui.steps import render_step_nav


def render(state: dict) -> bool:
    """Render Step 4 — Chipping.

    Reads from state
    ----------------
    chip_params : dict
        Parameters from the previous Run Chipping call (if any), used
        to seed widget defaults.
    chip_grid_spec : dict
        Grid spec returned by the backend (if any). When present the step
        enters Phase B (result display) instead of Phase A (configuration).
    chip_quality : dict
        Accepted/rejected chip indices from the last filter run (if any).
    chip_page : int
        Current page index in the chip viewer.

    Writes to state
    ---------------
    chip_params : dict
        Grid dimensions, overlap fraction, naming scheme, and edge mode.
    chip_grid_spec : dict
        {total, n_rows, n_cols} returned by the backend on Run Chipping.
    chip_page : int
        Updated by the Previous/Next page buttons in the chip viewer.
    chip_quality : dict
        Accepted/rejected chip indices from the active quality filters.
    chip_skipped : bool
        Set to True on Skip click.
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
        for k in ["chip_grid", "chip_params", "chip_skipped", "chip_quality", "chip_grid_spec"]:
            state.pop(k, None)
        state["step"] = "enhance"
        st.rerun()
    if skip:
        state["chip_skipped"] = True
        state.pop("chip_grid", None)
        state.pop("chip_grid_spec", None)
        state.pop("chip_quality", None)
        state["step"] = "review"
        st.rerun()

    st.header("Step 4 — Chipping")

    session_id = state["_session_id"]

    # Phase B — after chip_grid_spec exists (chipping has been run)
    if "chip_grid_spec" in state:
        spec = state["chip_grid_spec"]
        total = spec.get("total", "?")
        n_rows = spec.get("n_rows", "?")
        n_cols = spec.get("n_cols", "?")
        st.success(f"Chipping complete — {total} chips  ({n_rows} rows × {n_cols} cols)")

        with st.expander("Quality Filters", expanded=False):
            enable_cloud = st.toggle("Cloud coverage filter", value=False, key="qf_cloud_enable")
            cloud_thresh = st.slider("Max cloud fraction", 0.0, 1.0, 0.30, 0.01,
                                     key="qf_cloud_thresh", disabled=not enable_cloud)
            st.caption("*Reject chips where more than this fraction of pixels are detected as cloud.*")
            enable_var = st.toggle("Variance filter", value=False, key="qf_var_enable")
            var_thresh = st.slider("Min pixel variance", 0.0, 2000.0, 100.0, 10.0,
                                   key="qf_var_thresh", disabled=not enable_var)
            st.caption("*Reject chips with very low pixel variance — catches black-border padding and featureless areas.*")

        if enable_cloud or enable_var:
            try:
                filter_result = api.run_chip_filters(
                    session_id,
                    cloud_enabled=enable_cloud,
                    cloud_thresh=cloud_thresh,
                    variance_enabled=enable_var,
                    variance_thresh=var_thresh,
                )
                state["chip_quality"] = filter_result
                accepted = filter_result.get("accepted", [])
                rejected = filter_result.get("rejected", [])
                if rejected:
                    st.caption(f"Filters active: {len(accepted)} accepted · {len(rejected)} rejected")
            except Exception as exc:
                st.error(f"Chip filters failed: {exc}")
        else:
            state.pop("chip_quality", None)

        # TODO: wire up grid-overlay preview endpoint when available
        # For now just show the best available image
        st.image(
            api.fetch_best_preview(session_id),
            caption="Processed image (grid overlay pending backend support)",
            use_container_width=True,
        )

        if st.button("Re-chip", key="rechip"):
            state.pop("chip_grid_spec", None)
            state.pop("chip_quality", None)
            st.rerun()

        st.divider()

        # Tile viewer via API pagination
        st.subheader("Chip Viewer")
        include_rej = state.get("chip_quality", {}).get("rejected") is not None
        show_rejected = st.checkbox("Show rejected chips", value=False, key="qf_show_rejected")
        page_size = 16
        if "chip_page" not in state:
            state["chip_page"] = 0
        col_prev, col_next = st.columns(2)
        if col_prev.button("← Previous page") and state["chip_page"] > 0:
            state["chip_page"] -= 1
            st.rerun()
        try:
            chip_data = api.list_chips(
                session_id,
                page=state["chip_page"],
                page_size=page_size,
                include_rejected=show_rejected and include_rej,
            )
            items = chip_data.get("items", [])
            total_chips = chip_data.get("total", 0)
            total_pages = max(1, (total_chips + page_size - 1) // page_size)
            if col_next.button("Next page →") and state["chip_page"] < total_pages - 1:
                state["chip_page"] += 1
                st.rerun()
            st.caption(f"Page {state['chip_page'] + 1} of {total_pages}")
            cols = st.columns(4)
            for i, item in enumerate(items):
                try:
                    thumb_bytes = api.fetch_thumbnail(session_id, item["index"])
                    cols[i % 4].image(thumb_bytes, caption=f"r{item['row']} c{item['col']}", use_container_width=True)
                except Exception:
                    cols[i % 4].caption(f"Chip {item['index']}")
        except Exception as exc:
            st.error(f"Failed to load chip thumbnails: {exc}")

        st.divider()

        if st.button("Finalize →", type="primary", key="chip_finalize"):
            return True

        return False

    # Phase A — grid configuration (API path)
    _cp = state.get("chip_params") or {}

    unit = st.radio("Chip size unit", ["Pixels", "Metres"], horizontal=True)
    st.caption("*Metre-based sizing requires the backend to resolve pixel dimensions — use Pixels for predictable results.*")
    square = st.checkbox("Square chips", value=True)

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
            chip_w = chip_h = int(chip_size_m)  # backend resolves to pixels
        else:
            chip_w = int(st.number_input("Chip width (m)", 100.0, 50000.0, 1000.0, 100.0))
            chip_h = int(st.number_input("Chip height (m)", 100.0, 50000.0, 1000.0, 100.0))
        st.caption("Metre values will be resolved to pixels by the backend.")

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
             "'Overlap' shifts edge chips inward so they fully overlap with their neighbour.",
    )

    _naming_options = ["coords", "rowcol"]
    naming = st.selectbox(
        "Chip naming",
        _naming_options,
        index=_naming_options.index(_cp.get("naming", "coords")),
    )

    # Live chip-grid spec preview from backend
    _grid_params_key = str((chip_w, chip_h, overlap, edge_mode, naming))
    if state.get("_chip_grid_params_key") != _grid_params_key:
        try:
            spec_result = api.put_chip_grid(
                session_id, int(chip_w), int(chip_h), overlap, edge_mode, naming
            )
            state["_chip_grid_spec_preview"] = spec_result
            state["_chip_grid_params_key"] = _grid_params_key
        except Exception as exc:
            st.warning(f"Grid preview unavailable: {exc}")
            spec_result = {}
    else:
        spec_result = state.get("_chip_grid_spec_preview", {})

    if spec_result:
        n_r = spec_result.get("n_rows", "?")
        n_c = spec_result.get("n_cols", "?")
        tot = spec_result.get("total", "?")
        st.caption(f"Grid: {n_r} rows × {n_c} cols = {tot} chips")

    # TODO: grid overlay preview endpoint not yet in backend — show best available image
    st.image(
        api.fetch_best_preview(session_id),
        caption="Processed image (grid overlay pending backend support)",
        use_container_width=True,
    )

    if st.button("Run Chipping", type="primary"):
        try:
            spec_result = api.put_chip_grid(
                session_id, int(chip_w), int(chip_h), overlap, edge_mode, naming
            )
            state.pop("chip_skipped", None)
            state["chip_grid_spec"] = spec_result
            state["chip_page"] = 0
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
        except Exception as exc:
            st.error(f"Chipping failed: {exc}")

    return False