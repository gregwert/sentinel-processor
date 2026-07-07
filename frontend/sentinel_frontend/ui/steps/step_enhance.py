"""
Step 3 — CLAHE enhancement on full image, before/after comparison.
Computes and stores global_stats (mean/std) for use at export time.
Standardisation has been removed from this step — it now lives in tile_exporter.py.
"""
import streamlit as st
from sentinel_frontend import api_client as api

from sentinel_frontend.ui.steps import render_step_nav


def render(state: dict) -> bool:
    """Render Step 3 — Enhancement.

    Reads from state
    ----------------
    enhance_params : dict
        Parameters from the previous Apply run (if any), used to seed
        widget defaults and detect stale-result conditions.
    global_stats : dict
        Global mean/std from the last Apply run (if any).
    reference_stats : dict
        Mean/std from reference TIFFs (if uploaded in step_upload).

    Writes to state
    ---------------
    enhance_params : dict
        Enhancement method and parameters used for the last run, or
        {'skipped': True} if the step was skipped.
    enhance_done : bool
        True once an enhance job has succeeded.
    global_stats : dict
        {'mean': [r, g, b], 'std': [r, g, b]} returned by the backend.
    step : str
        Set to 'dehaze' on Back click, or 'chip' on Skip click.

    Returns
    -------
    bool
        True when the user clicks 'Proceed to Chipping →' to advance
        to the chip step. False on all other renders.
    """
    back, skip = render_step_nav("enhance")
    if back:
        for k in ["enhanced_image", "global_stats", "enhance_params", "enhance_done"]:
            state.pop(k, None)
        state["step"] = "dehaze"
        st.rerun()
    if skip:
        state["enhance_params"] = {"skipped": True}
        state["global_stats"] = None
        state["step"] = "chip"
        st.rerun()

    st.header("Step 3 — Enhancement")

    _ep = state.get("enhance_params") or {}

    enable_gw = st.toggle("Gray World white balance", value=bool(_ep.get("gray_world", False)), key="enhance_gw_enable")
    if enable_gw:
        st.caption("*Scales RGB channels to equalise their means, correcting colour cast from atmospheric scattering. Cloud pixels from the dehaze step are excluded from the mean computation when available. May over-correct on spectrally skewed scenes (all-ocean, all-snow).*")

    # --- Reference normalisation ---
    _has_ref_stats = "reference_stats" in state
    enable_ref_norm = st.toggle(
        "Reference normalisation",
        value=bool(_ep.get("ref_norm", False)) and _has_ref_stats,
        disabled=not _has_ref_stats,
        key="enhance_ref_norm_enable",
    )
    if not _has_ref_stats:
        st.caption(
            "*Upload reference images in Step 1 to enable. Reference normalisation "
            "anchors the target image's radiometry to cloudless reference acquisitions, "
            "improving consistency across dates.*"
        )
    elif enable_ref_norm:
        _rn_options = ["Histogram matching", "Linear (mean/std)"]
        _saved_rn = _ep.get("ref_norm_method", "Histogram matching")
        ref_norm_method = st.radio(
            "Normalisation method",
            _rn_options,
            index=_rn_options.index(_saved_rn) if _saved_rn in _rn_options else 0,
            horizontal=True,
            key="enhance_ref_norm_method",
        )
        st.caption(
            "*Histogram matching shifts the full per-band histogram to match the "
            "reference — handles non-linear illumination differences. "
            "Linear (mean/std) is simpler and more robust when averaging across "
            "multiple references. When Gray World is also enabled it runs first; "
            "the two are somewhat redundant — reference norm alone is usually "
            "sufficient when a clean reference is available.*"
        )
    else:
        ref_norm_method = _ep.get("ref_norm_method", "Histogram matching")

    # --- Controls section (always visible) ---
    _method_options = ["CLAHE", "None"]
    _saved_method = _ep.get("method")
    _method_index = _method_options.index(_saved_method) if _saved_method in _method_options else 1
    method = st.selectbox(
        "Enhancement method",
        _method_options,
        index=_method_index,
    )

    # Clear stale result when the selected method differs from what was last applied.
    if (
        state.get("enhance_params", {}).get("method") != method
        or state.get("enhance_params", {}).get("ref_norm") != enable_ref_norm
        or state.get("enhance_params", {}).get("ref_norm_method") != ref_norm_method
    ):
        state.pop("enhanced_image", None)
        state.pop("global_stats", None)
        state.pop("enhance_params", None)
        state.pop("enhance_done", None)

    # Define defaults for clip_limit and tile_size regardless of method
    clip_limit = 2.0
    tile_size = 8

    if method == "CLAHE":
        clip_limit = st.slider("CLAHE clip limit", 1.0, 10.0, max(_ep.get("clip_limit") or 2.0, 1.0), 0.5)
        st.caption("*Limits contrast amplification to prevent noise from being over-enhanced. Higher values increase local contrast more aggressively; 2.0 is a conservative default.*")
        tile_size = st.select_slider("Tile grid size", [4, 8, 16, 32], value=_ep.get("tile_size") or 8)
        st.caption("*Divides the image into a grid of this size for local histogram equalisation. Smaller grids (4) give more localised enhancement; larger grids (32) are closer to global equalisation.*")

    if method is not None:
        session_id = state["_session_id"]

        if state.get("enhance_done") and method == "CLAHE" and state.get("enhance_params", {}).get("method") == "CLAHE":
            if (
                state["enhance_params"].get("clip_limit") != clip_limit
                or state["enhance_params"].get("tile_size") != tile_size
                or state["enhance_params"].get("gray_world") != enable_gw
                or state["enhance_params"].get("ref_norm") != enable_ref_norm
                or state["enhance_params"].get("ref_norm_method") != ref_norm_method
            ):
                st.warning("Parameters have changed since the last run — click Apply Enhancement to update.")

        if st.button("Apply Enhancement", type="primary"):
            try:
                ref_norm_key = "histogram" if ref_norm_method == "Histogram matching" else "linear"
                params_dict = {
                    "gray_world": enable_gw,
                    "ref_norm_enabled": enable_ref_norm,
                    "ref_norm_method": ref_norm_key,
                    "clahe_enabled": method == "CLAHE",
                    "clahe_clip_limit": clip_limit if method == "CLAHE" else 2.0,
                    "clahe_tile_grid": [tile_size, tile_size] if method == "CLAHE" else [8, 8],
                }
                job_id = api.run_enhance(session_id, params_dict)
                result = api.poll_job(job_id, "Applying enhancement...")
                state["global_stats"] = result.get("global_stats")
                state["enhance_done"] = True
                state["enhance_params"] = {
                    "method": method,
                    "clip_limit": clip_limit if method == "CLAHE" else None,
                    "tile_size": tile_size if method == "CLAHE" else None,
                    "gray_world": enable_gw,
                    "ref_norm": enable_ref_norm,
                    "ref_norm_method": ref_norm_method,
                }
            except Exception as exc:
                st.error(f"Enhancement failed: {exc}")

        if state.get("enhance_done"):
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(api.fetch_preview(session_id, "stretched"), caption="Original", use_container_width=True)
            with col2:
                try:
                    st.image(api.fetch_preview(session_id, "dehazed"), caption="Dehazed", use_container_width=True)
                except Exception:
                    st.image(api.fetch_preview(session_id, "stretched"), caption="Stretched (dehaze skipped)", use_container_width=True)
            with col3:
                st.image(api.fetch_preview(session_id, "enhanced"), caption="Enhanced", use_container_width=True)

            p = state["enhance_params"]
            gw_prefix  = "Gray World → " if p.get("gray_world") else ""
            ref_prefix = "Ref Norm → "   if p.get("ref_norm")   else ""
            if p.get("method") == "CLAHE":
                st.caption(f"{gw_prefix}{ref_prefix}CLAHE  clip={p['clip_limit']}  grid={p['tile_size']}×{p['tile_size']}")
            else:
                if gw_prefix or ref_prefix:
                    st.info(f"{gw_prefix}{ref_prefix}No additional enhancement applied.")
                else:
                    st.info("No enhancement applied.")

            if st.button("Proceed to Chipping →", type="primary"):
                return True

    return False