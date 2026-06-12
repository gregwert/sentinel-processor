"""
Step 3 — CLAHE enhancement on full image, before/after comparison.
Computes and stores global_stats (mean/std) for use at export time.
Standardisation has been removed from this step — it now lives in tile_exporter.py.
"""
import streamlit as st
import numpy as np
from processing.enhancement import apply_clahe, apply_gray_world
from processing.reference_norm import apply_reference_normalisation
from ui.steps import render_step_nav


def render(state: dict) -> bool:
    """Render Step 3 — Enhancement.

    Reads from state
    ----------------
    dehazed_image : np.ndarray
        uint8 (H, W, 3) image produced by step_dehaze (or the stretched
        image when dehazing was skipped). Used as input to enhancement
        and displayed in the before/after comparison.
    stretched_image : np.ndarray
        Fallback source used only when dehazed_image is absent (skip path).
    enhance_params : dict
        Parameters from the previous Apply run (if any), used to detect
        stale-result conditions and display the last-run summary.

    Writes to state
    ---------------
    enhanced_image : np.ndarray
        uint8 (H, W, 3) result after CLAHE (or identity) enhancement.
        Set on Apply Enhancement click, or copied from dehazed_image
        when skipped.
    enhance_params : dict
        Enhancement method and parameters used for the last run, or
        {'skipped': True} if the step was skipped.
    global_stats : dict or None
        {'mean': [r, g, b], 'std': [r, g, b]} computed over enhanced_image.
        None when enhancement was skipped.
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
        for k in ["enhanced_image", "global_stats", "enhance_params"]:
            state.pop(k, None)
        state["step"] = "dehaze"
        st.rerun()
    if skip:
        state["enhanced_image"] = state.get("dehazed_image", state["stretched_image"]).copy()
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
    # After a successful Apply the stored method matches the widget value, so this
    # condition is False on the very next render and no flickering occurs.
    if (
        state.get("enhance_params", {}).get("method") != method
        or state.get("enhance_params", {}).get("ref_norm") != enable_ref_norm
        or state.get("enhance_params", {}).get("ref_norm_method") != ref_norm_method
    ):
        state.pop("enhanced_image", None)
        state.pop("global_stats", None)
        state.pop("enhance_params", None)

    # Define defaults for clip_limit and tile_size regardless of method
    clip_limit = 2.0
    tile_size = 8

    if method == "CLAHE":
        clip_limit = st.slider("CLAHE clip limit", 0.5, 10.0, _ep.get("clip_limit") or 2.0, 0.5)
        st.caption("*Limits contrast amplification to prevent noise from being over-enhanced. Higher values increase local contrast more aggressively; 2.0 is a conservative default.*")
        tile_size = st.select_slider("Tile grid size", [4, 8, 16, 32], value=_ep.get("tile_size") or 8)
        st.caption("*Divides the image into a grid of this size for local histogram equalisation. Smaller grids (4) give more localised enhancement; larger grids (32) are closer to global equalisation.*")

    if method is not None:
        if "enhanced_image" in state and method == "CLAHE" and state.get("enhance_params", {}).get("method") == "CLAHE":
            if (
                state["enhance_params"].get("clip_limit") != clip_limit
                or state["enhance_params"].get("tile_size") != tile_size
                or state["enhance_params"].get("gray_world") != enable_gw
                or state["enhance_params"].get("ref_norm") != enable_ref_norm
                or state["enhance_params"].get("ref_norm_method") != ref_norm_method
            ):
                st.warning("Parameters have changed since the last run — click Apply Enhancement to update.")

        if st.button("Apply Enhancement", type="primary"):
            with st.spinner("Applying enhancement..."):
                src = state["dehazed_image"]
                if enable_gw:
                    src = apply_gray_world(src, cloud_mask=state.get("cloud_mask"))
                if enable_ref_norm and "reference_stats" in state:
                    method_key = "histogram" if ref_norm_method == "Histogram matching" else "linear"
                    src = apply_reference_normalisation(src, state["reference_stats"], method_key)
                if method == "CLAHE":
                    enhanced = apply_clahe(
                        src, clip_limit, (tile_size, tile_size)
                    )
                else:
                    enhanced = src.copy()

            img_f = enhanced.astype(np.float32)
            global_stats = {
                "mean": img_f.mean(axis=(0, 1)).tolist(),
                "std": img_f.std(axis=(0, 1)).tolist(),
            }
            state["enhanced_image"] = enhanced
            state["global_stats"] = global_stats
            state["enhance_params"] = {
                "method": method,
                "clip_limit": clip_limit if method == "CLAHE" else None,
                "tile_size": tile_size if method == "CLAHE" else None,
                "gray_world": enable_gw,
                "ref_norm": enable_ref_norm,
                "ref_norm_method": ref_norm_method,
            }

    # --- Result section (only when enhanced_image in state) ---
    if "enhanced_image" in state:
        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(state["stretched_image"], caption="Original", use_column_width=True)
            st.caption(
                f"μ={state['stretched_image'].mean():.1f}  σ={state['stretched_image'].std():.1f}"
            )
        with col2:
            st.image(state["dehazed_image"], caption="Dehazed", use_column_width=True)
            st.caption(
                f"μ={state['dehazed_image'].mean():.1f}  σ={state['dehazed_image'].std():.1f}"
            )
        with col3:
            st.image(state["enhanced_image"], caption="Enhanced", use_column_width=True)
            st.caption(
                f"μ={state['enhanced_image'].mean():.1f}  σ={state['enhanced_image'].std():.1f}"
            )

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
