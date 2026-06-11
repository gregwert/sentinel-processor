"""
Step 3 — CLAHE enhancement on full image, before/after comparison.
Computes and stores global_stats (mean/std) for use at export time.
Standardisation has been removed from this step — it now lives in tile_exporter.py.
"""
import streamlit as st
import numpy as np
from processing.enhancement import apply_clahe


def render(state: dict) -> bool:
    """Render enhance step. Returns True when user clicks Next to advance."""

    col_back, col_spacer, col_skip = st.columns([1, 4, 1])
    with col_back:
        if st.button("← Back", key="back_enhance"):
            for k in ["enhanced_image", "global_stats", "enhance_params"]:
                state.pop(k, None)
            state["step"] = "dehaze"
            st.rerun()
    with col_skip:
        if st.button("Skip →", key="skip_enhance"):
            state["enhanced_image"] = state.get("dehazed_image", state["stretched_image"]).copy()
            state["enhance_params"] = {"skipped": True}
            state["global_stats"] = None
            state["step"] = "chip"
            st.rerun()

    st.header("Step 3 — Enhancement")

    # --- Controls section (always visible) ---
    method = st.selectbox(
        "Enhancement method",
        ["CLAHE", "None"],
        index=None,
        placeholder="Select an enhancement method...",
    )

    if method is None:
        st.info("Select an enhancement method above. Choose 'None' to proceed without enhancement.")

    # Clear stale result when the selected method differs from what was last applied
    if method is not None and state.get("enhance_params", {}).get("method") != method:
        state.pop("enhanced_image", None)
        state.pop("global_stats", None)
        state.pop("enhance_params", None)

    # Define defaults for clip_limit and tile_size regardless of method
    clip_limit = 2.0
    tile_size = 8

    if method == "CLAHE":
        clip_limit = st.slider("CLAHE clip limit", 0.5, 10.0, 2.0, 0.5)
        st.caption("*Limits contrast amplification to prevent noise from being over-enhanced. Higher values increase local contrast more aggressively; 2.0 is a conservative default.*")
        tile_size = st.select_slider("Tile grid size", [4, 8, 16, 32], value=8)
        st.caption("*Divides the image into a grid of this size for local histogram equalisation. Smaller grids (4) give more localised enhancement; larger grids (32) are closer to global equalisation.*")

    if method is not None:
        if st.button("Apply Enhancement", type="primary"):
            with st.spinner("Applying enhancement..."):
                if method == "CLAHE":
                    enhanced = apply_clahe(
                        state["dehazed_image"], clip_limit, (tile_size, tile_size)
                    )
                else:
                    enhanced = state["dehazed_image"].copy()

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
            }

    # --- Result section (only when enhanced_image in state) ---
    if "enhanced_image" in state:
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.image(state["dehazed_image"], caption="Before", use_column_width=True)
            st.caption(
                f"μ={state['dehazed_image'].mean():.1f}  σ={state['dehazed_image'].std():.1f}"
            )
        with col2:
            st.image(state["enhanced_image"], caption="After Enhancement", use_column_width=True)
            st.caption(
                f"μ={state['enhanced_image'].mean():.1f}  σ={state['enhanced_image'].std():.1f}"
            )

        p = state["enhance_params"]
        if p.get("method") == "CLAHE":
            st.caption(f"CLAHE  clip={p['clip_limit']}  grid={p['tile_size']}×{p['tile_size']}")
        else:
            st.info("No enhancement applied.")

        # Stale result warning
        if method is not None and method == "CLAHE" and p.get("method") == "CLAHE":
            if p.get("clip_limit") != clip_limit or p.get("tile_size") != tile_size:
                st.warning("Parameters have changed since the last run — click Apply Enhancement to update.")

        if st.button("Proceed to Chipping →", type="primary"):
            return True

    return False
