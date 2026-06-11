"""
Step 2 — Cloud overlay with live-adjustable thresholds, DCP dehazing controls,
side-by-side before/after comparison.
"""
import streamlit as st
import numpy as np
from processing.dehazing import detect_clouds_simple, dehaze
from ui.cloud_overlay import render_cloud_composite, compute_cloud_stats


def render(state: dict) -> bool:
    """Render dehaze step. Returns True when user clicks Next to advance."""

    col_back, col_spacer, col_skip = st.columns([1, 4, 1])
    with col_back:
        if st.button("← Back", key="back_dehaze"):
            for k in ["cloud_mask", "dehazed_image", "dehaze_params"]:
                state.pop(k, None)
            state["step"] = "upload"
            st.rerun()
    with col_skip:
        if st.button("Skip →", key="skip_dehaze"):
            state["dehazed_image"] = state["stretched_image"].copy()
            state["dehaze_params"] = {"skipped": True}
            state["step"] = "enhance"
            st.rerun()

    st.header("Step 2 — Dehazing")

    # --- Controls section (always visible) ---
    enable_dehaze = st.toggle(
        "Enable DCP dehazing",
        value=True,
    )
    st.caption("*Dark Channel Prior algorithm removes atmospheric haze and improves contrast over land. Safe to disable for already-clear imagery.*")

    if enable_dehaze:
        mask_clouds = st.checkbox(
            "Cloud-adaptive atmospheric light",
            value=True,
        )
        st.caption("*Detects cloud pixels and excludes them when estimating atmospheric light. Without this, clouds bias the haze estimate and degrade results over land.*")

        brightness_thresh = 0.75
        saturation_thresh = 0.08

        if mask_clouds:
            col_left, col_right = st.columns(2)

            brightness_thresh = st.slider(
                "Brightness threshold",
                0.5,
                0.95,
                0.75,
                0.01,
            )
            st.caption("*Pixels brighter than this (0–1 scale) are cloud candidates. Raise to classify fewer pixels as cloud; lower to catch more.*")
            saturation_thresh = st.slider(
                "Saturation threshold",
                0.01,
                0.20,
                0.08,
                0.01,
            )
            st.caption("*Cloud pixels are nearly white (low colour saturation). Pixels with max–min channel difference below this are cloud candidates. Raise to catch more greyish clouds; lower for only pure-white clouds.*")

            cloud_mask = detect_clouds_simple(
                state["stretched_image"], brightness_thresh, saturation_thresh
            )
            state["cloud_mask"] = cloud_mask
            composite = render_cloud_composite(state["stretched_image"], cloud_mask)

            with col_left:
                st.image(
                    state["stretched_image"],
                    caption="Original",
                    use_column_width=True,
                )
            with col_right:
                st.image(
                    composite,
                    caption="Cloud overlay (red)",
                    use_column_width=True,
                )

            stats = compute_cloud_stats(cloud_mask)
            st.caption(f"{stats['cloud_pct']}% cloud ({stats['cloud_px']:,} px)")

        with st.expander("DCP parameters"):
            omega = st.slider(
                "Omega — haze removal strength",
                0.5,
                1.0,
                0.95,
                0.05,
            )
            st.caption("*Controls how aggressively haze is removed. 0.95 removes most haze; lower values preserve more of the original atmospheric tone. Values above 0.98 can over-saturate.*")
            t0 = st.slider(
                "t₀ — min transmission",
                0.05,
                0.5,
                0.1,
                0.05,
            )
            st.caption("*Minimum transmission value — prevents over-brightening in very dense haze or fully clouded areas. Raise if output looks blown out; lower to dehaze more aggressively.*")
            patch_size = st.slider(
                "Patch size",
                5,
                31,
                15,
                2,
            )
            st.caption("*Size of the local window used to compute the dark channel. Larger patches give smoother results but may lose fine detail. 15 is a good default for satellite imagery.*")
            use_guided = st.checkbox(
                "Guided filter refinement",
                value=True,
            )
            st.caption("*Refines the transmission map to reduce halo artefacts at sharp edges (e.g. coastlines, cloud edges). Slightly slower but recommended.*")

        if st.button("Run Dehazing", type="primary"):
            with st.spinner("Running DCP dehazing..."):
                dehazed = dehaze(
                    state["stretched_image"],
                    patch_size,
                    omega,
                    t0,
                    use_guided,
                    mask_clouds,
                    brightness_thresh if mask_clouds else 0.75,
                    saturation_thresh if mask_clouds else 0.08,
                )
            state["dehazed_image"] = dehazed
            state["dehaze_params"] = {
                "omega": omega,
                "t0": t0,
                "patch_size": patch_size,
                "use_guided": use_guided,
                "mask_clouds": mask_clouds,
                "brightness_thresh": brightness_thresh,
                "saturation_thresh": saturation_thresh,
            }

    else:
        if st.button("Skip (use image as-is)", type="secondary"):
            state["dehazed_image"] = state["stretched_image"].copy()
            state["dehaze_params"] = {"skipped": True}

    # --- Result section (only when dehazed_image in state) ---
    if "dehazed_image" in state:
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.image(state["stretched_image"], caption="Before", use_column_width=True)
            st.caption(
                f"μ={state['stretched_image'].mean():.1f}  σ={state['stretched_image'].std():.1f}"
            )
        with col2:
            st.image(state["dehazed_image"], caption="After DCP", use_column_width=True)
            st.caption(
                f"μ={state['dehazed_image'].mean():.1f}  σ={state['dehazed_image'].std():.1f}"
            )

        if state.get("dehaze_params", {}).get("skipped"):
            st.info("Dehazing skipped.")
        else:
            p = state["dehaze_params"]
            st.caption(
                f"ω={p['omega']}  t₀={p['t0']}  patch={p['patch_size']}  "
                f"guided={p['use_guided']}  "
                f"clouds={'masked' if p['mask_clouds'] else 'not masked'}"
            )

        # Stale result warning
        was_skipped = state.get("dehaze_params", {}).get("skipped", False)
        if enable_dehaze == was_skipped:
            st.warning("Parameters have changed since the last run — click Run Dehazing to update.")
        elif enable_dehaze and not was_skipped:
            p = state["dehaze_params"]
            if (
                p.get("omega") != omega
                or p.get("t0") != t0
                or p.get("patch_size") != patch_size
                or p.get("use_guided") != use_guided
                or p.get("mask_clouds") != mask_clouds
                or (mask_clouds and (
                    p.get("brightness_thresh") != brightness_thresh
                    or p.get("saturation_thresh") != saturation_thresh
                ))
            ):
                st.warning("Parameters have changed since the last run — click Run Dehazing to update.")

        if st.button("Proceed to Enhancement →", type="primary"):
            return True

    return False
