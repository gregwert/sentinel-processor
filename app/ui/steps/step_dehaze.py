"""
Step 2 — Cloud overlay with live-adjustable thresholds, DCP dehazing controls,
side-by-side before/after comparison.
"""
import numpy as np
import streamlit as st
from processing.dehazing import detect_clouds_simple, dehaze
from ui.cloud_overlay import render_cloud_composite, compute_cloud_stats
from ui.steps import render_step_nav


def _render_brightness_histogram(img_uint8: np.ndarray, cloud_mask: np.ndarray) -> None:
    """Render overlapping Altair histograms: cloud pixels (red) vs land pixels (blue).

    Subsamples to 200 000 pixels max for performance on large images.

    Parameters
    ----------
    img_uint8 : np.ndarray
        Shape (H, W, 3), dtype uint8.
    cloud_mask : np.ndarray
        Shape (H, W), dtype bool.
    """
    import altair as alt
    import pandas as pd

    brightness = img_uint8.mean(axis=2).flatten()
    label = np.where(cloud_mask.flatten(), "Cloud", "Land")
    n = len(brightness)
    if n > 200_000:
        idx = np.random.default_rng(42).choice(n, 200_000, replace=False)
        brightness = brightness[idx]
        label = label[idx]

    df = pd.DataFrame({"brightness": brightness.astype(float), "type": label})
    chart = (
        alt.Chart(df)
        .mark_bar(opacity=0.55, binSpacing=0)
        .encode(
            alt.X("brightness:Q", bin=alt.Bin(maxbins=80), title="Mean brightness (0–255)"),
            alt.Y("count()", title="Pixel count"),
            alt.Color(
                "type:N",
                scale=alt.Scale(domain=["Cloud", "Land"], range=["#e45756", "#4c78a8"]),
            ),
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Red = cloud pixels  ·  Blue = non-cloud pixels")


def render(state: dict) -> bool:
    """Render Step 2 — Dehazing.

    Reads from state
    ----------------
    stretched_image : np.ndarray
        The uint8 (H, W, 3) image produced by step_upload.
    dehaze_params : dict
        Parameters used for the previous run (if any), used to detect
        stale-result conditions and display the last-run summary.
    dehazed_image : np.ndarray
        Previously computed dehaze result (if any), used in the
        before/after comparison panel.

    Writes to state
    ---------------
    cloud_mask : np.ndarray
        bool (H, W) cloud detection mask. Updated live on every
        threshold slider change when cloud-adaptive mode is enabled.
    dehazed_image : np.ndarray
        uint8 (H, W, 3) result after DCP dehazing. Set on Run Dehazing
        click, or copied from stretched_image when skipped.
    dehaze_params : dict
        Parameters used for the last run, or {'skipped': True} if
        dehazing was skipped or disabled.
    step : str
        Set to 'upload' on Back click, or 'enhance' on Skip click.

    Returns
    -------
    bool
        True when the user clicks 'Proceed to Enhancement →' to advance
        to the enhance step. False on all other renders.
    """
    back, skip = render_step_nav("dehaze")
    if back:
        for k in ["cloud_mask", "dehazed_image", "dehaze_params"]:
            state.pop(k, None)
        state["step"] = "upload"
        st.rerun()
    if skip:
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
        _dp = state.get("dehaze_params") or {}

        mask_clouds = st.checkbox(
            "Cloud-adaptive atmospheric light",
            value=_dp.get("mask_clouds", True),
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
                _dp.get("brightness_thresh", 0.75),
                0.01,
            )
            st.caption("*Pixels brighter than this (0–1 scale) are cloud candidates. Raise to classify fewer pixels as cloud; lower to catch more.*")
            saturation_thresh = st.slider(
                "Saturation threshold",
                0.01,
                0.20,
                _dp.get("saturation_thresh", 0.08),
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

            if cloud_mask.any() and (~cloud_mask).any():
                with st.expander("Pixel brightness histogram"):
                    _render_brightness_histogram(state["stretched_image"], cloud_mask)

        with st.expander("DCP parameters"):
            omega = st.slider(
                "Omega — haze removal strength",
                0.5,
                1.0,
                _dp.get("omega", 0.95),
                0.05,
            )
            st.caption("*Controls how aggressively haze is removed. 0.95 removes most haze; lower values preserve more of the original atmospheric tone. Values above 0.98 can over-saturate.*")
            t0 = st.slider(
                "t₀ — min transmission",
                0.05,
                0.5,
                _dp.get("t0", 0.1),
                0.05,
            )
            st.caption("*Minimum transmission value — prevents over-brightening in very dense haze or fully clouded areas. Raise if output looks blown out; lower to dehaze more aggressively.*")
            patch_size = st.slider(
                "Patch size",
                5,
                31,
                _dp.get("patch_size", 15),
                2,
            )
            st.caption("*Size of the local window used to compute the dark channel. Larger patches give smoother results but may lose fine detail. 15 is a good default for satellite imagery.*")
            use_guided = st.checkbox(
                "Guided filter refinement",
                value=_dp.get("use_guided", True),
            )
            st.caption("*Refines the transmission map to reduce halo artefacts at sharp edges (e.g. coastlines, cloud edges). Slightly slower but recommended.*")

        if "dehazed_image" in state:
            was_skipped = state.get("dehaze_params", {}).get("skipped", False)
            p = state.get("dehaze_params", {})
            stale = was_skipped or (
                p.get("omega") != omega
                or p.get("t0") != t0
                or p.get("patch_size") != patch_size
                or p.get("use_guided") != use_guided
                or p.get("mask_clouds") != mask_clouds
                or (mask_clouds and (
                    p.get("brightness_thresh") != brightness_thresh
                    or p.get("saturation_thresh") != saturation_thresh
                ))
            )
            if stale:
                st.warning("Parameters have changed since the last run — click Run Dehazing to update.")

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
            state["step"] = "enhance"
            st.rerun()

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

        if st.button("Proceed to Enhancement →", type="primary"):
            return True

    return False
