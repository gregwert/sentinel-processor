"""
Step 2 — Cloud overlay with live-adjustable thresholds, DCP dehazing controls,
side-by-side before/after comparison.
"""
import streamlit as st
from sentinel_frontend import api_client as api

from sentinel_frontend.ui.steps import render_step_nav


def _render_brightness_histogram_api(bins: dict) -> None:
    """Render brightness histogram from backend-supplied bin data.

    Parameters
    ----------
    bins : dict
        Expected keys: 'bins' (list of floats, length N+1, bin edges) and
        'cloud_counts' / 'land_counts' (lists of int counts, length N).
    """
    import altair as alt
    import pandas as pd

    edges = bins.get("bins", [])
    cloud_counts = bins.get("cloud_counts", [])
    land_counts = bins.get("land_counts", [])
    if not edges or len(edges) < 2:
        st.caption("Histogram data unavailable.")
        return

    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    rows = []
    for c, cnt in zip(centers, cloud_counts):
        rows.append({"brightness": c, "type": "Cloud", "count": cnt})
    for c, cnt in zip(centers, land_counts):
        rows.append({"brightness": c, "type": "Land", "count": cnt})

    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_bar(opacity=0.55, binSpacing=0)
        .encode(
            alt.X("brightness:Q", title="Mean brightness (0–255)"),
            alt.Y("count:Q", title="Pixel count"),
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
    dehaze_params : dict
        Parameters from the previous run (if any), used to seed
        slider defaults and detect stale-result conditions.
    cloud_mask_stats : dict
        Cloud detection summary from the previous mask call (if any).

    Writes to state
    ---------------
    dehaze_params : dict
        DCP parameters used for the last run, or {'skipped': True}
        when dehazing is skipped or disabled.
    dehaze_done : bool
        True once a dehaze job has succeeded.
    cloud_mask_stats : dict
        Cloud percentage and pixel counts returned by the backend.
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
        for k in ["cloud_mask", "dehazed_image", "dehaze_params", "cloud_mask_stats"]:
            state.pop(k, None)
        state["step"] = "upload"
        st.rerun()
    if skip:
        state["dehaze_params"] = {"skipped": True}
        state["cloud_mask_stats"] = None
        state["step"] = "enhance"
        st.rerun()

    st.header("Step 2 — Dehazing")

    # --- Controls section (always visible) ---
    enable_dehaze = st.toggle(
        "Enable DCP dehazing",
        value=True,
    )
    st.caption("*Dark Channel Prior algorithm removes atmospheric haze and improves contrast over land. Safe to disable for already-clear imagery.*")

    session_id = state["_session_id"]
    _dp = state.get("dehaze_params") or {}

    if enable_dehaze:
        mask_clouds = st.checkbox(
            "Cloud-adaptive atmospheric light",
            value=_dp.get("mask_clouds", True),
        )
        st.caption("*Detects cloud pixels and excludes them when estimating atmospheric light. Without this, clouds bias the haze estimate and degrade results over land.*")

        brightness_thresh = 0.75
        saturation_thresh = 0.08

        if mask_clouds:
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

            # Call backend live on every slider change
            try:
                mask_result = api.update_cloud_mask(
                    session_id, brightness_thresh, saturation_thresh
                )
                state["cloud_mask_stats"] = mask_result
            except Exception as exc:
                st.warning(f"Cloud mask update failed: {exc}")
                mask_result = state.get("cloud_mask_stats", {})

            # Show the stretched preview (backend cloud overlay not yet implemented)
            col_left, col_right = st.columns(2)
            with col_left:
                st.image(
                    api.fetch_preview(session_id, "stretched"),
                    caption="Original",
                    use_container_width=True,
                )
            with col_right:
                # TODO: wire up cloud-overlay preview endpoint when available
                st.image(
                    api.fetch_preview(session_id, "stretched"),
                    caption="Cloud overlay (pending backend support)",
                    use_container_width=True,
                )

            stats = state.get("cloud_mask_stats", {})
            if stats:
                cloud_pct = stats.get("cloud_pct", 0)
                cloud_px = stats.get("cloud_px", 0)
                st.caption(f"{cloud_pct * 100:.1f}% cloud ({cloud_px:,} px)")

            with st.expander("Pixel brightness histogram"):
                try:
                    hist_data = api.get_histogram(session_id, stage="stretched")
                    _render_brightness_histogram_api(hist_data)
                except Exception as exc:
                    st.caption(f"Histogram unavailable: {exc}")
        else:
            state["cloud_mask_stats"] = None

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
                value=_dp.get("use_guided_filter", True),
            )
            st.caption("*Refines the transmission map to reduce halo artefacts at sharp edges (e.g. coastlines, cloud edges). Slightly slower but recommended.*")

        if "dehaze_done" in state:
            p = state.get("dehaze_params", {})
            stale = p.get("skipped") or (
                p.get("omega") != omega
                or p.get("t0") != t0
                or p.get("patch_size") != patch_size
                or p.get("use_guided_filter") != use_guided
                or p.get("mask_clouds") != mask_clouds
                or (mask_clouds and (
                    p.get("brightness_thresh") != brightness_thresh
                    or p.get("saturation_thresh") != saturation_thresh
                ))
            )
            if stale:
                st.warning("Parameters have changed since the last run — click Run Dehazing to update.")

        if st.button("Run Dehazing", type="primary"):
            try:
                params_dict = {
                    "omega": omega,
                    "t0": t0,
                    "patch_size": patch_size,
                    "use_guided_filter": use_guided,
                    "mask_clouds": mask_clouds,
                    "brightness_thresh": brightness_thresh if mask_clouds else 0.75,
                    "saturation_thresh": saturation_thresh if mask_clouds else 0.08,
                }
                job_id = api.run_dehaze(session_id, params_dict)
                api.poll_job(job_id, "Running DCP dehazing...")
                state["dehaze_done"] = True
                state["dehaze_params"] = params_dict
            except Exception as exc:
                st.error(f"Dehazing failed: {exc}")

    else:
        state["cloud_mask_stats"] = None
        if st.button("Skip (use image as-is)", type="secondary"):
            state["dehaze_params"] = {"skipped": True}
            state["step"] = "enhance"
            st.rerun()

    # Result section
    if state.get("dehaze_done"):
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                api.fetch_preview(session_id, "stretched"),
                caption="Before",
                use_container_width=True,
            )
        with col2:
            st.image(
                api.fetch_preview(session_id, "dehazed"),
                caption="After DCP",
                use_container_width=True,
            )

        if state.get("dehaze_params", {}).get("skipped"):
            st.info("Dehazing skipped.")
        else:
            p = state["dehaze_params"]
            st.caption(
                f"ω={p['omega']}  t₀={p['t0']}  patch={p['patch_size']}  "
                f"guided={p['use_guided_filter']}  "
                f"clouds={'masked' if p['mask_clouds'] else 'not masked'}"
            )

        if st.button("Proceed to Enhancement →", type="primary"):
            return True

    return False