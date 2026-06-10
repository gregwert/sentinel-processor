"""
ui/sidebar.py

Streamlit sidebar component for the Sentinel Processor application.
Renders all pipeline configuration controls and returns a dict of
parameter values for main.py to assemble into a PipelineConfig.
"""

import streamlit as st


def render_sidebar() -> dict:
    """Render all sidebar controls; return dict of parameter values."""

    # --- Section 1: Band Selection ---
    st.sidebar.header("1. Band Selection")
    band_r = st.sidebar.number_input("Red band index", min_value=1, max_value=12, value=1, step=1)
    band_g = st.sidebar.number_input("Green band index", min_value=1, max_value=12, value=2, step=1)
    band_b = st.sidebar.number_input("Blue band index", min_value=1, max_value=12, value=3, step=1)

    # --- Section 2: Preprocessing ---
    st.sidebar.header("2. Preprocessing")
    p_low = st.sidebar.slider("Low percentile clip", 0.0, 10.0, 2.0, 0.5)
    p_high = st.sidebar.slider("High percentile clip", 90.0, 100.0, 98.0, 0.5)
    per_band = st.sidebar.checkbox(
        "Per-band stretch",
        value=True,
        help="Stretch each band independently. Recommended for true-color composites.",
    )

    # --- Section 3: Dehazing ---
    st.sidebar.header("3. Dehazing (Dark Channel Prior)")
    run_dehaze = st.sidebar.toggle("Enable dehazing", value=True)

    if run_dehaze:
        patch_size = st.sidebar.slider("Patch size", 5, 31, 15, 2)
        omega = st.sidebar.slider("Omega — haze removal strength", 0.5, 1.0, 0.95, 0.05)
        t0 = st.sidebar.slider("t₀ — min transmission", 0.05, 0.5, 0.1, 0.05)
        use_guided = st.sidebar.checkbox(
            "Guided filter refinement",
            value=True,
            help="Reduces halo artifacts at edges. Slightly slower.",
        )
        mask_clouds = st.sidebar.checkbox(
            "Cloud-adaptive atmospheric light",
            value=True,
            help="Excludes cloud pixels from atmospheric light estimation. Improves results on cloudy scenes.",
        )

        if mask_clouds:
            with st.sidebar.expander("Cloud detection thresholds"):
                brightness_thresh = st.slider("Brightness threshold", 0.5, 0.95, 0.75, 0.05)
                saturation_thresh = st.slider("Saturation threshold", 0.01, 0.2, 0.08, 0.01)
        else:
            brightness_thresh = 0.75
            saturation_thresh = 0.08
    else:
        patch_size = 15
        omega = 0.95
        t0 = 0.1
        use_guided = True
        mask_clouds = True
        brightness_thresh = 0.75
        saturation_thresh = 0.08

    # --- Section 4: Enhancement ---
    st.sidebar.header("4. Enhancement")
    enhancement = st.sidebar.selectbox(
        "Method",
        ["clahe", "standardization", "none"],
        format_func=lambda x: {"clahe": "CLAHE", "standardization": "Standardization", "none": "None"}[x],
    )

    if enhancement == "clahe":
        clip_limit = st.sidebar.slider("CLAHE clip limit", 0.5, 10.0, 2.0, 0.5)
        tile_size = st.sidebar.select_slider("Tile grid size", [4, 8, 16, 32], value=8)
    else:
        clip_limit = 2.0
        tile_size = 8

    if enhancement == "standardization":
        std_mean = st.sidebar.slider("Target mean", 64.0, 192.0, 127.5, 0.5)
        std_std = st.sidebar.slider("Target std", 10.0, 80.0, 45.0, 1.0)
    else:
        std_mean = 127.5
        std_std = 45.0

    # --- Section 5: Chipping ---
    st.sidebar.header("5. Chipping")
    chip_w = st.sidebar.number_input("Chip width (px)", min_value=64, max_value=2048, value=256, step=64)
    chip_h = st.sidebar.number_input("Chip height (px)", min_value=64, max_value=2048, value=256, step=64)
    overlap = st.sidebar.slider(
        "Overlap fraction",
        0.0,
        0.49,
        0.0,
        0.05,
        help="Fraction of overlap between adjacent chips. 0 = no overlap.",
    )
    naming = st.sidebar.selectbox("Chip naming", ["rowcol", "coords"])

    return {
        "band_indices": (int(band_r), int(band_g), int(band_b)),
        "p_low": p_low,
        "p_high": p_high,
        "per_band_stretch": per_band,
        "run_dehaze": run_dehaze,
        "patch_size": int(patch_size),
        "omega": omega,
        "t0": t0,
        "use_guided_filter": use_guided,
        "mask_clouds": mask_clouds,
        "cloud_brightness_thresh": brightness_thresh,
        "cloud_saturation_thresh": saturation_thresh,
        "enhancement": enhancement,
        "clahe_clip_limit": clip_limit if enhancement == "clahe" else 2.0,
        "clahe_tile_grid": (tile_size, tile_size) if enhancement == "clahe" else (8, 8),
        "std_target_mean": std_mean if enhancement == "standardization" else 127.5,
        "std_target_std": std_std if enhancement == "standardization" else 45.0,
        "chip_w": int(chip_w),
        "chip_h": int(chip_h),
        "overlap": overlap,
        "naming": naming,
    }
