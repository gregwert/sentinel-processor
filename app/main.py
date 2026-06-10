import streamlit as st
import tempfile
import os

from processing.pipeline import run_pipeline, PipelineConfig
from chipping.gdal_chipper import build_chip_grid
from ui.sidebar import render_sidebar
from ui.preview import render_stage_preview
from ui.tile_viewer import render_tile_viewer, render_export_controls

st.set_page_config(page_title="Sentinel Processor", layout="wide")
st.title("Sentinel-2 Imagery Processing Pipeline")

# Always render the sidebar first so widget state is preserved
params = render_sidebar()

# -----------------------------------------------------------------------
# Phase B — Tile Viewer (chip_grid exists in session state)
# -----------------------------------------------------------------------
if "chip_grid" in st.session_state:
    if st.button("← Re-run Pipeline", key="rerun"):
        del st.session_state["chip_grid"]
        del st.session_state["pipeline_result"]
        st.session_state["tile_page"] = 0
        st.rerun()

    render_stage_preview(st.session_state["pipeline_result"].stages)
    st.divider()

    render_tile_viewer(st.session_state["chip_grid"])
    st.divider()

    zip_bytes = render_export_controls(st.session_state["chip_grid"])
    if zip_bytes is not None:
        fmt = st.session_state.get("export_fmt", "png")
        st.download_button(
            label=f"Download {st.session_state['chip_grid'].total} chips (.zip)",
            data=zip_bytes,
            file_name=f"chips_{fmt}.zip",
            mime="application/zip",
            type="primary",
        )

# -----------------------------------------------------------------------
# Phase A — Upload + Configure + Run
# -----------------------------------------------------------------------
else:
    uploaded = st.file_uploader(
        "Upload a Sentinel-2 TIFF",
        type=["tif", "tiff"],
        help="16-bit multi-band GeoTIFF. Files up to 2 GB supported.",
    )

    if uploaded:
        if st.button("Run Pipeline", type="primary", key="run"):
            progress = st.progress(0, text="Reading and preprocessing...")
            tmp_path = None
            try:
                # Write upload to temp file
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                # Build config from sidebar params
                config = PipelineConfig(
                    band_indices=params["band_indices"],
                    p_low=params["p_low"],
                    p_high=params["p_high"],
                    per_band_stretch=params["per_band_stretch"],
                    run_dehaze=params["run_dehaze"],
                    patch_size=params["patch_size"],
                    omega=params["omega"],
                    t0=params["t0"],
                    use_guided_filter=params["use_guided_filter"],
                    mask_clouds=params["mask_clouds"],
                    cloud_brightness_thresh=params["cloud_brightness_thresh"],
                    cloud_saturation_thresh=params["cloud_saturation_thresh"],
                    enhancement=params["enhancement"],
                    clahe_clip_limit=params["clahe_clip_limit"],
                    clahe_tile_grid=params["clahe_tile_grid"],
                    std_target_mean=params["std_target_mean"],
                    std_target_std=params["std_target_std"],
                )

                progress.progress(20, text="Preprocessing...")
                result = run_pipeline(tmp_path, config)

                progress.progress(80, text="Building chip grid...")
                grid = build_chip_grid(
                    result.image,
                    result.meta,
                    params["chip_w"],
                    params["chip_h"],
                    params["overlap"],
                )

                progress.progress(100, text="Done!")
                st.session_state["pipeline_result"] = result
                st.session_state["chip_grid"] = grid
                st.session_state["tile_page"] = 0

            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                raise
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                progress.empty()

            st.rerun()
    else:
        st.info("Upload a Sentinel-2 TIFF to begin. Use the sidebar to configure processing parameters.")
