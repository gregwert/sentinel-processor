"""
Step 5 — Review & Export.
Shows processing parameters, processed image, and chips (if not skipped).
All exports are selected via checkboxes then generated together as a single zip.
"""
import streamlit as st
import numpy as np
import io, os, zipfile, tempfile
import yaml
from PIL import Image as PILImage


def _build_export_zip(state, export_params, export_image, img_fmt,
                      export_chips, chip_fmt, chip_naming, normalise_chips) -> "bytes | None":
    buf = io.BytesIO()
    added = False

    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:

        # 1. Parameters YAML
        if export_params:
            params = {}
            if state.get("upload_params"):
                params["upload"] = state["upload_params"]
            if state.get("dehaze_params"):
                params["dehazing"] = state["dehaze_params"]
            if state.get("enhance_params"):
                params["enhancement"] = state["enhance_params"]
            if state.get("chip_params"):
                params["chipping"] = state["chip_params"]
            yaml_bytes = yaml.dump(params, default_flow_style=False, sort_keys=False).encode("utf-8")
            zf.writestr("params.yaml", yaml_bytes)
            added = True

        # 2. Processed image
        if export_image and state.get("enhanced_image") is not None:
            img = state["enhanced_image"]
            if img_fmt == "PNG":
                img_buf = io.BytesIO()
                PILImage.fromarray(img).save(img_buf, format="PNG")
                zf.writestr("processed_image.png", img_buf.getvalue())
                added = True
            elif img_fmt == "GeoTIFF":
                import rasterio
                meta = state.get("source_meta", {})
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    h, w, c = img.shape
                    with rasterio.open(tmp_path, 'w', driver='GTiff',
                                       height=h, width=w, count=c, dtype='uint8',
                                       crs=meta.get("crs"),
                                       transform=meta.get("transform"),
                                       compress='lzw') as dst:
                        for b in range(c):
                            dst.write(img[:, :, b], b + 1)
                    with open(tmp_path, 'rb') as f:
                        zf.writestr("processed_image.tif", f.read())
                    added = True
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        # 3. Chips
        if export_chips and state.get("chip_grid") is not None:
            from chipping.tile_exporter import export_chips as _export_chips
            with tempfile.TemporaryDirectory() as tmp_dir:
                paths = _export_chips(
                    state["chip_grid"], tmp_dir,
                    fmt=chip_fmt, naming=chip_naming,
                    normalise=normalise_chips,
                    global_stats=state.get("global_stats"),
                )
                for path in paths:
                    zf.write(path, arcname=os.path.join("chips", os.path.basename(path)))
                added = True

    return buf.getvalue() if added else None


def render(state: dict) -> bool:
    """Render review step. Returns True if user clicks Start Over."""

    st.header("Review & Export")

    # ------------------------------------------------------------------ #
    # Section 1 — Processing parameters
    # ------------------------------------------------------------------ #
    st.subheader("Processing Parameters")
    export_params = st.checkbox("Export parameters (YAML)", value=True, key="export_params_cb")

    with st.expander("View parameters", expanded=True):
        params = {}
        if state.get("upload_params"):
            params["upload"] = state["upload_params"]
        if state.get("dehaze_params") and not state["dehaze_params"].get("skipped"):
            params["dehazing"] = state["dehaze_params"]
        elif state.get("dehaze_params", {}).get("skipped"):
            params["dehazing"] = "skipped"
        if state.get("enhance_params") and not state["enhance_params"].get("skipped"):
            params["enhancement"] = state["enhance_params"]
        elif state.get("enhance_params", {}).get("skipped"):
            params["enhancement"] = "skipped"
        if not state.get("chip_skipped") and state.get("chip_params"):
            params["chipping"] = state["chip_params"]
        elif state.get("chip_skipped"):
            params["chipping"] = "skipped"
        st.code(yaml.dump(params, default_flow_style=False, sort_keys=False), language="yaml")

    st.divider()

    # ------------------------------------------------------------------ #
    # Section 2 — Processed image
    # ------------------------------------------------------------------ #
    st.subheader("Processed Image")
    export_image = st.checkbox("Export processed image", value=True, key="export_image_cb")
    if export_image:
        img_fmt = st.radio("Image format", ["PNG", "GeoTIFF"], horizontal=True, key="export_image_fmt")
    else:
        img_fmt = "PNG"

    with st.expander("View image", expanded=True):
        img = state.get("enhanced_image")
        if img is not None:
            h, w = img.shape[:2]
            scale = min(900 / max(h, w), 1.0)
            disp = PILImage.fromarray(img).resize(
                (max(1, int(w * scale)), max(1, int(h * scale))), PILImage.LANCZOS
            )
            st.image(disp, use_column_width=True)

    st.divider()

    # ------------------------------------------------------------------ #
    # Section 3 — Chips (only if not skipped)
    # ------------------------------------------------------------------ #
    if not state.get("chip_skipped") and state.get("chip_grid") is not None:
        st.subheader("Chips")
        export_chips_cb = st.checkbox("Export chips", value=True, key="export_chips_cb")
        if export_chips_cb:
            chip_fmt = st.selectbox("Chip format", ["png", "jpeg", "geotiff", "npy"], key="export_chip_fmt")
            chip_naming = state.get("chip_params", {}).get("naming", "coords")
            normalise_chips = False
            if state.get("global_stats"):
                normalise_chips = st.checkbox("Normalise chips (z-score)", value=False, key="export_norm_cb")
        else:
            chip_fmt = "png"
            chip_naming = state.get("chip_params", {}).get("naming", "coords")
            normalise_chips = False

        with st.expander("View chips", expanded=True):
            from ui.tile_viewer import render_tile_viewer
            render_tile_viewer(state["chip_grid"])

        st.divider()
    else:
        export_chips_cb = False
        chip_fmt = "png"
        chip_naming = "coords"
        normalise_chips = False

    # ------------------------------------------------------------------ #
    # Section 4 — Export button
    # ------------------------------------------------------------------ #
    st.subheader("Export")

    if st.button("Export Selected", type="primary", key="export_all"):
        zip_bytes = _build_export_zip(
            state, export_params, export_image,
            img_fmt if export_image else "PNG",
            export_chips_cb, chip_fmt, chip_naming,
            normalise_chips,
        )
        if zip_bytes:
            st.download_button(
                label="Download export.zip",
                data=zip_bytes,
                file_name="export.zip",
                mime="application/zip",
                type="primary",
                key="download_export",
            )
        else:
            st.warning("Nothing selected to export.")

    st.divider()

    if st.button("← Start Over", key="review_start_over"):
        for key in list(state.keys()):
            del state[key]
        st.rerun()

    return False
