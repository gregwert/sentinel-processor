"""
Step 5 — Review & Export.
Shows processing parameters, processed image, and chips (if not skipped).
All exports are selected via checkboxes then generated together as a single zip.
"""
import streamlit as st
import io, json, os, zipfile, tempfile
import yaml
from PIL import Image as PILImage
from utils import _yaml_safe


def _add_params_section(zf: zipfile.ZipFile, state: dict) -> bool:
    """Write params.yaml into an open ZipFile.

    Args:
        zf: Open ZipFile to write into.
        state: Wizard session state dict.

    Returns:
        True (always writes an entry).
    """
    params = {}
    if state.get("upload_params"):
        params["upload"] = state["upload_params"]
    if state.get("dehaze_params"):
        params["dehazing"] = state["dehaze_params"]
    if state.get("enhance_params"):
        params["enhancement"] = state["enhance_params"]
    if state.get("chip_params"):
        params["chipping"] = state["chip_params"]
    yaml_bytes = yaml.dump(_yaml_safe(params), default_flow_style=False, sort_keys=False).encode("utf-8")
    zf.writestr("params.yaml", yaml_bytes)
    return True


def _add_image_section(zf: zipfile.ZipFile, state: dict, img_fmt: str) -> bool:
    """Write the processed image into an open ZipFile.

    Args:
        zf: Open ZipFile to write into.
        state: Wizard session state dict. Reads 'enhanced_image' and
            'source_meta'.
        img_fmt: 'PNG' or 'GeoTIFF'.

    Returns:
        True if the image was written, False if enhanced_image is absent.
    """
    img = state.get("enhanced_image")
    if img is None:
        return False
    if img_fmt == "PNG":
        img_buf = io.BytesIO()
        PILImage.fromarray(img).save(img_buf, format="PNG")
        zf.writestr("processed_image.png", img_buf.getvalue())
        return True
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
            return True
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    return False


def _add_chips_section(
    zf: zipfile.ZipFile,
    state: dict,
    chip_fmt: str,
    chip_naming: str,
    normalise_chips: bool,
    apply_ref_norm_chips: bool,
    ref_norm_method: str,
    include_rejected: bool,
    export_coco: bool,
    export_yolo: bool,
) -> bool:
    """Write chips, manifest CSV, and optional annotation files into an open ZipFile.

    Args:
        zf: Open ZipFile to write into.
        state: Wizard session state dict. Reads 'chip_grid', 'chip_quality',
            'global_stats', and 'reference_stats'.
        chip_fmt: Chip format — 'png', 'jpeg', 'geotiff', or 'npy'.
        chip_naming: Chip naming scheme — 'rowcol' or 'coords'.
        normalise_chips: Apply z-score normalisation per chip.
        apply_ref_norm_chips: Apply reference normalisation per chip.
        ref_norm_method: Reference normalisation method — 'histogram' or 'linear'.
        include_rejected: Include quality-rejected chips in the export.
        export_coco: Include a COCO JSON annotation manifest.
        export_yolo: Include YOLO label files and dataset.yaml.

    Returns:
        True if at least one file was added to zf, False if chip_grid is absent.
    """
    if state.get("chip_grid") is None:
        return False

    from chipping.tile_exporter import ExportConfig, export_chips as _export_chips_fn
    from chipping.manifest import build_manifest, write_manifest_csv

    rejected_indices = state.get("chip_quality", {}).get("rejected")
    ext_map = {"png": ".png", "jpeg": ".jpg", "geotiff": ".tif", "npy": ".npy"}
    fmt_ext = ext_map.get(chip_fmt, ".png")
    added = False

    config = ExportConfig(
        fmt=chip_fmt,
        naming=chip_naming,
        normalise=normalise_chips,
        global_stats=state.get("global_stats"),
        apply_ref_norm=apply_ref_norm_chips,
        reference_stats=state.get("reference_stats") if apply_ref_norm_chips else None,
        ref_norm_method=ref_norm_method,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = _export_chips_fn(
            state["chip_grid"], tmp_dir,
            config=config,
            rejected_indices=rejected_indices,
            include_rejected=include_rejected,
        )
        for path in paths:
            zf.write(path, arcname=os.path.join("chips", os.path.basename(path)))
        if paths:
            added = True

        chip_stats = state.get("chip_quality", {}).get("stats")
        manifest_rows = build_manifest(state["chip_grid"], chip_stats, chip_naming, fmt_ext)
        rejected_set = set(state.get("chip_quality", {}).get("rejected", []))
        if not include_rejected:
            manifest_rows = [r for r in manifest_rows if r["rejected"] == "false" or int(r["chip_index"]) not in rejected_set]
        zf.writestr("chips/manifest.csv", write_manifest_csv(manifest_rows))

        if export_coco:
            from chipping.annotation_export import build_coco_manifest
            coco_dict = build_coco_manifest(
                state["chip_grid"],
                state.get("chip_quality", {}).get("stats"),
                chip_naming, fmt_ext, rejected_indices, include_rejected,
            )
            zf.writestr("annotations/coco.json", json.dumps(coco_dict, indent=2).encode("utf-8"))
            added = True

        if export_yolo:
            from chipping.annotation_export import build_yolo_files
            yolo_files = build_yolo_files(
                state["chip_grid"],
                state.get("chip_quality", {}).get("stats"),
                chip_naming, fmt_ext, rejected_indices, include_rejected,
            )
            for arc_path, content in yolo_files.items():
                zf.writestr(arc_path, content)
            added = True

    return added


def _build_export_zip(state, export_params, export_image, img_fmt,
                      export_chips, chip_fmt, chip_naming, normalise_chips,
                      apply_ref_norm_chips: bool = False,
                      ref_norm_method: str = "histogram",
                      include_rejected: bool = False,
                      export_coco: bool = False,
                      export_yolo: bool = False) -> "bytes | None":
    """Assemble the selected export artefacts into an in-memory ZIP.

    Args:
        state: Wizard session state dict.
        export_params: Include params.yaml in the archive.
        export_image: Include the processed image.
        img_fmt: Image format — 'PNG' or 'GeoTIFF'.
        export_chips: Include chip files.
        chip_fmt: Chip format — 'png', 'jpeg', 'geotiff', or 'npy'.
        chip_naming: Chip naming scheme — 'rowcol' or 'coords'.
        normalise_chips: Apply z-score normalisation per chip.
        apply_ref_norm_chips: Apply reference normalisation per chip.
        ref_norm_method: Reference normalisation method — 'histogram' or 'linear'.
        include_rejected: Include quality-rejected chips in the export.
        export_coco: Include a COCO JSON annotation manifest.
        export_yolo: Include YOLO label files and dataset.yaml.

    Returns:
        ZIP bytes when at least one artefact was added, otherwise None.
    """
    buf = io.BytesIO()
    added = False
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        if export_params:
            added |= _add_params_section(zf, state)
        if export_image:
            added |= _add_image_section(zf, state, img_fmt)
        if export_chips:
            added |= _add_chips_section(
                zf, state, chip_fmt, chip_naming, normalise_chips,
                apply_ref_norm_chips, ref_norm_method, include_rejected,
                export_coco, export_yolo,
            )
    return buf.getvalue() if added else None


def render(state: dict) -> bool:
    """Render Step 5 — Review & Export.

    Reads from state
    ----------------
    upload_params : dict
        Band indices and stretch percentiles from step_upload. Included
        in the parameters YAML export when present.
    dehaze_params : dict
        DCP parameters or {'skipped': True} from step_dehaze. Included
        in the parameters YAML export when present.
    enhance_params : dict
        Enhancement method and params or {'skipped': True} from
        step_enhance. Included in the parameters YAML export when present.
    chip_params : dict
        Grid dimensions, overlap, naming, and edge mode from step_chip.
        Included in the parameters YAML export when present.
    enhanced_image : np.ndarray
        uint8 (H, W, 3) final processed image. Displayed and exported as
        PNG or GeoTIFF.
    source_meta : dict
        Rasterio metadata dict used when exporting a GeoTIFF.
    chip_grid : ChipGrid or None
        Built chip grid from step_chip. When present the chips section
        is shown and chips can be exported.
    chip_skipped : bool
        When True the chips section is hidden and no chips are exported.
    global_stats : dict or None
        {'mean': [r,g,b], 'std': [r,g,b]} from step_enhance. Exposed as
        an optional z-score normalisation pass during chip export.

    Writes to state
    ---------------
    (all keys) : cleared
        When the user clicks '← Start Over' every key in state is deleted
        so the wizard resets to step_upload.

    Returns
    -------
    bool
        Always returns False. The Start Over button clears state and
        calls st.rerun() directly rather than signalling via return value.
    """

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
        st.code(yaml.dump(_yaml_safe(params), default_flow_style=False, sort_keys=False), language="yaml")

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
            apply_ref_norm_chips = False
            ref_norm_method_export = "histogram"
            if state.get("reference_stats"):
                apply_ref_norm_chips = st.checkbox(
                    "Apply reference normalisation per chip",
                    value=False,
                    key="export_ref_norm_cb",
                )
                if apply_ref_norm_chips:
                    _rn_options = ["Histogram matching", "Linear (mean/std)"]
                    _rn_choice = st.radio(
                        "Method",
                        _rn_options,
                        index=0,
                        horizontal=True,
                        key="export_ref_norm_method",
                    )
                    ref_norm_method_export = "histogram" if _rn_choice == "Histogram matching" else "linear"
                    st.caption(
                        "*Applied per-chip using pre-computed reference statistics from Step 1. "
                        "Independent of any reference normalisation applied to the full image in Step 3.*"
                    )
            chip_quality = state.get("chip_quality", {})
            include_rejected = False
            if chip_quality.get("rejected"):
                include_rejected = st.checkbox(
                    f"Include {len(chip_quality['rejected'])} rejected chips in export",
                    value=False,
                    key="export_include_rejected",
                )
        else:
            chip_fmt = "png"
            chip_naming = state.get("chip_params", {}).get("naming", "coords")
            normalise_chips = False
            apply_ref_norm_chips = False
            ref_norm_method_export = "histogram"
            include_rejected = False

        st.markdown("**Annotation export**")
        export_coco = st.checkbox("COCO JSON manifest", value=False, key="export_coco_cb")
        st.caption("*Exports chip image entries with geographic bounding boxes in COCO JSON format. Annotations are empty — suitable as a dataset manifest.*")
        export_yolo = st.checkbox("YOLO labels + dataset.yaml", value=False, key="export_yolo_cb")
        st.caption("*Exports empty per-chip .txt label files and a dataset.yaml descriptor. Add bounding box annotations to the .txt files before training.*")

        with st.expander("View chips", expanded=True):
            from ui.tile_viewer import render_tile_viewer
            render_tile_viewer(state["chip_grid"], quality=state.get("chip_quality"))

        st.divider()
    else:
        export_chips_cb = False
        chip_fmt = "png"
        chip_naming = "coords"
        normalise_chips = False
        apply_ref_norm_chips = False
        ref_norm_method_export = "histogram"
        include_rejected = False
        export_coco = False
        export_yolo = False

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
            apply_ref_norm_chips=apply_ref_norm_chips,
            ref_norm_method=ref_norm_method_export,
            include_rejected=include_rejected,
            export_coco=export_coco,
            export_yolo=export_yolo,
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
