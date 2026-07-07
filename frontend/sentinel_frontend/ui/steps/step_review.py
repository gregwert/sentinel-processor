"""
Step 5 — Review & Export.
Shows processing parameters, processed image, and chips (if not skipped).
All exports are selected via checkboxes then generated together as a single zip.
"""
import os
import streamlit as st
import yaml
from sentinel_frontend import api_client as api

from sentinel_frontend.utils import _yaml_safe


def render(state: dict) -> bool:
    """Render Step 5 — Review & Export.

    Reads from state
    ----------------
    upload_params : dict
        Band indices and stretch percentiles. Included in the YAML export.
    dehaze_params : dict
        DCP parameters or {'skipped': True}. Included in the YAML export.
    enhance_params : dict
        Enhancement method and params or {'skipped': True}. Included in
        the YAML export.
    chip_params : dict
        Grid dimensions, overlap, naming, and edge mode. Included in the
        YAML export.
    chip_skipped : bool
        When True the chips section is hidden and no chips are exported.
    chip_grid_spec : dict
        {total, n_rows, n_cols} from step_chip. Controls whether the
        chips export section is shown.
    chip_quality : dict
        Accepted/rejected chip indices. Exposed in the export options.
    global_stats : dict
        {'mean': [r,g,b], 'std': [r,g,b]} from step_enhance. Exposed as
        an optional z-score normalisation pass during chip export.
    reference_stats : dict
        Mean/std from reference TIFFs. Exposed as an optional per-chip
        normalisation pass during chip export.
    _export_job_id : str
        Job ID for a completed export — enables the download button.
    _export_selection_key : str
        Cache key for export selection dedupe.

    Writes to state
    ---------------
    _export_job_id : str
        Set on successful Export Selected click.
    _export_selection_key : str
        Updated whenever the export selection changes.
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

    session_id = state["_session_id"]

    # ------------------------------------------------------------------ #
    # Section 2 — Processed image
    # ------------------------------------------------------------------ #
    st.subheader("Processed Image")
    export_image = st.checkbox("Export processed image", value=True, key="export_image_cb")
    img_fmt = "PNG"
    if export_image:
        img_fmt = st.radio("Image format", ["PNG", "GeoTIFF"], horizontal=True, key="export_image_fmt")

    with st.expander("View image", expanded=True):
        st.image(api.fetch_best_preview(session_id), use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------ #
    # Section 3 — Chips
    # ------------------------------------------------------------------ #
    _has_chips = not state.get("chip_skipped") and state.get("chip_grid_spec") is not None
    if _has_chips:
        st.subheader("Chips")
        export_chips_cb = st.checkbox("Export chips", value=True, key="export_chips_cb")
        chip_fmt = "png"
        chip_naming = state.get("chip_params", {}).get("naming", "coords")
        normalise_chips = False
        apply_ref_norm_chips = False
        ref_norm_method_export = "histogram"
        include_rejected = False
        export_coco = False
        export_yolo = False

        if export_chips_cb:
            chip_fmt = st.selectbox("Chip format", ["png", "jpeg", "geotiff", "npy"], key="export_chip_fmt")
            if state.get("global_stats"):
                normalise_chips = st.checkbox("Normalise chips (z-score)", value=False, key="export_norm_cb")
            if state.get("reference_stats"):
                apply_ref_norm_chips = st.checkbox(
                    "Apply reference normalisation per chip", value=False, key="export_ref_norm_cb"
                )
                if apply_ref_norm_chips:
                    _rn_options = ["Histogram matching", "Linear (mean/std)"]
                    _rn_choice = st.radio("Method", _rn_options, index=0, horizontal=True, key="export_ref_norm_method")
                    ref_norm_method_export = "histogram" if _rn_choice == "Histogram matching" else "linear"
            chip_quality = state.get("chip_quality", {})
            if chip_quality.get("rejected"):
                include_rejected = st.checkbox(
                    f"Include {len(chip_quality['rejected'])} rejected chips in export",
                    value=False,
                    key="export_include_rejected",
                )

        st.markdown("**Annotation export**")
        export_coco = st.checkbox("COCO JSON manifest", value=False, key="export_coco_cb")
        st.caption("*Exports chip image entries with geographic bounding boxes in COCO JSON format.*")
        export_yolo = st.checkbox("YOLO labels + dataset.yaml", value=False, key="export_yolo_cb")
        st.caption("*Exports empty per-chip .txt label files and a dataset.yaml descriptor.*")

        with st.expander("View chips", expanded=True):
            chip_data = {}
            try:
                chip_data = api.list_chips(session_id, page=0, page_size=16)
            except Exception as exc:
                st.caption(f"Chip preview unavailable: {exc}")
            items = chip_data.get("items", [])
            cols = st.columns(4)
            for i, item in enumerate(items):
                try:
                    thumb_bytes = api.fetch_thumbnail(session_id, item["index"])
                    cols[i % 4].image(thumb_bytes, caption=f"r{item['row']} c{item['col']}", use_container_width=True)
                except Exception:
                    cols[i % 4].caption(f"Chip {item['index']}")

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

    _export_sel = str((
        export_params, export_image, img_fmt,
        export_chips_cb, chip_fmt, chip_naming,
        normalise_chips, apply_ref_norm_chips, ref_norm_method_export,
        include_rejected, export_coco, export_yolo,
    ))
    if state.get("_export_selection_key") != _export_sel:
        state.pop("_export_job_id", None)
        state["_export_selection_key"] = _export_sel

    if st.button("Export Selected", type="primary", key="export_all"):
        export_request = {
            "include_params": export_params,
            "image_include": export_image,
            "image_format": img_fmt.lower(),
            "chips_include": export_chips_cb,
            "chips_format": chip_fmt,
            "chips_naming": chip_naming,
            "chips_zscore_normalise": normalise_chips,
            "chips_ref_norm_enabled": apply_ref_norm_chips,
            "chips_ref_norm_method": ref_norm_method_export,
            "chips_include_rejected": include_rejected,
            "annotations_coco": export_coco,
            "annotations_yolo": export_yolo,
        }
        try:
            job_id = api.run_export(session_id, export_request)
            api.poll_job(job_id, "Assembling export archive...")
            state["_export_job_id"] = job_id
        except Exception as exc:
            st.error(f"Export failed: {exc}")

    if state.get("_export_job_id"):
        try:
            zip_bytes = api.download_export(session_id, state["_export_job_id"])
            st.download_button(
                label="Download export.zip",
                data=zip_bytes,
                file_name="export.zip",
                mime="application/zip",
                type="primary",
                key="download_export_api",
            )
        except Exception as exc:
            st.error(f"Download failed: {exc}")

    st.divider()

    if st.button("← Start Over", key="review_start_over"):
        if state.get("_session_id"):
            try:
                api.delete_session(state["_session_id"])
            except Exception:
                pass
        for key in list(state.keys()):
            del state[key]
        st.rerun()

    return False