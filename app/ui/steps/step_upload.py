"""
Step 1 — File upload, band selection, percentile normalisation, immediate preview.
Advances to the dehaze step once the user is satisfied with the display.
"""
import streamlit as st
import tempfile
import os
import numpy as np
from PIL import Image as PILImage
from processing.preprocess import read_sentinel_tiff, percentile_stretch
from processing.reference_norm import compute_reference_stats
from utils import _yaml_safe


def _display_image(img_uint8: np.ndarray, max_px: int = 800) -> PILImage.Image:
    """Downsample img_uint8 so its longest edge is at most max_px.

    Args:
        img_uint8: uint8 (H, W, 3) array.
        max_px: Maximum size in pixels for the longer dimension.

    Returns:
        PIL Image, downsampled with LANCZOS when the original exceeds
        max_px, otherwise converted unchanged.
    """
    h, w = img_uint8.shape[:2]
    scale = min(max_px / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        return PILImage.fromarray(img_uint8).resize((new_w, new_h), PILImage.LANCZOS)
    return PILImage.fromarray(img_uint8)


def _build_params_yaml(state: dict) -> bytes:
    """Serialise current *_params keys from state to UTF-8 YAML bytes.

    Tuples are converted to lists before dumping so the output is
    compatible with yaml.safe_load.

    Args:
        state: Wizard session state dict.

    Returns:
        UTF-8 encoded YAML bytes.
    """
    import yaml
    params = {}
    for section, key in [
        ("upload", "upload_params"),
        ("dehazing", "dehaze_params"),
        ("enhancement", "enhance_params"),
        ("chipping", "chip_params"),
    ]:
        if state.get(key):
            params[section] = _yaml_safe(state[key])
    return yaml.dump(params, default_flow_style=False, sort_keys=False).encode("utf-8")


def _load_params_yaml(file_like, state: dict) -> None:
    """Parse a params.yaml and merge recognised sections into state.

    Shows st.error and returns early on malformed YAML or if the top-level
    value is not a mapping. Unknown sections are silently ignored.

    Args:
        file_like: File-like object with a .read() method (e.g. from
            st.file_uploader).
        state: Wizard session state dict. Keys upload_params, dehaze_params,
            enhance_params, and chip_params are overwritten when present.
    """
    import yaml
    try:
        data = yaml.safe_load(file_like.read())
    except Exception as exc:
        st.error(f"Could not parse YAML: {exc}")
        return
    if not isinstance(data, dict):
        st.error("Invalid params.yaml — expected a YAML mapping at the top level.")
        return
    mapping = {
        "upload": "upload_params",
        "dehazing": "dehaze_params",
        "enhancement": "enhance_params",
        "chipping": "chip_params",
    }
    for section, key in mapping.items():
        if section in data and isinstance(data[section], dict):
            state[key] = data[section]


def render(state: dict) -> bool:
    """Render Step 1 — Upload.

    Reads from state
    ----------------
    _upload_params_key : str
        Internal cache key used to detect parameter changes and avoid
        re-reading the TIFF when nothing has changed.

    Writes to state
    ---------------
    stretched_image : np.ndarray
        uint8 (H, W, 3) percentile-stretched true-colour composite.
        Updated whenever the file or any display parameter changes.
    source_meta : dict
        Rasterio metadata dict containing CRS and Affine transform.
        Updated alongside stretched_image.
    upload_params : dict
        Band indices and stretch percentiles used for the last read.
        Keys: band_indices, p_low, p_high, per_band_stretch.
    _upload_params_key : str
        Internal cache key reflecting the current file-id and parameters.

    Returns
    -------
    bool
        True when the user clicks the 'Next: Dehazing →' button to advance
        to the dehaze step. False on all other renders.
    """
    st.header("Step 1 — Upload")

    with st.expander("Load saved config (params.yaml)"):
        config_file = st.file_uploader(
            "Upload a params.yaml from a previous export",
            type=["yaml", "yml"],
            key="config_loader",
        )
        if config_file is not None:
            _load_params_yaml(config_file, state)
            st.success("Config loaded — parameters pre-populated for each step.")

    uploaded_file = st.file_uploader("Upload a Sentinel-2 TIFF", type=["tif", "tiff"])

    if uploaded_file is not None:
        with st.expander("Band & display settings"):
            band_r = st.number_input("Red band", min_value=1, max_value=12, value=1)
            st.caption("*Band index to map to the red channel. Sentinel-2 L2A band 4 is the red reflectance band.*")
            band_g = st.number_input("Green band", min_value=1, max_value=12, value=2)
            st.caption("*Band index to map to the green channel. Sentinel-2 L2A band 3 is the green reflectance band.*")
            band_b = st.number_input("Blue band", min_value=1, max_value=12, value=3)
            st.caption("*Band index to map to the blue channel. Sentinel-2 L2A band 2 is the blue reflectance band.*")
            p_low = st.slider("Low percentile clip", 0.0, 10.0, 2.0, 0.5)
            st.caption("*Pixels below this brightness percentile are clipped to black. Raise to brighten the image by discarding the darkest values (shadows, deep water). 2.0 is a good default.*")
            p_high = st.slider("High percentile clip", 90.0, 100.0, 98.0, 0.5)
            st.caption("*Pixels above this brightness percentile are clipped to white. Lower to brighten the image by discarding the brightest values (clouds, sunglint). 98.0 is a good default.*")
            per_band = st.checkbox("Per-band normalisation", value=False)
            st.caption("*Normalise each band independently before combining into RGB. Can cause colour casts on Sentinel-2 true-colour composites because the bands are stretched to different ranges — leave off unless the image looks very green or very red.*")

        band_indices = (int(band_r), int(band_g), int(band_b))
        params_key = str((uploaded_file.file_id, band_r, band_g, band_b, p_low, p_high, per_band))

        if state.get("_upload_params_key") != params_key or "stretched_image" not in state:
            state.pop("reference_stats", None)
            state.pop("_ref_stats_key", None)
            try:
                suffix = os.path.splitext(uploaded_file.name)[-1] or ".tif"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    arr, meta = read_sentinel_tiff(tmp_path, band_indices=band_indices)
                    stretched = percentile_stretch(arr, p_low=p_low, p_high=p_high, per_band=per_band)
                finally:
                    os.unlink(tmp_path)

                state["stretched_image"] = stretched
                state["source_meta"] = meta
                state["upload_params"] = {
                    "band_indices": band_indices,
                    "p_low": p_low,
                    "p_high": p_high,
                    "per_band_stretch": per_band,
                }
                state["_upload_params_key"] = params_key

            except Exception as e:
                st.error(f"Failed to read TIFF: {e}")
                return False

        if "stretched_image" in state:
            img = state["stretched_image"]
            st.image(_display_image(state["stretched_image"]), use_column_width=True)

            meta = state["source_meta"]
            crs_str = str(meta["crs"]) if meta.get("crs") else "Unknown"
            px_size = abs(meta["transform"].a)
            st.caption(
                f"{img.shape[1]}×{img.shape[0]} px  |  CRS: {crs_str}  |  Pixel size: {px_size:.4f}"
            )

            st.divider()
            st.subheader("Reference Images (optional)")
            st.caption(
                "*Upload one or more cloudless Sentinel-2 TIFFs of the same area acquired at "
                "different dates. Band selection and stretch settings from above are applied to "
                "each reference. Statistics are averaged across all references and used in Step 3 "
                "to anchor the target image's radiometry.*"
            )

            ref_files = st.file_uploader(
                "Upload reference Sentinel-2 TIFFs",
                type=["tif", "tiff"],
                accept_multiple_files=True,
                key="ref_uploader",
            )

            if ref_files:
                _ref_files_key = str(sorted([f.file_id for f in ref_files]))
                _ref_params_fragment = str((band_r, band_g, band_b, p_low, p_high, per_band))
                _ref_stats_key = str((_ref_files_key, _ref_params_fragment))

                if state.get("_ref_stats_key") != _ref_stats_key:
                    stretched_refs = []
                    for rf in ref_files:
                        try:
                            suffix = os.path.splitext(rf.name)[-1] or ".tif"
                            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                                tmp.write(rf.read())
                                tmp_path = tmp.name
                            try:
                                arr_ref, _ = read_sentinel_tiff(tmp_path, band_indices=band_indices)
                                stretched_refs.append(
                                    percentile_stretch(arr_ref, p_low=p_low, p_high=p_high, per_band=per_band)
                                )
                            except Exception as exc:
                                st.error(f"Failed to read reference TIFF '{rf.name}': {exc}")
                            finally:
                                os.unlink(tmp_path)
                        except Exception as exc:
                            st.error(f"Failed to load '{rf.name}': {exc}")

                    if stretched_refs:
                        state["reference_stats"] = compute_reference_stats(stretched_refs)
                        state["_ref_stats_key"] = _ref_stats_key
                    else:
                        state.pop("reference_stats", None)
                        state.pop("_ref_stats_key", None)

                if "reference_stats" in state:
                    rs = state["reference_stats"]
                    m = rs["mean"]
                    st.success(
                        f"{rs['n']} reference image{'s' if rs['n'] != 1 else ''} loaded  ·  "
                        f"mean R/G/B: {m[0]:.1f} / {m[1]:.1f} / {m[2]:.1f}"
                    )
            else:
                state.pop("reference_stats", None)
                state.pop("_ref_stats_key", None)

            if any(k in state for k in ("upload_params", "dehaze_params", "enhance_params", "chip_params")):
                st.download_button(
                    "⬇ Download current config (params.yaml)",
                    data=_build_params_yaml(state),
                    file_name="params.yaml",
                    mime="text/yaml",
                    key="download_config",
                )

            if st.button("Next: Dehazing →", type="primary"):
                return True

    return False
