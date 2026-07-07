"""
Step 1 — File upload, band selection, percentile normalisation, immediate preview.
Advances to the dehaze step once the user is satisfied with the display.
"""
import streamlit as st
from sentinel_frontend import api_client as api

from sentinel_frontend.utils import _yaml_safe


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


def _validate_params_section(state_key: str, data: dict) -> dict:
    # Numeric rules: (coerce_type, min, max) — bounds match the widget bounds exactly.
    NUMERIC = {
        "upload_params": {
            "p_low":  (float, 0.0, 10.0),
            "p_high": (float, 90.0, 100.0),
        },
        "dehaze_params": {
            "omega":              (float, 0.5,  1.0),
            "t0":                 (float, 0.05, 0.5),
            "patch_size":         (int,   5,    31),
            "brightness_thresh":  (float, 0.5,  0.95),
            "saturation_thresh":  (float, 0.01, 0.20),
        },
        "enhance_params": {
            "clip_limit": (float, 1.0, 10.0),
        },
        "chip_params": {
            "chip_w":   (int,   64,  2048),
            "chip_h":   (int,   64,  2048),
            "overlap":  (float, 0.0, 0.99),
        },
    }
    # Enum rules: allowed string values.
    ENUM = {
        "enhance_params": {
            "method":          {"clahe", "gray_world", "ref_norm", "none"},
            "ref_norm_method": {"histogram", "linear"},
            "tile_size":       {"4", "8", "16", "32"},
        },
        "chip_params": {
            "naming":    {"rowcol", "coords"},
            "edge_mode": {"pad", "overlap"},
        },
    }
    # Boolean keys: must be actual Python bools (yaml.safe_load gives true/false as bool).
    BOOL_KEYS = {
        "dehaze_params":  {"mask_clouds", "use_guided", "use_guided_filter"},
        "upload_params":  {"per_band_stretch"},
        "enhance_params": {},
        "chip_params":    {},
    }
    num_rules  = NUMERIC.get(state_key, {})
    enum_rules = ENUM.get(state_key, {})
    bool_keys  = BOOL_KEYS.get(state_key, set())
    out = {}
    for k, v in data.items():
        if k in bool_keys:
            if not isinstance(v, bool):
                st.warning(f"Ignoring invalid config value '{k}' in {state_key}: {v!r} must be true or false.")
                continue
        elif k in num_rules:
            coerce, lo, hi = num_rules[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                st.warning(f"Ignoring invalid config value '{k}' in {state_key}: {v!r} is not numeric.")
                continue
            v = coerce(v)
            if not (lo <= v <= hi):
                st.warning(f"Ignoring out-of-range config value '{k}' in {state_key}: {v!r} must be between {lo} and {hi}.")
                continue
        elif k in enum_rules:
            if str(v) not in enum_rules[k]:
                st.warning(f"Ignoring invalid config value '{k}' in {state_key}: {v!r} must be one of {sorted(enum_rules[k])}.")
                continue
        out[k] = v
    return out


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
            state[key] = _validate_params_section(key, data[section])


def render(state: dict) -> bool:
    """Render Step 1 — Upload.

    Reads from state
    ----------------
    _upload_params_key : str
        Cache key used to detect parameter changes and avoid re-uploading
        when nothing has changed.
    upload_params : dict
        Pre-populated from a previously loaded params.yaml (if any).

    Writes to state
    ---------------
    upload_params : dict
        Band indices and stretch percentiles used for the last read.
    stretch_result : dict
        Result payload from the stretch job.
    source_meta_api : dict
        Image dimensions, CRS, and pixel size returned by the backend.
    _upload_params_key : str
        Cache key reflecting the current file-id and parameters.
    reference_stats : dict
        Mean/std computed over uploaded reference TIFFs.
    _ref_stats_key : str
        Cache key for reference upload dedupe.

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

        session_id = state["_session_id"]

        if state.get("_upload_params_key") != params_key or "stretch_result" not in state:
            state.pop("reference_stats", None)
            state.pop("_ref_stats_key", None)
            try:
                file_bytes = uploaded_file.read()
                api.upload_source(session_id, file_bytes, uploaded_file.name)
                state["upload_params"] = {
                    "band_indices": band_indices,
                    "p_low": p_low,
                    "p_high": p_high,
                    "per_band_stretch": per_band,
                }
                job_id = api.run_stretch(session_id, band_indices, p_low, p_high, per_band)
                result = api.poll_job(job_id, "Stretching...")
                state["stretch_result"] = result
                state["source_meta_api"] = result.get("meta", {})
                state["_upload_params_key"] = params_key
            except Exception as e:
                st.error(f"Failed to process TIFF: {e}")
                return False

        if "stretch_result" in state:
            meta = state.get("source_meta_api", {})
            preview_bytes = api.fetch_preview(session_id, "stretched")
            st.image(preview_bytes, use_container_width=True)
            crs_str = meta.get("crs_wkt") or "Unknown"
            px_size = meta.get("pixel_size", 0.0)
            width = meta.get("width", "?")
            height = meta.get("height", "?")
            st.caption(
                f"{width}×{height} px  |  CRS: {crs_str}  |  Pixel size: {px_size:.4f}"
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
                _ref_stats_key = str((_ref_files_key, params_key))
                if state.get("_ref_stats_key") != _ref_stats_key:
                    try:
                        files_payload = [(rf.name, rf.read()) for rf in ref_files]
                        ref_result = api.upload_references(session_id, files_payload)
                        state["reference_stats"] = ref_result.get("stats")
                        state["_ref_stats_key"] = _ref_stats_key
                    except Exception as exc:
                        st.error(f"Failed to upload references: {exc}")
                if state.get("reference_stats"):
                    rs = state["reference_stats"]
                    m = rs.get("mean", [0, 0, 0])
                    n = rs.get("n", len(ref_files))
                    st.success(
                        f"{n} reference image{'s' if n != 1 else ''} loaded  ·  "
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