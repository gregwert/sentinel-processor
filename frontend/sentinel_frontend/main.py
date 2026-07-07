"""
main.py — Streamlit entry point and step router.

Session state keys (API path)
------------------------------
step : str
    Current wizard step.
_session_id : str
    UUID for the backend session.
upload_params : dict
    Band indices and stretch percentiles.
source_meta_api : dict
    Image dimensions, CRS, pixel size from backend.
stretch_result : dict
    Result from stretch job.
_upload_params_key : str
    Cache key for upload dedupe.
reference_stats : dict
    Mean/std from uploaded reference TIFFs.
_ref_stats_key : str
    Cache key for reference dedupe.
dehaze_params : dict
    DCP parameters or {'skipped': True}.
dehaze_done : bool
    True once a dehaze job has succeeded.
cloud_mask_stats : dict
    Cloud percentage from backend mask.
enhance_params : dict
    Enhancement parameters or {'skipped': True}.
enhance_done : bool
    True once enhance job has succeeded.
global_stats : dict
    {'mean': [r,g,b], 'std': [r,g,b]} from backend.
chip_params : dict
    Grid dimensions, overlap, naming, edge mode.
chip_grid_spec : dict
    {total, n_rows, n_cols} from backend.
chip_page : int
    Current page in chip viewer.
chip_quality : dict
    Accepted/rejected chip indices from filters.
chip_skipped : bool
    True when chipping step was skipped.
_export_job_id : str
    Job ID for the running/completed export.
_export_selection_key : str
    Cache key for export selection dedupe.
"""
import streamlit as st
from sentinel_frontend import api_client as api

from sentinel_frontend.ui.steps import render_breadcrumb
from sentinel_frontend.ui.steps.step_upload import render as render_upload
from sentinel_frontend.ui.steps.step_dehaze import render as render_dehaze
from sentinel_frontend.ui.steps.step_enhance import render as render_enhance
from sentinel_frontend.ui.steps.step_chip import render as render_chip
from sentinel_frontend.ui.steps.step_review import render as render_review

st.set_page_config(page_title="Sentinel Processor", layout="wide")
st.title("Sentinel-2 Imagery Processing Pipeline")


if "step" not in st.session_state:
    st.session_state["step"] = "upload"

if "_session_id" not in st.session_state:
    st.session_state["_session_id"] = api.create_session()

step = st.session_state["step"]
render_breadcrumb(step)

if step == "upload":
    if render_upload(st.session_state):
        st.session_state["step"] = "dehaze"
        st.rerun()

elif step == "dehaze":
    if render_dehaze(st.session_state):
        st.session_state["step"] = "enhance"
        st.rerun()

elif step == "enhance":
    if render_enhance(st.session_state):
        st.session_state["step"] = "chip"
        st.rerun()

elif step == "chip":
    if render_chip(st.session_state):
        st.session_state["step"] = "review"
        st.rerun()

elif step == "review":
    render_review(st.session_state)