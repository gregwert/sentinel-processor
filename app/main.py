"""
main.py — Streamlit entry point and step router.

Session state keys
------------------
step : str
    Current wizard step. One of: 'upload', 'dehaze', 'enhance', 'chip', 'review'.
stretched_image : np.ndarray  (H, W, 3) uint8
    Written by step_upload. Percentile-stretched true-colour composite.
source_meta : dict
    Written by step_upload. Rasterio metadata with CRS and Affine transform.
upload_params : dict
    Written by step_upload. Band indices and stretch percentiles used.
cloud_mask : np.ndarray  (H, W) bool
    Written by step_dehaze (live, on every threshold slider change).
dehazed_image : np.ndarray  (H, W, 3) uint8
    Written by step_dehaze on Run click, or copied from stretched_image on skip.
dehaze_params : dict
    Written by step_dehaze. DCP parameters, or {'skipped': True}.
enhanced_image : np.ndarray  (H, W, 3) uint8
    Written by step_enhance on Apply click, or copied from dehazed_image on skip.
enhance_params : dict
    Written by step_enhance. Enhancement method and params, or {'skipped': True}.
global_stats : dict or None
    Written by step_enhance. {'mean': [r,g,b], 'std': [r,g,b]} of enhanced_image.
    None when enhancement was skipped.
chip_grid : ChipGrid
    Written by step_chip on Run Chipping click.
chip_params : dict
    Written by step_chip. Grid dimensions, overlap, naming, edge mode.
chip_skipped : bool
    Written by step_chip Skip button. True when chipping step was skipped.
"""
import streamlit as st

from ui.steps import STEPS, STEP_LABELS, render_breadcrumb
from ui.steps.step_upload import render as render_upload
from ui.steps.step_dehaze import render as render_dehaze
from ui.steps.step_enhance import render as render_enhance
from ui.steps.step_chip import render as render_chip
from ui.steps.step_review import render as render_review

st.set_page_config(page_title="Sentinel Processor", layout="wide")
st.title("Sentinel-2 Imagery Processing Pipeline")


if "step" not in st.session_state:
    st.session_state["step"] = "upload"

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
