import streamlit as st

from ui.steps.step_upload import render as render_upload
from ui.steps.step_dehaze import render as render_dehaze
from ui.steps.step_enhance import render as render_enhance
from ui.steps.step_chip import render as render_chip
from ui.steps.step_review import render as render_review

st.set_page_config(page_title="Sentinel Processor", layout="wide")
st.title("Sentinel-2 Imagery Processing Pipeline")

STEPS = ["upload", "dehaze", "enhance", "chip", "review"]
STEP_LABELS = {
    "upload": "1. Upload",
    "dehaze": "2. Dehaze",
    "enhance": "3. Enhance",
    "chip": "4. Chip",
    "review": "5. Review",
}

def render_breadcrumb(current: str):
    cols = st.columns(len(STEPS))
    for col, step in zip(cols, STEPS):
        label = STEP_LABELS[step]
        if step == current:
            col.markdown(f"**:blue[{label}]**")
        elif STEPS.index(step) < STEPS.index(current):
            col.markdown(f"<span style='color:grey'>✓ {label}</span>", unsafe_allow_html=True)
        else:
            col.markdown(f"{label}")
    st.divider()


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
