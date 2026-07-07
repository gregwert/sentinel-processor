import streamlit as st

STEPS = ["upload", "dehaze", "enhance", "chip", "review"]
STEP_LABELS = {
    "upload": "1. Upload",
    "dehaze": "2. Dehaze",
    "enhance": "3. Enhance",
    "chip": "4. Chip",
    "review": "5. Review",
}

def render_breadcrumb(current: str) -> None:
    """Render the 5-step wizard breadcrumb as a Streamlit column layout.

    The active step is shown in bold blue; completed steps appear in grey
    with a checkmark; future steps are unstyled.

    Args:
        current: Step identifier — one of 'upload', 'dehaze', 'enhance',
            'chip', 'review'.
    """
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


def render_step_nav(step_name: str) -> tuple:
    """Render the Back / Skip button row and return click state.

    Args:
        step_name: Short identifier used to generate unique widget keys
            (e.g. 'dehaze').

    Returns:
        Tuple of (back_clicked, skip_clicked). Both are False when neither
        button was pressed this frame.
    """
    col_back, _, col_skip = st.columns([1, 4, 1])
    with col_back:
        back = st.button("← Back", key=f"back_{step_name}")
    with col_skip:
        skip = st.button("Skip →", key=f"skip_{step_name}")
    return back, skip
