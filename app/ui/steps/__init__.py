import streamlit as st


def render_step_nav(step_name: str) -> tuple:
    """Render the Back / Skip button row and return click state.

    Parameters
    ----------
    step_name : str
        Short identifier used to generate unique widget keys (e.g. 'dehaze').

    Returns
    -------
    tuple of (bool, bool)
        (back_clicked, skip_clicked). Exactly one will be True on any given
        render; both are False if neither button was pressed this frame.
    """
    col_back, _, col_skip = st.columns([1, 4, 1])
    with col_back:
        back = st.button("← Back", key=f"back_{step_name}")
    with col_skip:
        skip = st.button("Skip →", key=f"skip_{step_name}")
    return back, skip
