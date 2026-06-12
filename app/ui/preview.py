"""
ui/preview.py

Streamlit component for displaying full-image stage previews.
Shows a 3-column side-by-side comparison of pipeline intermediate results
(preprocessed / dehazed / enhanced) stored in PipelineResult.stages.
"""

import streamlit as st
import numpy as np
from typing import Dict


def render_stage_preview(stages: Dict[str, np.ndarray]) -> None:
    """Render a 3-column before/after stage comparison in Streamlit.

    Args:
        stages: Dict mapping stage name to a uint8 (H, W, 3) array.
            Required keys: 'preprocessed', 'enhanced'.
            Optional key: 'dehazed' — falls back to 'preprocessed' when absent.
    """
    st.subheader("Processing Stages")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(stages["preprocessed"], caption="After Stretch", use_column_width=True)

    with col2:
        dehazed_img = stages.get("dehazed", stages["preprocessed"])
        st.image(dehazed_img, caption="After Dehaze", use_column_width=True)

    with col3:
        st.image(stages["enhanced"], caption="Final Output", use_column_width=True)

    # Per-stage metrics row
    for col, label, key in zip(
        [col1, col2, col3],
        ["Stretch", "Dehaze", "Enhance"],
        ["preprocessed", "dehazed_or_pre", "enhanced"],
    ):
        img = stages.get("dehazed", stages["preprocessed"]) if key == "dehazed_or_pre" else stages.get(key)
        if img is not None:
            with col:
                st.caption(f"μ={img.mean():.1f}  σ={img.std():.1f}")
