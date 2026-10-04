import streamlit as st

from transforms import run_pipeline
from ui import render_sidebar, show_download_button, show_side_by_side
from utils import decode_image

# Req 9.1: wide layout, page title
st.set_page_config(page_title="VisionLab", layout="wide")

st.title("VisionLab")

# Req 1.1: file uploader accepting JPEG/JPG/PNG
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Cache decoded image by (filename, size) to avoid re-decoding on every rerender
    file_key = (uploaded.name, uploaded.size)
    if st.session_state.get("source_key") != file_key:
        try:
            # Req 1.2: decode into in-memory NumPy array
            st.session_state["source_image"] = decode_image(uploaded.read())
            st.session_state["source_key"] = file_key
            # Req 1.3: success confirmation
            st.success("Image loaded successfully.")
        except ValueError as e:
            # Req 1.4: error on invalid/unsupported file
            st.error(f"Could not load image: {e}")

source = st.session_state.get("source_image")

if source is None:
    # Req 1.5: placeholder when no image has been uploaded
    st.info("Upload an image to get started.")
else:
    # Req 7.2: recompute pipeline on every interaction
    config = render_sidebar()
    processed = run_pipeline(source, config)  # Req 2.4: identity when no transforms enabled
    show_side_by_side(source, processed)
    show_download_button(processed)
