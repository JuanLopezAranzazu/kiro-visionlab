import cv2
import numpy as np
import streamlit as st

from transforms import PipelineConfig
from utils import encode_image_png


def render_sidebar() -> PipelineConfig:
    """
    Render tabbed sidebar (Color | Blur | Edges) and return a PipelineConfig
    populated from current widget values.
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 5.1, 5.2, 5.3, 6.1, 6.2, 6.3
    """
    with st.sidebar:
        color_tab, blur_tab, edges_tab = st.tabs(["Color", "Blur", "Edges"])

        # --- Color tab (Req 3.2) ---
        with color_tab:
            grayscale_enabled = st.checkbox("Grayscale", value=False)

        # --- Blur tab (Req 3.3) ---
        with blur_tab:
            # Gaussian Blur (Req 5.1)
            gaussian_enabled = st.checkbox("Gaussian Blur", value=False)
            gaussian_ksize = st.slider(
                "Kernel size",
                min_value=1,
                max_value=31,
                value=5,
                step=2,
                key="gaussian_ksize",
                disabled=not gaussian_enabled,
            )
            st.divider()
            # Median Blur (Req 5.2)
            median_enabled = st.checkbox("Median Blur", value=False)
            median_ksize = st.slider(
                "Kernel size",
                min_value=1,
                max_value=31,
                value=5,
                step=2,
                key="median_ksize",
                disabled=not median_enabled,
            )
            st.divider()
            # Bilateral Filter (Req 5.3)
            bilateral_enabled = st.checkbox("Bilateral Filter", value=False)
            bilateral_d = st.slider(
                "Diameter",
                min_value=1,
                max_value=20,
                value=9,
                key="bilateral_d",
                disabled=not bilateral_enabled,
            )
            bilateral_sigma_color = st.slider(
                "Sigma Color",
                min_value=10,
                max_value=150,
                value=75,
                key="bilateral_sigma_color",
                disabled=not bilateral_enabled,
            )
            bilateral_sigma_space = st.slider(
                "Sigma Space",
                min_value=10,
                max_value=150,
                value=75,
                key="bilateral_sigma_space",
                disabled=not bilateral_enabled,
            )

        # --- Edges tab (Req 3.4) ---
        with edges_tab:
            # Canny (Req 6.1)
            canny_enabled = st.checkbox("Canny", value=False)
            canny_low = st.slider(
                "Low Threshold",
                min_value=0,
                max_value=255,
                value=50,
                key="canny_low",
                disabled=not canny_enabled,
            )
            canny_high = st.slider(
                "High Threshold",
                min_value=0,
                max_value=255,
                value=150,
                key="canny_high",
                disabled=not canny_enabled,
            )
            st.divider()
            # Sobel (Req 6.2)
            sobel_enabled = st.checkbox("Sobel", value=False)
            sobel_ksize = st.slider(
                "Kernel size",
                min_value=1,
                max_value=31,
                value=3,
                step=2,
                key="sobel_ksize",
                disabled=not sobel_enabled,
            )
            st.divider()
            # Laplacian (Req 6.3)
            laplacian_enabled = st.checkbox("Laplacian", value=False)
            laplacian_ksize = st.slider(
                "Kernel size",
                min_value=1,
                max_value=31,
                value=3,
                step=2,
                key="laplacian_ksize",
                disabled=not laplacian_enabled,
            )

    return PipelineConfig(
        grayscale_enabled=grayscale_enabled,
        gaussian_enabled=gaussian_enabled,
        gaussian_ksize=gaussian_ksize,
        median_enabled=median_enabled,
        median_ksize=median_ksize,
        bilateral_enabled=bilateral_enabled,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=float(bilateral_sigma_color),
        bilateral_sigma_space=float(bilateral_sigma_space),
        canny_enabled=canny_enabled,
        canny_low=canny_low,
        canny_high=canny_high,
        sobel_enabled=sobel_enabled,
        sobel_ksize=sobel_ksize,
        laplacian_enabled=laplacian_enabled,
        laplacian_ksize=laplacian_ksize,
    )


def show_side_by_side(source: np.ndarray, processed: np.ndarray) -> None:
    """
    Render two equal-width columns labeled "Original" and "Processed".
    - Source is always BGR; converts to RGB before st.image.
    - Processed may be single-channel (2-D) or multi-channel (3-D BGR).
      Single-channel images are displayed with channels="GRAY".
    Requirements: 2.1, 2.2, 2.3, 2.5
    """
    col_original, col_processed = st.columns(2)

    # Source is always a BGR color image — convert to RGB for correct display.
    source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    with col_original:
        st.subheader("Original")
        st.image(source_rgb, use_container_width=True)

    with col_processed:
        st.subheader("Processed")
        if processed.ndim == 2:
            # Single-channel (grayscale / edge-detected) — use GRAY colormap.
            st.image(processed, channels="GRAY", use_container_width=True)
        else:
            # Multi-channel BGR — convert to RGB before display.
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            st.image(processed_rgb, use_container_width=True)


def show_download_button(processed: np.ndarray) -> None:
    """
    Render the Download Button. Encodes the processed image as PNG on click.
    Shows st.error if encoding fails; does not initiate a download in that case.
    Requirements: 8.1, 8.2, 8.3, 8.4
    """
    try:
        png_bytes = encode_image_png(processed)
    except RuntimeError as exc:
        st.error(f"Failed to encode image for download: {exc}")
        return

    st.download_button(
        label="Download Processed Image",
        data=png_bytes,
        file_name="processed_image.png",
        mime="image/png",
    )
