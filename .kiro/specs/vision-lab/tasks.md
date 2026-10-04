# Implementation Plan: VisionLab

## Overview

Build VisionLab as a flat, functional Python/Streamlit application composed of four modules: `utils.py` (image I/O), `transforms.py` (pure transform functions + pipeline), `ui.py` (Streamlit rendering helpers), and `app.py` (entry point). Tasks follow the data-flow order so each step integrates immediately into the growing application.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create `requirements.txt` with pinned minimum versions for `streamlit>=1.32.0`, `opencv-python>=4.9.0`, and `numpy>=1.26.0`
  - Create empty stub files: `app.py`, `transforms.py`, `ui.py`, `utils.py`
  - _Requirements: 9.1, 9.2, 9.3_

- [x] 2. Implement image I/O utilities (`utils.py`)
  - [x] 2.1 Implement `decode_image(file_bytes: bytes) -> np.ndarray`
    - Use `cv2.imdecode` on a NumPy byte buffer to produce a BGR `(H, W, 3)` uint8 array
    - Raise `ValueError` if decoding returns `None`
    - _Requirements: 1.2, 1.4_
  - [ ]* 2.2 Write property test for decode/encode round-trip
    - **Property 1: Image decode/encode round-trip**
    - **Validates: Requirements 1.2, 8.2**
    - Use `hypothesis` to generate synthetic PNG/JPEG byte sequences; assert spatial dimensions and channel count are preserved after decode → encode → decode
  - [x] 2.3 Implement `encode_image_png(image: np.ndarray) -> bytes`
    - Use `cv2.imencode(".png", image)` and raise `RuntimeError` if it returns `False`
    - _Requirements: 8.2, 8.3, 8.4_
  - [ ]* 2.4 Write property test for PNG encode/decode round-trip pixel data
    - **Property 7: PNG encode/decode round-trip preserves pixel data**
    - **Validates: Requirements 8.2, 8.3**
    - Generate arbitrary uint8 single-channel and three-channel arrays; assert identical shape and pixel values after encode → decode

- [x] 3. Implement individual transform functions (`transforms.py`)
  - [x] 3.1 Define `PipelineConfig` dataclass with all fields and defaults as specified in the design
    - _Requirements: 7.1_
  - [x] 3.2 Implement `apply_grayscale(image)`
    - Use `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`; return unchanged if already single-channel
    - Output shape `(H, W)`, dtype uint8
    - _Requirements: 4.2, 4.3_
  - [ ]* 3.3 Write property test for grayscale output shape
    - **Property 2: Grayscale output is single-channel**
    - **Validates: Requirements 4.2**
    - Generate arbitrary `(H, W, 3)` uint8 arrays; assert output shape is `(H, W)`
  - [x] 3.4 Implement `apply_gaussian_blur(image, kernel_size)`
    - Apply `cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)`; preserve shape and channel count
    - _Requirements: 5.4, 5.8_
  - [x] 3.5 Implement `apply_median_blur(image, kernel_size)`
    - Apply `cv2.medianBlur(image, kernel_size)`; preserve shape and channel count
    - _Requirements: 5.5, 5.8_
  - [x] 3.6 Implement `apply_bilateral_filter(image, diameter, sigma_color, sigma_space)`
    - Apply `cv2.bilateralFilter`; preserve shape and channel count
    - _Requirements: 5.6, 5.8_
  - [ ]* 3.7 Write property test for blur transforms preserving image shape
    - **Property 4: Blur transforms preserve image shape**
    - **Validates: Requirements 5.4, 5.5, 5.6, 5.8**
    - Generate arbitrary single-channel and multi-channel uint8 arrays with valid parameter ranges; assert output shape equals input shape for all three blur functions
  - [x] 3.8 Implement `apply_canny(image, low, high)`
    - Convert to grayscale if multi-channel; apply `cv2.Canny`; output shape `(H, W)`, dtype uint8
    - _Requirements: 6.4, 6.8_
  - [x] 3.9 Implement `apply_sobel(image, kernel_size)`
    - Convert to grayscale if multi-channel; compute `Gx` and `Gy` with `cv2.Sobel`; compute magnitude, normalize to uint8
    - _Requirements: 6.5, 6.8_
  - [x] 3.10 Implement `apply_laplacian(image, kernel_size)`
    - Convert to grayscale if multi-channel; apply `cv2.Laplacian`; take absolute value, convert to uint8
    - _Requirements: 6.6, 6.8_
  - [ ]* 3.11 Write property test for edge detectors producing single-channel uint8 output
    - **Property 5: Edge detectors produce single-channel uint8 output**
    - **Validates: Requirements 6.4, 6.5, 6.6, 6.8**
    - Generate arbitrary single-channel and multi-channel uint8 arrays; assert output shape is `(H, W)` and dtype is `uint8` for Canny, Sobel, and Laplacian

- [x] 4. Implement the transform pipeline (`transforms.py`)
  - [x] 4.1 Implement `run_pipeline(image, config)`
    - Apply enabled transforms in fixed order: Grayscale → Gaussian Blur → Median Blur → Bilateral Filter → Canny → Sobel → Laplacian
    - Return source image unchanged when all flags are `False`
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  - [ ]* 4.2 Write property test for disabled-transform identity
    - **Property 3: Disabled transform is identity**
    - **Validates: Requirements 4.3, 5.7, 6.7**
    - For each transform function, assert element-wise equality between input and output when the enabled flag is `False`
  - [ ]* 4.3 Write property test for empty pipeline identity
    - **Property 6: Empty pipeline is identity**
    - **Validates: Requirements 2.4, 7.4**
    - Generate arbitrary uint8 arrays; run `run_pipeline` with all-`False` `PipelineConfig`; assert output is element-wise equal to input

- [x] 5. Checkpoint — Ensure all transform and utility tests pass
  - Run the full test suite against `utils.py` and `transforms.py`; ask the user if questions arise.

- [x] 6. Implement Streamlit UI helpers (`ui.py`)
  - [x] 6.1 Implement `render_sidebar() -> PipelineConfig`
    - Render three tabs: "Color", "Blur", "Edges"
    - "Color" tab: checkbox for Grayscale
    - "Blur" tab: toggles and odd-step sliders (`step=2`, starting from 1) for Gaussian Blur, Median Blur, and Bilateral Filter with the ranges specified in requirements 5.1–5.3
    - "Edges" tab: toggles and sliders for Canny (thresholds 0–255), Sobel (kernel 1–31 odd), Laplacian (kernel 1–31 odd) per requirements 6.1–6.3
    - Return a populated `PipelineConfig`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 5.1, 5.2, 5.3, 6.1, 6.2, 6.3_
  - [x] 6.2 Implement `show_side_by_side(source, processed)`
    - Render two equal-width columns labeled "Original" and "Processed"
    - Convert BGR color images to RGB before `st.image`; use grayscale colormap for single-channel processed images
    - _Requirements: 2.1, 2.2, 2.3, 2.5_
  - [x] 6.3 Implement `show_download_button(processed)`
    - Call `encode_image_png(processed)` inside the button callback; deliver bytes as `"processed_image.png"`
    - Call `st.error` if encoding raises `RuntimeError`; do not initiate download
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 7. Implement the application entry point (`app.py`)
  - Wire `file_uploader` (types `["jpg", "jpeg", "png"]`) to `decode_image`; cache result in `st.session_state["source_image"]` keyed by `(filename, size)`
  - Display `st.success` on successful load, `st.error` on `ValueError`, and `st.info` placeholder when no image is present
  - Call `render_sidebar`, `run_pipeline`, `show_side_by_side`, and `show_download_button` in sequence when a source image exists
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.4, 7.2, 9.1_

- [x] 8. Final checkpoint — Ensure all tests pass
  - Run the full test suite; verify the app launches with `streamlit run app.py` without import errors; ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Odd-integer kernel sliders use `st.slider(..., step=2)` starting from 1 — this naturally yields odd values without extra conversion
- Color images must be converted BGR → RGB before `st.image`; single-channel images use `channels="GRAY"` or equivalent
- Property tests use `hypothesis` with `numpy` strategies; add `hypothesis` to `requirements.txt` if property tests are included
- Checkpoints ensure incremental validation at natural module boundaries

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1", "3.1"] },
    { "id": 2, "tasks": ["2.3", "3.2", "3.4", "3.5", "3.6", "3.8", "3.9", "3.10"] },
    { "id": 3, "tasks": ["2.2", "2.4", "3.3", "3.7", "3.11", "4.1"] },
    { "id": 4, "tasks": ["4.2", "4.3", "6.1", "6.2", "6.3"] },
    { "id": 5, "tasks": ["7"] }
  ]
}
```
