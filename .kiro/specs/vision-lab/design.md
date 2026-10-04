# Design Document — VisionLab

## Overview

VisionLab is a single-page Streamlit application for interactive computer vision experimentation. The user uploads an image, configures a sequential transform pipeline through a tabbed sidebar, and views the original and processed images side by side. The architecture is a flat, functional Python module decomposition with no database, no server-side state beyond the Streamlit session, and no external API calls.

---

## Architecture

```
app.py                  ← Streamlit entry point; assembles UI and wires callbacks
transforms.py           ← Pure functions: one function per transform + pipeline runner
ui.py                   ← Reusable Streamlit UI helpers (sidebar tabs, image display)
utils.py                ← Image encode/decode utilities (bytes ↔ NumPy array)
requirements.txt        ← Pinned runtime dependencies
```

All transform logic lives in **`transforms.py`** as pure functions that accept and return NumPy arrays. This makes them independently testable without Streamlit. **`app.py`** is the only file that imports Streamlit directly.

### Data Flow

```
User uploads file bytes
        ↓
utils.decode_image(bytes) → ndarray (Source Image, stored in st.session_state)
        ↓
transforms.run_pipeline(source, config) → ndarray (Processed Image)
        ↓
ui.show_side_by_side(source, processed)
        ↓
utils.encode_image_png(processed) → bytes → st.download_button
```

Streamlit reruns the entire script on every user interaction. The Source Image is cached in `st.session_state["source_image"]` to avoid re-decoding on every rerender. The pipeline is recomputed from scratch on every run using the current slider/toggle values.

---

## Components

### `utils.py`

Stateless image I/O utilities.

```python
def decode_image(file_bytes: bytes) -> np.ndarray:
    """
    Decode raw file bytes (JPEG/PNG) into a BGR NumPy array.
    Returns array of shape (H, W, 3), dtype uint8.
    Raises ValueError if decoding fails.
    """

def encode_image_png(image: np.ndarray) -> bytes:
    """
    Encode a NumPy array (any channel count, uint8) to PNG bytes.
    Raises RuntimeError if encoding fails.
    """
```

### `transforms.py`

Pure image transform functions and pipeline orchestrator.

```python
# --- Individual transforms ---

def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR or single-channel image to single-channel grayscale.
    If already single-channel, returns unchanged.
    Output shape: (H, W), dtype uint8.
    """

def apply_gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply cv2.GaussianBlur with the given odd kernel_size.
    Preserves shape and channel count.
    """

def apply_median_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply cv2.medianBlur with the given odd kernel_size.
    Preserves shape and channel count.
    """

def apply_bilateral_filter(
    image: np.ndarray, diameter: int, sigma_color: float, sigma_space: float
) -> np.ndarray:
    """
    Apply cv2.bilateralFilter.
    Preserves shape and channel count.
    """

def apply_canny(image: np.ndarray, low: int, high: int) -> np.ndarray:
    """
    Apply cv2.Canny edge detection.
    Converts to grayscale first if image is multi-channel.
    Output shape: (H, W), dtype uint8.
    """

def apply_sobel(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Compute Sobel gradient magnitude (sqrt(Gx^2 + Gy^2)), normalized to uint8.
    Converts to grayscale first if image is multi-channel.
    Output shape: (H, W), dtype uint8.
    """

def apply_laplacian(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply cv2.Laplacian, take absolute value, convert to uint8.
    Converts to grayscale first if image is multi-channel.
    Output shape: (H, W), dtype uint8.
    """

# --- Pipeline ---

@dataclass
class PipelineConfig:
    grayscale_enabled: bool = False
    gaussian_enabled: bool = False
    gaussian_ksize: int = 5
    median_enabled: bool = False
    median_ksize: int = 5
    bilateral_enabled: bool = False
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    canny_enabled: bool = False
    canny_low: int = 50
    canny_high: int = 150
    sobel_enabled: bool = False
    sobel_ksize: int = 3
    laplacian_enabled: bool = False
    laplacian_ksize: int = 3

def run_pipeline(image: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """
    Apply enabled transforms in fixed order:
    Grayscale → Gaussian Blur → Median Blur → Bilateral Filter
    → Canny → Sobel → Laplacian.
    Returns the Source Image unchanged if no transforms are enabled.
    """
```

### `ui.py`

Streamlit rendering helpers.

```python
def render_sidebar() -> PipelineConfig:
    """
    Render tabbed sidebar (Color | Blur | Edges) and return a PipelineConfig
    populated from current widget values.
    """

def show_side_by_side(source: np.ndarray, processed: np.ndarray) -> None:
    """
    Render two-column layout with source in left column and processed in right.
    Uses grayscale colormap for single-channel processed images.
    """

def show_download_button(processed: np.ndarray) -> None:
    """
    Render the Download Button. Encodes processed image as PNG on click.
    Shows st.error if encoding fails.
    """
```

### `app.py`

```python
import streamlit as st
from utils import decode_image
from ui import render_sidebar, show_side_by_side, show_download_button
from transforms import run_pipeline

st.set_page_config(page_title="VisionLab", layout="wide")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Decode only when the uploaded file changes (by name+size key)
    file_key = (uploaded.name, uploaded.size)
    if st.session_state.get("source_key") != file_key:
        try:
            st.session_state["source_image"] = decode_image(uploaded.read())
            st.session_state["source_key"] = file_key
            st.success("Image loaded successfully.")
        except ValueError as e:
            st.error(f"Could not load image: {e}")

source = st.session_state.get("source_image")

if source is None:
    st.info("Upload an image to get started.")
else:
    config = render_sidebar()
    processed = run_pipeline(source, config)
    show_side_by_side(source, processed)
    show_download_button(processed)
```

---

## Sidebar Layout

```
Sidebar
├── [Color tab]
│   └── ☐ Grayscale
├── [Blur tab]
│   ├── ☐ Gaussian Blur
│   │   └── Kernel size: ─●──── [1–31, odd]
│   ├── ☐ Median Blur
│   │   └── Kernel size: ─●──── [1–31, odd]
│   └── ☐ Bilateral Filter
│       ├── Diameter: ──●─── [1–20]
│       ├── Sigma Color: ──●─── [10–150]
│       └── Sigma Space: ──●─── [10–150]
└── [Edges tab]
    ├── ☐ Canny
    │   ├── Low threshold: ──●─── [0–255]
    │   └── High threshold: ──●─── [0–255]
    ├── ☐ Sobel
    │   └── Kernel size: ─●──── [1–31, odd]
    └── ☐ Laplacian
        └── Kernel size: ─●──── [1–31, odd]
```

**Odd kernel sliders:** Streamlit sliders do not natively step by 2 over odd values. The pattern is to use a slider with `step=2` starting from 1, which naturally produces odd integers: `st.slider("Kernel size", 1, 31, 5, step=2)`.

---

## Data Models

### Image Representation

All images inside the pipeline are `numpy.ndarray` with `dtype=uint8`.

| State | Shape | Notes |
|---|---|---|
| After decode | `(H, W, 3)` | BGR channel order (OpenCV default) |
| After grayscale | `(H, W)` | Single-channel luminance |
| After blur (color) | `(H, W, 3)` | Shape preserved |
| After blur (gray) | `(H, W)` | Shape preserved |
| After edge detection | `(H, W)` | Always single-channel |

**Display note:** Streamlit's `st.image` expects RGB (not BGR) for color images. Color images must be converted with `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` before passing to `st.image`. Single-channel images are passed with `clamp=True` and `channels="GRAY"` (or equivalent `PIL` conversion).

### Session State Keys

| Key | Type | Description |
|---|---|---|
| `source_image` | `np.ndarray \| None` | Decoded source image |
| `source_key` | `tuple[str, int] \| None` | `(filename, size)` to detect re-uploads |

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Uploaded file is not a valid image | `decode_image` raises `ValueError`; `app.py` calls `st.error(...)` and leaves `source_image` unchanged |
| PNG encoding fails at download | `encode_image_png` raises `RuntimeError`; `show_download_button` calls `st.error(...)` and does not deliver download |
| Edge detector receives multi-channel image | `apply_canny`, `apply_sobel`, `apply_laplacian` silently convert to grayscale before processing |
| Blur applied to single-channel image | OpenCV handles natively; no special branching needed |

---

## Dependencies (`requirements.txt`)

```
streamlit>=1.32.0
opencv-python>=4.9.0
numpy>=1.26.0
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Image decode/encode round-trip

*For any* valid PNG or JPEG image byte sequence, decoding it into a NumPy array and re-encoding as PNG then decoding again SHALL produce an array with the same spatial dimensions (height, width) and the same number of channels as the original decode result.

**Validates: Requirements 1.2, 8.2**

---

### Property 2: Grayscale output is single-channel

*For any* color image (shape `(H, W, 3)`), applying the grayscale transform SHALL produce an array of shape `(H, W)` — that is, a single-channel image with the same height and width.

**Validates: Requirements 4.2**

---

### Property 3: Disabled transform is identity

*For any* image array and any transform (grayscale, Gaussian blur, median blur, bilateral filter, Canny, Sobel, Laplacian), when that transform's enabled flag is `False`, the transform function SHALL return an array that is element-wise equal to its input.

**Validates: Requirements 4.3, 5.7, 6.7**

---

### Property 4: Blur transforms preserve image shape

*For any* image (single-channel or multi-channel) and any valid parameter set within the specified ranges, applying Gaussian blur, median blur, or bilateral filter SHALL produce an output array with the same shape `(H, W[, C])` as the input.

**Validates: Requirements 5.4, 5.5, 5.6, 5.8**

---

### Property 5: Edge detectors produce single-channel uint8 output

*For any* image (single-channel or multi-channel) and any valid parameter set within the specified ranges, applying Canny, Sobel, or Laplacian edge detection SHALL produce an output array with shape `(H, W)` and dtype `uint8`.

**Validates: Requirements 6.4, 6.5, 6.6, 6.8**

---

### Property 6: Empty pipeline is identity

*For any* source image array, running `run_pipeline` with a `PipelineConfig` where all enabled flags are `False` SHALL return an array that is element-wise equal to the source image.

**Validates: Requirements 2.4, 7.4**

---

### Property 7: PNG encode/decode round-trip preserves pixel data

*For any* `uint8` NumPy array (single-channel or three-channel) with valid pixel values in `[0, 255]`, encoding it as PNG bytes via `encode_image_png` and then decoding those bytes via `decode_image` SHALL produce an array with identical shape and pixel values (PNG is lossless).

**Validates: Requirements 8.2, 8.3**
