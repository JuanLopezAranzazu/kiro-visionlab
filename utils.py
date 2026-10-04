import cv2
import numpy as np


def decode_image(file_bytes: bytes) -> np.ndarray:
    """
    Decode raw file bytes (JPEG/PNG) into a BGR NumPy array.
    Returns array of shape (H, W, 3), dtype uint8.
    Raises ValueError if decoding fails.
    """
    if not file_bytes:
        raise ValueError(
            "Failed to decode image: received empty byte string. "
            "Accepted formats: JPEG, JPG, PNG."
        )
    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            "Failed to decode image. Accepted formats: JPEG, JPG, PNG."
        )
    return image


def encode_image_png(image: np.ndarray) -> bytes:
    """
    Encode a NumPy array (any channel count, uint8) to PNG bytes.
    Raises RuntimeError if encoding fails.
    """
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode image as PNG.")
    return buffer.tobytes()
