from dataclasses import dataclass

import cv2
import numpy as np


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


def apply_gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply cv2.GaussianBlur with the given odd kernel_size.
    Preserves shape and channel count.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR or single-channel image to single-channel grayscale.
    If already single-channel, returns unchanged.
    Output shape: (H, W), dtype uint8.
    """
    if image.ndim == 2:
        # Already single-channel — return unchanged
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_median_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply cv2.medianBlur with the given odd kernel_size.
    Preserves shape and channel count (works on both single-channel and
    multi-channel images without any conversion).
    """
    return cv2.medianBlur(image, kernel_size)


def apply_bilateral_filter(
    image: np.ndarray, diameter: int, sigma_color: float, sigma_space: float
) -> np.ndarray:
    """
    Apply cv2.bilateralFilter with the given parameters.
    Preserves shape and channel count.
    """
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def apply_sobel(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Compute Sobel gradient magnitude (sqrt(Gx^2 + Gy^2)), normalized to uint8.
    Converts to grayscale first if image is multi-channel.
    Output shape: (H, W), dtype uint8.
    """
    # Req 6.8: convert to grayscale if multi-channel
    gray = apply_grayscale(image)

    # Compute horizontal and vertical gradients in float64 to avoid overflow
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # Req 6.5: combine into gradient magnitude and normalize to [0, 255] uint8
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    magnitude = np.clip(magnitude / magnitude.max() * 255, 0, 255) if magnitude.max() > 0 else magnitude
    return magnitude.astype(np.uint8)


def apply_canny(image: np.ndarray, low: int, high: int) -> np.ndarray:
    """
    Apply cv2.Canny edge detection.
    Converts to grayscale first if image is multi-channel.
    Output shape: (H, W), dtype uint8.
    """
    if image.ndim != 2:
        image = apply_grayscale(image)
    return cv2.Canny(image, low, high)


def apply_laplacian(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply cv2.Laplacian, take absolute value, convert to uint8.
    Converts to grayscale first if image is multi-channel.
    Output shape: (H, W), dtype uint8.
    """
    if image.ndim != 2:
        image = apply_grayscale(image)
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
    return cv2.convertScaleAbs(laplacian)


def run_pipeline(image: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """
    Apply enabled transforms in fixed order:
    Grayscale → Gaussian Blur → Median Blur → Bilateral Filter
    → Canny → Sobel → Laplacian.
    Returns the source image unchanged if no transforms are enabled.
    """
    result = image

    # Req 7.1: fixed execution order
    if config.grayscale_enabled:
        result = apply_grayscale(result)

    if config.gaussian_enabled:
        result = apply_gaussian_blur(result, config.gaussian_ksize)

    if config.median_enabled:
        result = apply_median_blur(result, config.median_ksize)

    if config.bilateral_enabled:
        result = apply_bilateral_filter(
            result,
            config.bilateral_d,
            config.bilateral_sigma_color,
            config.bilateral_sigma_space,
        )

    if config.canny_enabled:
        result = apply_canny(result, config.canny_low, config.canny_high)

    if config.sobel_enabled:
        result = apply_sobel(result, config.sobel_ksize)

    if config.laplacian_enabled:
        result = apply_laplacian(result, config.laplacian_ksize)

    # Req 7.4: all flags False → return source image unchanged
    return result
